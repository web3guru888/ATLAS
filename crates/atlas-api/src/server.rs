//! ATLAS API server — TCP listener, connection dispatch, model loading.
//!
//! Connections handled sequentially on the main thread. The inference state (model + tokenizer)
//! is shared via `Arc<Mutex<InferState>>`. Connections block during inference
//! (serialised through the mutex), which is safe and correct for a single-GPU
//! scenario where the GPU is also the bottleneck.
//!
//! # Usage
//!
//! ```no_run
//! use atlas_api::{ApiServer, types::ServerConfig};
//! let cfg = ServerConfig { port: 8080, ..ServerConfig::default() };
//! ApiServer::new(cfg).serve().unwrap(); // blocks
//! ```

use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
// thread import removed — connections handled on main thread (GPU inference is single-threaded)
use std::time::Duration;

use atlas_infer::InferEngine;
use atlas_model::{ModelConfig, OlmoModel, load_model_from_safetensors, load_model_from_dir};
use atlas_tokenize::Tokenizer;

use crate::handler::{handle, parse_http_request, InferState};
use crate::types::{ChatTemplate, ServerConfig};

/// The ATLAS OpenAI-compatible HTTP API server.
pub struct ApiServer {
    cfg: ServerConfig,
}

impl ApiServer {
    /// Create a new server with the given [`ServerConfig`].
    pub fn new(cfg: ServerConfig) -> Self {
        Self { cfg }
    }

    /// Create a server with all default settings (port 8080, no model loaded).
    pub fn with_defaults() -> Self {
        Self::new(ServerConfig::default())
    }

    /// Return the bind address string ("host:port").
    pub fn addr(&self) -> String {
        format!("{}:{}", self.cfg.host, self.cfg.port)
    }

    /// Load model/tokenizer, bind the TCP port, and serve until interrupted.
    ///
    /// This call **blocks** until the process receives SIGINT or the listener fails.
    pub fn serve(&self) -> std::io::Result<()> {
        let addr = self.addr();

        // Build shared inference state
        let state = Arc::new(Mutex::new(self.load_infer_state()));

        // Bind
        let listener = TcpListener::bind(&addr)?;

        eprintln!("┌─ atlas-api ─────────────────────────────────────────────────");
        eprintln!("│  address  : http://{addr}");
        eprintln!("│  model    : {}", self.cfg.model_id);
        eprintln!("│  weights  : {}", self.cfg.weights_dir.as_deref().unwrap_or("(none — echo mode)"));
        eprintln!("│  max_tok  : {}", self.cfg.max_tokens);
        let olmo3 = atlas_model::SamplingConfig::olmo3();
        eprintln!("│  defaults : rep_pen={:.1} window={} freq_pen={:.1} top_p={:.2} top_k={} min_p={:.2}",
            olmo3.repetition_penalty, olmo3.repetition_window,
            olmo3.frequency_penalty, olmo3.top_p, olmo3.top_k, olmo3.min_p);
        eprintln!("│");
        eprintln!("│  OpenAI base URL  : http://{addr}/v1");
        eprintln!("│  Endpoints:");
        eprintln!("│    GET  /health");
        eprintln!("│    GET  /v1/models");
        eprintln!("│    POST /v1/chat/completions");
        eprintln!("│    POST /v1/completions");
        eprintln!("│");
        eprintln!("│  Press Ctrl+C to stop.");
        eprintln!("└─────────────────────────────────────────────────────────────");

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    // Handle on main thread — GPU inference is single-threaded
                    // anyway (Mutex serialises), and spawning threads caused
                    // garbage output from CUDA context issues.
                    handle_connection(stream, Arc::clone(&state));
                }
                Err(e) => {
                    eprintln!("atlas-api: accept error: {e}");
                }
            }
        }
        Ok(())
    }

    // ─── Private helpers ──────────────────────────────────────────────────

    /// Load model and tokenizer from `weights_dir`, or return an echo-mode state.
    pub(crate) fn load_infer_state(&self) -> InferState {
        if let Some(ref weights_dir) = self.cfg.weights_dir {
            eprintln!("atlas-api: loading model from {weights_dir} …");

            let tok_path = format!("{}/tokenizer.json", weights_dir.trim_end_matches('/'));
            let tokenizer = match Tokenizer::from_file(&tok_path) {
                Ok(t)  => { eprintln!("  tokenizer: ✓ {} tokens", t.vocab_size()); Some(t) }
                Err(e) => { eprintln!("  tokenizer: ⚠ {e} — byte fallback"); None }
            };

            let dir = weights_dir.trim_end_matches('/');
            let index_path = format!("{dir}/model.safetensors.index.json");
            let cfg   = model_config_from_id(&self.cfg.model_id);
            let mut raw_model: Option<OlmoModel> = if std::path::Path::new(&index_path).exists() {
                // Sharded model (OLMo-2/3 7B etc.) — use index-based dir loader
                eprintln!("  model: sharded index detected — using load_model_from_dir");
                match load_model_from_dir(dir, cfg) {
                    Ok(m)  => { eprintln!("  model: ✓ {} M params (sharded)", m.param_count() / 1_000_000); Some(m) }
                    Err(e) => { eprintln!("  model: ⚠ {e} — echo mode"); None }
                }
            } else {
                // Single-file model (SmolLM2, TinyLlama etc.)
                let weights_path = format!("{dir}/model.safetensors");
                match load_model_from_safetensors(&weights_path, cfg) {
                    Ok(m)  => { eprintln!("  model: ✓ {} M params", m.param_count() / 1_000_000); Some(m) }
                    Err(e) => { eprintln!("  model: ⚠ {e} — echo mode"); None }
                }
            };

            // Wire EOS token → model so generate() stops naturally.
            // Priority: config.json (parsed during load_model_from_dir) > tokenizer.json
            if let (Some(ref tok), Some(ref mut mdl)) = (&tokenizer, &mut raw_model) {
                if mdl.eos_token_id.is_none() {
                    // config.json didn't have it — try tokenizer
                    mdl.eos_token_id = tok.eos_token_id;
                }
                if let Some(eos) = mdl.eos_token_id {
                    eprintln!("  eos_token_id: {eos} (\"{}\")", tok.id_to_token(eos));
                }
                // ChatML models (OLMo-3, Qwen, etc.) use <|im_end|> to end assistant turns.
                // Add it as an extra stop token so generation stops at end of turn.
                if let Some(im_end_id) = tok.token_to_id("<|im_end|>") {
                    let eos_id = mdl.eos_token_id.unwrap_or(u32::MAX);
                    if im_end_id != eos_id {
                        mdl.extra_stop_tokens.push(im_end_id);
                        eprintln!("  extra_stop: {im_end_id} (\"<|im_end|>\")");
                    }
                }
            }

            // Wrap OlmoModel in InferEngine (adds StigmergicHook support).
            // No hook attached by default — zero overhead on the hot path.
            let engine: Option<InferEngine> = raw_model.map(InferEngine::new);

            // Auto-detect chat template from tokenizer special tokens.
            let chat_template = detect_chat_template(&tokenizer, &self.cfg.model_id);
            eprintln!("  chat_template: {:?}", chat_template);

            InferState { model: engine, tokenizer, model_id: self.cfg.model_id.clone(), chat_template }
        } else {
            eprintln!("atlas-api: no --weights — running in echo mode");
            InferState { model: None, tokenizer: None, model_id: self.cfg.model_id.clone(), chat_template: ChatTemplate::default() }
        }
    }
}

/// Pick a `ModelConfig` based on the model-id string.
pub fn model_config_from_id(id: &str) -> ModelConfig {
    let id_lc = id.to_lowercase();
    if id_lc.contains("135m") {
        ModelConfig::smollm2_135m()
    } else if id_lc.contains("1.7b") || id_lc.contains("1b7") {
        ModelConfig::smollm2_1b7()
    } else if id_lc.contains("olmo") && id_lc.contains("7b") {
        ModelConfig::olmo3_actual_7b()
    } else if id_lc.contains("olmo") {
        ModelConfig::olmo3_1b()
    } else if id_lc.contains("llama") {
        ModelConfig::llama32_1b()
    } else {
        ModelConfig::smollm2_135m() // safe default
    }
}

/// Detect the chat template format from tokenizer special tokens.
///
/// Priority: tokenizer special tokens > model_id heuristic > default (ChatML).
fn detect_chat_template(tokenizer: &Option<Tokenizer>, model_id: &str) -> ChatTemplate {
    if let Some(ref tok) = tokenizer {
        // ChatML: OLMo-3, SmolLM2-Instruct, Qwen use <|im_start|>/<|im_end|>
        if tok.special_token_id("<|im_start|>").is_some() {
            return ChatTemplate::ChatML;
        }
        // Llama-3: uses <|start_header_id|>/<|end_header_id|>
        if tok.special_token_id("<|start_header_id|>").is_some() {
            return ChatTemplate::Llama3;
        }
    }
    // Fallback: heuristic from model_id string
    let id_lc = model_id.to_lowercase();
    if id_lc.contains("llama") && id_lc.contains("3") {
        return ChatTemplate::Llama3;
    }
    // Default to ChatML — most modern HF models use it.
    ChatTemplate::ChatML
}

/// Read a full HTTP/1.1 request from the TCP stream, respecting Content-Length.
fn read_request(stream: &mut TcpStream) -> Option<Vec<u8>> {
    let mut raw = Vec::with_capacity(8192);
    let mut buf = [0u8; 8192];

    // Read until we have headers + full body.
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);

                if let Some(sep) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    let header_str = std::str::from_utf8(&raw[..sep]).unwrap_or("");
                    let content_length: usize = header_str
                        .lines()
                        .find(|l| l.to_lowercase().starts_with("content-length:"))
                        .and_then(|l| l[15..].trim().parse().ok())
                        .unwrap_or(0);

                    let body_received = raw.len().saturating_sub(sep + 4);
                    if body_received >= content_length {
                        return Some(raw);
                    }
                    // Read remaining body bytes.
                    let still_needed = content_length - body_received;
                    let mut leftover = vec![0u8; still_needed];
                    let mut got = 0;
                    while got < still_needed {
                        match stream.read(&mut leftover[got..]) {
                            Ok(0) => break,
                            Ok(n) => got += n,
                            Err(_) => break,
                        }
                    }
                    raw.extend_from_slice(&leftover[..got]);
                    return Some(raw);
                }

                // Safety cap: 10 MB
                if raw.len() > 10 * 1024 * 1024 { return None; }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut   => break,
            Err(_) => return None,
        }
    }

    if raw.is_empty() { None } else { Some(raw) }
}

/// Handle one accepted TCP connection end-to-end.
fn handle_connection(mut stream: TcpStream, state: Arc<Mutex<InferState>>) {
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    // Inference can be slow — allow up to 5 minutes for the write side.
    stream.set_write_timeout(Some(Duration::from_secs(300))).ok();

    if let Some(raw) = read_request(&mut stream) {
        if let Some((method, path, _headers, body)) = parse_http_request(&raw) {
            handle(&mut stream, &method, &path, &body, &state);
        }
    }
    // TcpStream drop closes the connection.
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ServerConfig;

    #[test]
    fn server_addr() {
        let srv = ApiServer::new(ServerConfig { host: "127.0.0.1".to_string(), port: 9999, ..ServerConfig::default() });
        assert_eq!(srv.addr(), "127.0.0.1:9999");
    }

    #[test]
    fn model_config_135m() {
        let cfg = model_config_from_id("smollm2-135m");
        assert_eq!(cfg.n_layers, 30);
    }

    #[test]
    fn model_config_1b7() {
        let cfg = model_config_from_id("smollm2-1.7b");
        assert_eq!(cfg.n_layers, 24); // smollm2-1.7b has 24 layers, d_model=2048
    }

    #[test]
    fn model_config_olmo_1b() {
        let cfg = model_config_from_id("olmo3-1b");
        assert!(cfg.vocab_size > 0);
    }

    #[test]
    fn model_config_llama() {
        let cfg = model_config_from_id("llama32-1b");
        assert!(cfg.vocab_size > 0);
    }

    #[test]
    fn model_config_unknown_falls_back() {
        let cfg = model_config_from_id("my-custom-model");
        // Should fall back to smollm2_135m
        assert_eq!(cfg.n_layers, 30);
    }

    #[test]
    fn load_infer_state_no_weights() {
        let srv = ApiServer::new(ServerConfig { weights_dir: None, model_id: "test-model".to_string(), ..ServerConfig::default() });
        let st  = srv.load_infer_state();
        assert!(st.model.is_none());
        assert!(st.tokenizer.is_none());
        assert_eq!(st.model_id, "test-model");
    }

    #[test]
    fn load_infer_state_bad_weights_path() {
        // A non-existent path should not panic — fall back to echo mode.
        let srv = ApiServer::new(ServerConfig {
            weights_dir: Some("/nonexistent/path".to_string()),
            model_id: "smollm2-135m".to_string(),
            ..ServerConfig::default()
        });
        let st = srv.load_infer_state();
        // tokenizer and model will both fail gracefully → None
        assert!(st.model.is_none());
    }

    #[test]
    fn detect_template_no_tokenizer_defaults_chatml() {
        let t = detect_chat_template(&None, "my-model");
        assert_eq!(t, ChatTemplate::ChatML);
    }

    #[test]
    fn detect_template_llama3_from_id() {
        let t = detect_chat_template(&None, "llama-3-8b");
        assert_eq!(t, ChatTemplate::Llama3);
    }

    #[test]
    fn api_server_new() {
        let srv = ApiServer::new(ServerConfig::default());
        assert_eq!(srv.cfg.port, 8080);
        assert_eq!(srv.cfg.host, "0.0.0.0");
    }
}
