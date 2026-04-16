//! Atlas API server — HTTP/1.1 listener, thread pool, request dispatch.
//!
//! Uses std::net::TcpListener for connection acceptance.
//! Each connection is handled in a dedicated OS thread.
//! The inference state (model + tokenizer) is shared via Arc<Mutex<>>.

use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use atlas_model::{ModelConfig, load_model_from_safetensors};
use atlas_tokenize::Tokenizer;

use crate::handler::{handle, parse_http_request, InferState};
use crate::types::ServerConfig;

/// The ATLAS OpenAI-compatible API server.
pub struct ApiServer {
    cfg: ServerConfig,
}

impl ApiServer {
    /// Create a new server with the given config.
    pub fn new(cfg: ServerConfig) -> Self {
        Self { cfg }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(ServerConfig::default())
    }

    /// Load model and tokenizer, bind port, and start serving.
    /// This call blocks until the server exits (SIGINT).
    pub fn serve(&self) -> std::io::Result<()> {
        let addr = format!("{}:{}", self.cfg.host, self.cfg.port);

        // ── Load inference state ─────────────────────────────────────────
        let state = Arc::new(Mutex::new(self.load_infer_state()));

        // ── Bind TCP listener ────────────────────────────────────────────
        let listener = TcpListener::bind(&addr)?;
        listener.set_nonblocking(false)?;

        eprintln!("┌─ atlas-api serving ─────────────────────────────────────────");
        eprintln!("│  address  : http://{addr}");
        eprintln!("│  model    : {}", self.cfg.model_id);
        eprintln!("│  weights  : {}", self.cfg.weights_dir.as_deref().unwrap_or("(none — echo mode)"));
        eprintln!("│  max_tok  : {}", self.cfg.max_tokens);
        eprintln!("│");
        eprintln!("│  Endpoints:");
        eprintln!("│    GET  /health");
        eprintln!("│    GET  /v1/models");
        eprintln!("│    POST /v1/chat/completions");
        eprintln!("│    POST /v1/completions");
        eprintln!("│");
        eprintln!("│  OpenAI base URL: http://{addr}/v1");
        eprintln!("│  Press Ctrl+C to stop.");
        eprintln!("└─────────────────────────────────────────────────────────────");

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let state_clone = Arc::clone(&state);
                    let max_tokens  = self.cfg.max_tokens;
                    thread::spawn(move || {
                        handle_connection(stream, state_clone, max_tokens);
                    });
                }
                Err(e) => {
                    eprintln!("atlas-api: accept error: {e}");
                }
            }
        }

        Ok(())
    }

    /// Load the inference state (model + tokenizer) based on config.
    pub fn load_infer_state(&self) -> InferState {
        let (model, tokenizer) = if let Some(ref weights_dir) = self.cfg.weights_dir {
            eprintln!("atlas-api: loading model from {weights_dir}…");
            let tok_path = format!("{}/tokenizer.json", weights_dir.trim_end_matches('/'));
            let tokenizer = match Tokenizer::from_file(&tok_path) {
                Ok(t) => {
                    eprintln!("  tokenizer: ✓ {} tokens", t.vocab_size());
                    Some(t)
                }
                Err(e) => {
                    eprintln!("  tokenizer: ⚠ {e} — byte fallback");
                    None
                }
            };

            let weights_path = format!("{}/model.safetensors", weights_dir.trim_end_matches('/'));
            let cfg = infer_model_config(&self.cfg.model_id);
            let model = match load_model_from_safetensors(&weights_path, cfg) {
                Ok(m) => {
                    eprintln!("  model: ✓ {} M params", m.param_count() / 1_000_000);
                    Some(m)
                }
                Err(e) => {
                    eprintln!("  model: ⚠ {e} — echo mode");
                    None
                }
            };
            (model, tokenizer)
        } else {
            eprintln!("atlas-api: no weights dir — running in echo mode");
            (None, None)
        };

        InferState {
            model,
            tokenizer,
            model_id: self.cfg.model_id.clone(),
        }
    }
}

/// Infer ModelConfig from model ID string.
fn infer_model_config(model_id: &str) -> ModelConfig {
    let id_lower = model_id.to_lowercase();
    if id_lower.contains("135m") {
        ModelConfig::smollm2_135m()
    } else if id_lower.contains("1.7b") || id_lower.contains("1b7") {
        ModelConfig::smollm2_1b7()
    } else if id_lower.contains("olmo") && id_lower.contains("7b") {
        ModelConfig::olmo3_7b()
    } else if id_lower.contains("olmo") || id_lower.contains("1b") {
        ModelConfig::olmo3_1b()
    } else if id_lower.contains("llama") {
        ModelConfig::llama32_1b()
    } else {
        // Default: smollm2-135m (smallest, always loads fast)
        ModelConfig::smollm2_135m()
    }
}

/// Handle a single TCP connection: read request, dispatch, write response.
fn handle_connection(
    mut stream: TcpStream,
    state: Arc<Mutex<InferState>>,
    _max_tokens: usize,
) {
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(120))).ok();

    // Read the full HTTP request
    let mut raw = Vec::with_capacity(8192);
    let mut buf = [0u8; 8192];

    // Read until we have the full headers (and body based on Content-Length)
    let _header_end = loop {
        match stream.read(&mut buf) {
            Ok(0) => return, // Connection closed
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                // Check if we have the end of headers
                if let Some(sep) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    // Check Content-Length to see if we need to read more body
                    let header_str = std::str::from_utf8(&raw[..sep]).unwrap_or("");
                    let content_length: usize = header_str.lines()
                        .find(|l| l.to_lowercase().starts_with("content-length:"))
                        .and_then(|l| l[15..].trim().parse().ok())
                        .unwrap_or(0);
                    let body_received = raw.len().saturating_sub(sep + 4);
                    if body_received >= content_length {
                        break sep;
                    }
                    // Need to read more body
                    while raw.len() < sep + 4 + content_length {
                        match stream.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => raw.extend_from_slice(&buf[..n]),
                            Err(_) => break,
                        }
                    }
                    break sep;
                }
                // Safety: don't read more than 10MB
                if raw.len() > 10 * 1024 * 1024 { return; }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break 0,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut   => break 0,
            Err(_) => return,
        }
    };

    if let Some((method, path, _headers, body)) = parse_http_request(&raw) {
        handle(&mut stream, &method, &path, &body, &state);
    }
    // Connection closes when TcpStream drops
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ServerConfig;

    #[test]
    fn server_config_default() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.max_tokens, 2048);
    }

    #[test]
    fn infer_model_config_135m() {
        let cfg = infer_model_config("atlas-smollm2-135m");
        assert_eq!(cfg.n_layers, 30); // smollm2-135m has 30 layers
    }

    #[test]
    fn infer_model_config_default() {
        let cfg = infer_model_config("atlas-custom");
        // Should fall back to smollm2_135m
        assert!(cfg.vocab_size > 0);
    }

    #[test]
    fn load_infer_state_no_weights() {
        let server = ApiServer::new(ServerConfig {
            weights_dir: None,
            model_id: "test-model".to_string(),
            ..ServerConfig::default()
        });
        let state = server.load_infer_state();
        assert!(state.model.is_none());
        assert!(state.tokenizer.is_none());
        assert_eq!(state.model_id, "test-model");
    }

    #[test]
    fn api_server_new() {
        let server = ApiServer::new(ServerConfig::default());
        assert_eq!(server.cfg.port, 8080);
    }
}
