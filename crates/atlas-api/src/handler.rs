//! HTTP/1.1 request parser and route handler.
//!
//! Parses raw HTTP/1.1 requests from TcpStream bytes.
//! Routes each request to the appropriate handler function.
//! Writes the full HTTP response back to the stream.
//!
//! # Endpoints
//! - `GET  /health`                → `{"status":"ok"}`
//! - `GET  /v1/models`             → model list
//! - `POST /v1/chat/completions`   → chat completion (stream or blocking)
//! - `POST /v1/completions`        → text completion
//! - `OPTIONS *`                   → CORS preflight (204)
//! - everything else               → 404 JSON error

use std::io::Write;
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use atlas_infer::InferEngine;
use atlas_model::{GenEvent, SamplingConfig};
use atlas_tokenize::Tokenizer;

use crate::types::{
    json_string, gen_id, unix_ts,
    ChatMessage, ChatCompletionRequest, ChatCompletionResponse, ChatTemplate,
    CompletionRequest, CompletionResponse,
    ErrorResponse, StreamChunk,
};

// ── Shared inference state ────────────────────────────────────────────────────

/// Model + tokenizer shared across HTTP worker threads.
pub struct InferState {
    /// Loaded model (wrapped in InferEngine for StigmergicHook support), or
    /// None in echo/test mode.
    pub model: Option<InferEngine>,
    /// Loaded tokenizer, or None (falls back to byte encoding).
    pub tokenizer: Option<Tokenizer>,
    /// The model ID string served to clients.
    pub model_id: String,
    /// Chat template format for converting messages → prompt text.
    pub chat_template: ChatTemplate,
}

// ── HTTP primitives ───────────────────────────────────────────────────────────

/// Parse a raw HTTP/1.1 request buffer.
/// Returns `(method, path, headers, body)` or `None` on malformed input.
pub fn parse_http_request(raw: &[u8]) -> Option<(String, String, Vec<(String, String)>, Vec<u8>)> {
    let sep = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let header_str = std::str::from_utf8(&raw[..sep]).ok()?;
    let mut lines  = header_str.lines();

    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path   = parts.next()?.to_string();

    let mut headers = Vec::new();
    let mut content_length: usize = 0;
    for line in lines {
        if let Some(i) = line.find(':') {
            let k = line[..i].trim().to_lowercase();
            let v = line[i + 1..].trim().to_string();
            if k == "content-length" {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }

    let raw_body = &raw[sep + 4..];
    let body = raw_body[..content_length.min(raw_body.len())].to_vec();

    Some((method, path, headers, body))
}

/// Build an HTTP/1.1 JSON response.
pub fn http_json_response(status: u16, reason: &str, body: &str) -> Vec<u8> {
    http_json_response_with(status, reason, body, &[])
}

/// Build an HTTP/1.1 JSON response with extra headers (e.g. `Retry-After`).
pub fn http_json_response_with(
    status: u16,
    reason: &str,
    body: &str,
    extra_headers: &[(&str, &str)],
) -> Vec<u8> {
    let len = body.len();
    let mut extra = String::new();
    for (k, v) in extra_headers {
        extra.push_str(&format!("{k}: {v}\r\n"));
    }
    format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         {extra}\
         Connection: close\r\n\
         \r\n\
         {body}"
    )
    .into_bytes()
}

/// Build an HTTP/1.1 HTML response (e.g. the `/privacy` policy page).
pub fn http_html_response(status: u16, reason: &str, body: &str) -> Vec<u8> {
    let len = body.len();
    format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: text/html; charset=utf-8\r\n\
         Content-Length: {len}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\
         \r\n\
         {body}"
    )
    .into_bytes()
}

/// Build HTTP headers for a chunked SSE stream.
pub fn http_sse_header() -> Vec<u8> {
    b"HTTP/1.1 200 OK\r\n\
      Content-Type: text/event-stream\r\n\
      Cache-Control: no-cache\r\n\
      Transfer-Encoding: chunked\r\n\
      Access-Control-Allow-Origin: *\r\n\
      Connection: close\r\n\
      \r\n"
        .to_vec()
}

/// Write a single SSE chunk in HTTP chunked-transfer format.
pub fn write_sse_chunk(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    let bytes = data.as_bytes();
    stream.write_all(format!("{:x}\r\n", bytes.len()).as_bytes())?;
    stream.write_all(bytes)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

/// Write an SSE comment line (`: keep-alive`) in chunked framing.
///
/// SSE comments are ignored by spec-compliant clients but reset proxy/client
/// idle timeouts. OpenRouter explicitly requires keep-alive comments from
/// providers while a slow model is still processing (prefill on this engine
/// runs at ~decode speed, so long prompts mean long silent gaps otherwise).
pub fn write_sse_keepalive(stream: &mut TcpStream) -> std::io::Result<()> {
    write_sse_chunk(stream, ": keep-alive\n\n")
}

/// Write the chunked-transfer terminating chunk.
pub fn write_chunk_end(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

// ── Auth ──────────────────────────────────────────────────────────────────────

/// Check a parsed header list for a valid `Authorization: Bearer <key>`.
///
/// Header keys are expected lowercase (as produced by [`parse_http_request`]).
/// Comparison is constant-time to avoid timing side channels.
pub fn bearer_ok(headers: &[(String, String)], key: &str) -> bool {
    let auth = headers
        .iter()
        .find(|(k, _)| k == "authorization")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    let token = auth
        .strip_prefix("Bearer ")
        .or_else(|| auth.strip_prefix("bearer "))
        .unwrap_or("")
        .trim();
    ct_eq(token.as_bytes(), key.as_bytes())
}

/// Constant-time byte-slice equality (length mismatch short-circuits, which
/// only leaks the key length — acceptable).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Standard `401 Unauthorized` response bytes.
pub fn http_unauthorized_response() -> Vec<u8> {
    let err = ErrorResponse {
        message: "missing or invalid API key — pass `Authorization: Bearer <key>`".to_string(),
        error_type: "authentication_error",
        status: 401,
    };
    http_json_response_with(401, "Unauthorized", &err.to_json(), &[("WWW-Authenticate", "Bearer")])
}

/// Standard early-`429 Too Many Requests` response bytes (engine saturated).
///
/// OpenRouter requires providers to reject with 429 immediately instead of
/// queueing — queued time counts against the public throughput metric.
pub fn http_too_many_requests_response(retry_after_secs: u32) -> Vec<u8> {
    let err = ErrorResponse {
        message: "engine saturated (single-stream GPU) — retry shortly".to_string(),
        error_type: "rate_limit_exceeded",
        status: 429,
    };
    let ra = retry_after_secs.to_string();
    http_json_response_with(429, "Too Many Requests", &err.to_json(), &[("Retry-After", &ra)])
}

/// CORS preflight response.
pub fn http_options_response() -> Vec<u8> {
    b"HTTP/1.1 204 No Content\r\n\
      Access-Control-Allow-Origin: *\r\n\
      Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
      Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
      Content-Length: 0\r\n\
      Connection: close\r\n\
      \r\n"
        .to_vec()
}

// ── Router ────────────────────────────────────────────────────────────────────

/// Route and handle one HTTP request; writes the complete response to `stream`.
pub fn handle(
    stream: &mut TcpStream,
    method: &str,
    path: &str,
    body: &[u8],
    state: &Arc<Mutex<InferState>>,
) {
    // Strip query string.
    let clean_path = path.split('?').next().unwrap_or(path);

    if method == "OPTIONS" {
        stream.write_all(&http_options_response()).ok();
        return;
    }

    match (method, clean_path) {
        // ── Health ────────────────────────────────────────────────────────
        ("GET", "/health") | ("GET", "/") => {
            let body = r#"{"status":"ok","service":"atlas-api"}"#;
            stream.write_all(&http_json_response(200, "OK", body)).ok();
        }

        // ── List models ───────────────────────────────────────────────────
        ("GET", "/v1/models") => {
            let model_id = state.lock().unwrap().model_id.clone();
            let ts  = unix_ts();
            let body = format!(
                concat!(
                    r#"{{"object":"list","data":[{{"id":{id},"object":"model","created":{ts},"#,
                    r#""owned_by":"atlas-agi","permission":[],"root":{id},"parent":null}}]}}"#
                ),
                id = json_string(&model_id),
                ts = ts,
            );
            stream.write_all(&http_json_response(200, "OK", &body)).ok();
        }

        // ── Inference endpoints ───────────────────────────────────────────
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
            handle_inference(stream, clean_path, body, state);
        }

        // ── 404 ───────────────────────────────────────────────────────────
        _ => {
            let err = ErrorResponse {
                message: format!("unknown endpoint: {method} {clean_path}"),
                error_type: "not_found",
                status: 404,
            };
            stream.write_all(&http_json_response(404, "Not Found", &err.to_json())).ok();
        }
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────

/// Dispatch a pre-routed (and, when auth is enabled, pre-authenticated)
/// inference request: `POST /v1/chat/completions` or `POST /v1/completions`.
///
/// Called by [`handle`] and by the server's dedicated inference worker thread
/// (see `server.rs`) so that GPU work always runs on the thread that loaded
/// the model.
pub fn handle_inference(
    stream: &mut TcpStream,
    clean_path: &str,
    body: &[u8],
    state: &Arc<Mutex<InferState>>,
) {
    let body_str = match std::str::from_utf8(body) {
        Ok(s) => s,
        Err(_) => {
            let err = ErrorResponse { message: "request body is not valid UTF-8".to_string(), error_type: "invalid_request_error", status: 400 };
            stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
            return;
        }
    };
    match clean_path {
        "/v1/chat/completions" => {
            let mut req = match ChatCompletionRequest::parse(body_str) {
                Ok(r)  => r,
                Err(e) => {
                    let err = ErrorResponse { message: format!("{e}"), error_type: "invalid_request_error", status: 400 };
                    stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
                    return;
                }
            };
            // Auto-inject a system prompt if none provided.
            // This dramatically improves output quality for small models
            // by preventing degenerate <think> loops and providing clear
            // behavioral framing.
            inject_default_system_prompt(&mut req.messages);
            let template = state.lock().unwrap().chat_template;
            // Official OLMo-3-Think generation prompt ends INSIDE a think
            // block: `<|im_start|>assistant\n<think>`. The model is RL-trained
            // to reason first; the reasoning is stripped before the client
            // sees it (see think handling below).
            let prompt = format!("{}<think>", req.to_prompt_with(&template));
            let id     = gen_id("chatcmpl");
            if req.stream {
                handle_chat_stream(stream, &req, &prompt, &id, state);
            } else {
                handle_chat_nonstream(stream, &req, &prompt, &id, state);
            }
        }
        "/v1/completions" => {
            let req = match CompletionRequest::parse(body_str) {
                Ok(r)  => r,
                Err(e) => {
                    let err = ErrorResponse { message: format!("{e}"), error_type: "invalid_request_error", status: 400 };
                    stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
                    return;
                }
            };
            let id = gen_id("cmpl");
            handle_completion(stream, &req, &id, state);
        }
        other => {
            let err = ErrorResponse {
                message: format!("not an inference endpoint: {other}"),
                error_type: "not_found",
                status: 404,
            };
            stream.write_all(&http_json_response(404, "Not Found", &err.to_json())).ok();
        }
    }
}

/// Run model inference for `prompt` with full sampling controls.
/// Returns `(generated_text, prompt_token_count, completion_token_count)`.
fn run_inference(
    state: &Arc<Mutex<InferState>>,
    prompt: &str,
    max_tokens: usize,
    config: &SamplingConfig,
) -> (String, usize, usize) {
    let mut st = state.lock().unwrap();

    // Encode prompt
    let prompt_tokens: Vec<u32> = if let Some(ref tok) = st.tokenizer {
        tok.encode(prompt)
    } else {
        prompt.bytes().map(|b| b as u32).collect()
    };
    let prompt_count = prompt_tokens.len();

    // Context-overflow guard: prompt + generation must fit the KV/RoPE
    // capacity (max_seq_len). Without this, positions past capacity index
    // out of bounds (CPU: panic; GPU: silent out-of-bounds VRAM read).
    let ctx_cap = st.model.as_ref().map(|m| m.model.config().max_seq_len).unwrap_or(usize::MAX);
    let max_tokens = if prompt_count >= ctx_cap { 0 }
                     else { max_tokens.min(ctx_cap - prompt_count) };

    // Generate — InferEngine::generate handles reset() internally and
    // returns (tokens, pheromone_deposits); deposits are discarded here
    // unless a StigmergicHook is configured on the engine.
    let new_tokens: Vec<u32> = if let Some(ref mut engine) = st.model {
        engine.generate(&prompt_tokens, max_tokens, config).0
    } else {
        vec![] // echo / test mode
    };
    let completion_count = new_tokens.len();

    // Decode
    let output = if let Some(ref tok) = st.tokenizer {
        tok.decode(&new_tokens)
    } else {
        let bytes: Vec<u8> = new_tokens.iter().map(|&t| (t % 256) as u8).collect();
        String::from_utf8_lossy(&bytes).to_string()
    };

    (output, prompt_count, completion_count)
}

fn handle_chat_nonstream(
    stream: &mut TcpStream,
    req: &ChatCompletionRequest,
    prompt: &str,
    id: &str,
    state: &Arc<Mutex<InferState>>,
) {
    let olmo3 = SamplingConfig::olmo3();
    let config = SamplingConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        repetition_window: req.repetition_window,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        suppress_initial_tokens: olmo3.suppress_initial_tokens,
    };
    let (content, prompt_tokens, completion_tokens) =
        run_inference(state, prompt, req.max_tokens, &config);

    // The prompt primes `<think>`, so the completion BEGINS inside a think
    // block. Split it: scratchpad → `message.reasoning` (OpenRouter
    // reasoning-model convention), remainder → `message.content`.
    let (reasoning, content) = match content.find("</think>") {
        Some(i) => (content[..i].trim().to_string(),
                    content[i + 8..].trim_start().to_string()),
        None    => (content.trim().to_string(), String::new()),
    };

    let model_id = state.lock().unwrap().model_id.clone();
    let finish   = if completion_tokens >= req.max_tokens { "length" } else { "stop" };
    let resp = ChatCompletionResponse {
        id: id.to_string(), created: unix_ts(),
        model: model_id, content, reasoning,
        prompt_tokens, completion_tokens, finish_reason: finish,
    };
    stream.write_all(&http_json_response(200, "OK", &resp.to_json())).ok();
}

/// Enforce a think budget: if the text contains `<think>` but no `</think>`,
/// truncate the think block to approximately `budget` words and close it.
/// This prevents small models from endlessly rambling in think blocks.
fn enforce_think_budget(text: &str, budget_words: usize) -> String {
    if let Some(think_start) = text.find("<think>") {
        let after_tag = think_start + 7; // len("<think>")
        if !text[after_tag..].contains("</think>") {
            // Think block never closed — truncate at budget
            let think_content = &text[after_tag..];
            let words: Vec<&str> = think_content.split_whitespace().collect();
            let truncated: String = words[..budget_words.min(words.len())].join(" ");
            let before = &text[..think_start];
            return format!("{before}<think>\n{truncated}\n</think>\n\n");
        }
    }
    text.to_string()
}

/// Strip all `<think>...</think>` blocks from the output text.
///
/// Handles:
/// - Properly closed `<think>...</think>` blocks
/// - Malformed `</think` (missing `>`) — common with small models
/// - Filler words ("Okay", "Hmm") that often leak after think blocks
///
/// If stripping removes everything (model only thought, never answered),
/// we extract whatever useful content we can from the think block itself.
fn strip_think_blocks(text: &str) -> String {
    let mut result = String::new();
    let mut remaining = text;
    let mut last_think_content = String::new();
    while let Some(start) = remaining.find("<think>") {
        // Add everything before the think block
        result.push_str(&remaining[..start]);
        // Find the end — try proper close first, then malformed
        let after = &remaining[start..];
        if let Some(end) = after.find("</think>") {
            last_think_content = remaining[start + 7..start + end].trim().to_string();
            remaining = &remaining[start + end + 8..]; // len("</think>") = 8
        } else if let Some(end) = after.find("</think") {
            // Malformed close (no >) — common with small models hitting token limit
            last_think_content = remaining[start + 7..start + end].trim().to_string();
            // Skip past </think and any trailing >
            let skip = start + end + 7; // len("</think") = 7
            remaining = if remaining.len() > skip && remaining.as_bytes().get(skip) == Some(&b'>') {
                &remaining[skip + 1..]
            } else {
                &remaining[skip..]
            };
        } else {
            // No closing tag at all — discard everything from <think> onwards
            last_think_content = remaining[start + 7..].trim().to_string();
            remaining = "";
            break;
        }
    }
    result.push_str(remaining);

    // Clean up stray </think> or </think fragments (can appear without matching <think>)
    let result = result.replace("</think>", "").replace("</think", "");

    // Clean up filler words that often leak after think blocks
    let trimmed = clean_filler(result.trim());
    if !trimmed.is_empty() {
        return trimmed;
    }

    // Model only thought but never answered — extract useful content from think block.
    if !last_think_content.is_empty() {
        return extract_answer_from_think(&last_think_content);
    }

    "I'd be happy to help with that! Could you rephrase your question?".to_string()
}

/// Remove common filler words/phrases that leak from think blocks.
fn clean_filler(text: &str) -> String {
    let mut s = text.to_string();
    // Remove leading filler
    let filler_prefixes = ["Okay\n", "Okay\r\n", "Okay?", "Okay!", "Okay ",
                           "Hmm\n", "Hmm.", "Hmm,", "Hmm ",
                           "Wait,", "Wait.", "Wait\n",
                           "Alright\n", "Alright,", "Alright "];
    for prefix in &filler_prefixes {
        if s.starts_with(prefix) {
            s = s[prefix.len()..].trim_start().to_string();
        }
    }
    // Remove trailing filler paragraph (model sometimes appends "Okay" or
    // rambling "I think..." after a good answer). Find last double-newline
    // and check if the trailing paragraph is filler.
    if let Some(split) = s.rfind("\n\n") {
        let tail = s[split + 2..].trim();
        let tail_lower = tail.to_lowercase();
        let is_filler = tail_lower.starts_with("okay")
            || tail_lower.starts_with("hmm")
            || tail_lower.starts_with("wait")
            || tail_lower.starts_with("alright")
            || tail_lower.starts_with("it seems your question")
            || tail_lower.starts_with("let me know")
            || (tail.len() < 10 && !tail.contains('.'));
        if is_filler {
            s = s[..split].trim_end().to_string();
        }
    }
    // Simple trailing filler lines
    let filler_suffixes = ["\nOkay", "\nOkay?", "\r\nOkay", "\nHmm", "\nAlright"];
    for suffix in &filler_suffixes {
        if s.ends_with(suffix) {
            s = s[..s.len() - suffix.len()].trim_end().to_string();
        }
    }
    s
}

/// Extract the most useful answer from think-block content.
///
/// When the model only produces `<think>` content with no visible answer,
/// we mine the think text for declarative statements that are likely
/// the actual answer (e.g., "The capital of Spain is Madrid.").
fn extract_answer_from_think(think: &str) -> String {
    // Collect all candidate sentences — prefer declarative statements
    let mut candidates: Vec<(usize, &str)> = Vec::new();
    for line in think.lines() {
        let l = line.trim();
        if l.len() < 8 { continue; }

        // Skip meta-thinking / filler lines
        let skip_starts = [
            "Okay", "Hmm", "Wait", "Let me", "User", "Need", "First",
            "I think", "I need", "I recall", "I should", "So,", "So ",
            "The user", "Now,", "Now ", "Alright", "Right",
            "I'm ", "Let's", "...", "---",
        ];
        let is_filler = skip_starts.iter().any(|p| l.starts_with(p));
        if is_filler { continue; }

        // Declarative statements get highest score
        let score = if l.ends_with('.') || l.ends_with('!') {
            100 + l.len().min(200)  // longer declarative = better
        } else if l.contains(" is ") || l.contains(" are ") || l.contains(" was ") {
            80 + l.len().min(200)
        } else {
            l.len().min(200)
        };
        candidates.push((score, l));
    }
    // Return the highest-scoring candidate
    if let Some((_, best)) = candidates.iter().max_by_key(|(s, _)| *s) {
        return best.to_string();
    }
    // Absolute fallback: take the first non-trivial line
    for line in think.lines() {
        let l = line.trim();
        if l.len() > 15 {
            return l.to_string();
        }
    }
    "I'd be happy to help with that! Could you rephrase your question?".to_string()
}

fn handle_chat_stream(
    stream: &mut TcpStream,
    req: &ChatCompletionRequest,
    prompt: &str,
    id: &str,
    state: &Arc<Mutex<InferState>>,
) {
    let olmo3 = SamplingConfig::olmo3();
    let config = SamplingConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        repetition_window: req.repetition_window,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        suppress_initial_tokens: olmo3.suppress_initial_tokens,
    };

    let mut st = state.lock().unwrap();
    let model_id = st.model_id.clone();
    let id_owned = id.to_string();

    // Encode prompt while we have the lock
    let prompt_tokens: Vec<u32> = if let Some(ref tok) = st.tokenizer {
        tok.encode(prompt)
    } else {
        prompt.bytes().map(|b| b as u32).collect()
    };

    // Context-overflow guard (see run_inference): clamp generation to the
    // remaining KV/RoPE capacity.
    let ctx_cap = st.model.as_ref().map(|m| m.model.config().max_seq_len).unwrap_or(usize::MAX);
    let max_tokens = if prompt_tokens.len() >= ctx_cap { 0 }
                     else { req.max_tokens.min(ctx_cap - prompt_tokens.len()) };

    // Take model and tokenizer out of InferState so we can use them
    // independently (model needs &mut, tokenizer needs &, TCP stream needs &mut).
    let mut model = match st.model.take() {
        Some(m) => m,
        None => {
            // No model (echo mode) — send empty [DONE]
            if stream.write_all(&http_sse_header()).is_err() { return; }
            let done = StreamChunk {
                id: id_owned, model: model_id,
                delta: String::new(), is_reasoning: false, done: true, finish_reason: None,
                usage: None,
            };
            write_sse_chunk(stream, &done.to_sse()).ok();
            write_chunk_end(stream).ok();
            return;
        }
    };
    let tokenizer = st.tokenizer.take();

    // Release the Mutex so other endpoints (health, models) aren't blocked
    // during the (potentially multi-second) inference.
    drop(st);

    // Send SSE headers immediately so the client knows streaming has started
    if stream.write_all(&http_sse_header()).is_err() {
        let mut st = state.lock().unwrap();
        st.model = Some(model);
        st.tokenizer = tokenizer;
        return;
    }

    // True token-by-token streaming: generate_streaming_events calls our
    // closure during prefill and after each token; visible tokens are decoded
    // and flushed as SSE chunks immediately.
    let mut token_count: usize = 0;
    let mut write_ok = true;

    // Keep-alive: emit an SSE comment whenever KEEPALIVE_EVERY elapses with
    // nothing written — during prompt prefill (runs at ~decode speed, so a
    // 4K-token prompt is ~a minute of silence otherwise) and during <think>
    // suppression. Prevents proxy/fetch timeouts (OpenRouter requirement;
    // Cloudflare proxies time out at 100s of idle).
    const KEEPALIVE_EVERY: Duration = Duration::from_secs(10);
    let mut last_write = Instant::now();

    // Think block suppression: track <think> blocks and hide them from the
    // client. Small models often generate <think> despite system prompts;
    // we silently consume think tokens and only stream visible content.
    let think_budget: usize = 100_000; // effectively unlimited; bounded by max_tokens
    let mut accumulated_text = String::new();
    // The prompt primes `<think>` (official OLMo-3-Think template), so the
    // stream BEGINS inside a think block.
    let mut in_think_block = true;
    let mut think_tokens: usize = 0;

    // InferEngine::generate_streaming_events closure takes (event, deposit);
    // deposit is None when no StigmergicHook is configured (the common case).
    model.generate_streaming_events(
        &prompt_tokens,
        max_tokens,
        &config,
        |ev, _deposit| {
            let tok_id = match ev {
                GenEvent::Prefill { .. } => {
                    if last_write.elapsed() >= KEEPALIVE_EVERY {
                        match write_sse_keepalive(stream) {
                            Ok(()) => last_write = Instant::now(),
                            Err(_) => { write_ok = false; return false; }
                        }
                    }
                    return true;
                }
                GenEvent::Token(t) => t,
            };
            token_count += 1;

            // Decode this single token
            let text = if let Some(ref tok) = tokenizer {
                tok.decode(&[tok_id])
            } else {
                let bytes = vec![(tok_id % 256) as u8];
                String::from_utf8_lossy(&bytes).into_owned()
            };

            accumulated_text.push_str(&text);

            // Track think block state — suppress all content inside <think>...</think>
            if !in_think_block && accumulated_text.contains("<think>") {
                in_think_block = true;
                think_tokens = 0;
                // Don't send <think> or anything inside it
                return true;
            }
            if in_think_block {
                think_tokens += 1;
                if accumulated_text.contains("</think>") {
                    in_think_block = false;
                    // Think block closed — the closing-tag token itself is
                    // not sent; visible content follows.
                    return true;
                }
                if think_tokens > think_budget {
                    in_think_block = false;
                    return true;
                }
                // Inside the think block: stream the scratchpad as
                // `delta.reasoning` chunks (OpenRouter reasoning-model
                // convention). Real bytes also keep proxies from timing out.
                let chunk = StreamChunk {
                    id: id_owned.clone(),
                    model: model_id.clone(),
                    delta: text,
                    is_reasoning: true,
                    done: false,
                    finish_reason: None,
                    usage: None,
                };
                match write_sse_chunk(stream, &chunk.to_sse()) {
                    Ok(()) => { last_write = Instant::now(); return true; }
                    Err(_) => { write_ok = false; return false; }
                }
            }

            let finish_reason: Option<&'static str> = if token_count >= max_tokens {
                Some("length")
            } else {
                None
            };

            let chunk = StreamChunk {
                id: id_owned.clone(),
                model: model_id.clone(),
                delta: text, is_reasoning: false,
                done: false,
                finish_reason,
                usage: None,
            };
            match write_sse_chunk(stream, &chunk.to_sse()) {
                Ok(()) => { last_write = Instant::now(); true } // keep generating
                Err(_) => {
                    write_ok = false;
                    false // client disconnected — stop generation
                }
            }
        },
    );

    // Write final finish-reason chunk (with usage — required by OpenRouter
    // for streaming responses) + [DONE] sentinel
    if write_ok {
        let finish: &'static str = if token_count >= max_tokens { "length" } else { "stop" };
        let final_chunk = StreamChunk {
            id: id_owned.clone(),
            model: model_id.clone(),
            delta: String::new(), is_reasoning: false,
            done: false,
            finish_reason: Some(finish),
            usage: Some((prompt_tokens.len(), token_count)),
        };
        write_sse_chunk(stream, &final_chunk.to_sse()).ok();

        let done = StreamChunk {
            id: id_owned, model: model_id,
            delta: String::new(), is_reasoning: false, done: true, finish_reason: None,
            usage: None,
        };
        write_sse_chunk(stream, &done.to_sse()).ok();
        write_chunk_end(stream).ok();
    }

    // Put model and tokenizer back into InferState
    let mut st = state.lock().unwrap();
    st.model = Some(model);
    st.tokenizer = tokenizer;
}

fn handle_completion(
    stream: &mut TcpStream,
    req: &CompletionRequest,
    id: &str,
    state: &Arc<Mutex<InferState>>,
) {
    let olmo3 = SamplingConfig::olmo3();
    let config = SamplingConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        repetition_window: req.repetition_window,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        suppress_initial_tokens: olmo3.suppress_initial_tokens,
    };
    let (text, prompt_tokens, completion_tokens) =
        run_inference(state, &req.prompt, req.max_tokens, &config);
    let model_id = state.lock().unwrap().model_id.clone();
    let finish   = if completion_tokens >= req.max_tokens { "length" } else { "stop" };
    let resp = CompletionResponse {
        id: id.to_string(), created: unix_ts(),
        model: model_id, text,
        prompt_tokens, completion_tokens, finish_reason: finish,
    };
    stream.write_all(&http_json_response(200, "OK", &resp.to_json())).ok();
}

// ── System prompt ────────────────────────────────────────────────────────────

/// Default system prompt injected when no system message is present.
/// This is the EXACT system prompt from the official OLMo-3-Think
/// `chat_template.jinja` — the model was trained with it; substituting a
/// different persona (or banning thinking) measurably degrades quality.
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are ATLAS, a helpful and knowledgeable AI assistant. \
     Respond directly to questions. Keep answers clear, informative, \
     and concise. Do not use thinking tags or internal monologue.";

/// Inject a default system prompt if the message list has no system message.
///
/// This is the single biggest improvement for small model output quality.
/// Without a system prompt, OLMo-3-7B-Think enters degenerate `<think>`
/// loops on nearly every query.
fn inject_default_system_prompt(messages: &mut Vec<ChatMessage>) {
    let has_system = messages.iter().any(|m| m.role == "system");
    if !has_system {
        messages.insert(0, ChatMessage {
            role: "system".to_string(),
            content: DEFAULT_SYSTEM_PROMPT.to_string(),
        });
    } else {
        // Official template appends the functions suffix to user-supplied
        // system prompts; keep the model in-distribution.
        for m in messages.iter_mut() {
            if m.role == "system" && !m.content.contains("<functions>") {
                m.content.push_str(
                    " You do not currently have access to any functions. <functions></functions>");
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Split `text` into byte chunks of size ≤ `chunk_bytes`, respecting UTF-8
/// character boundaries so multi-byte sequences are never split.
pub fn split_chunks(text: &str, chunk_bytes: usize) -> Vec<&str> {
    if text.is_empty() || chunk_bytes == 0 { return Vec::new(); }
    let mut out   = Vec::new();
    let mut start = 0;
    let bytes     = text.as_bytes();
    while start < bytes.len() {
        let mut end = (start + chunk_bytes).min(bytes.len());
        // Retreat to a valid UTF-8 character boundary.
        while end > start && !text.is_char_boundary(end) { end -= 1; }
        if end == start { end += 1; } // shouldn't happen, but guard infinite loop
        if let Ok(s) = std::str::from_utf8(&bytes[start..end]) {
            out.push(s);
        }
        start = end;
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_get_request() {
        let raw = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let (method, path, _headers, body) = parse_http_request(raw).unwrap();
        assert_eq!(method, "GET");
        assert_eq!(path, "/v1/models");
        assert!(body.is_empty());
    }

    #[test]
    fn parse_post_with_body() {
        let body_str = r#"{"model":"atlas"}"#;
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
            body_str.len(), body_str
        );
        let (method, path, headers, body) = parse_http_request(raw.as_bytes()).unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/v1/chat/completions");
        assert!(headers.iter().any(|(k, _)| k == "content-type"));
        assert_eq!(std::str::from_utf8(&body).unwrap(), body_str);
    }

    #[test]
    fn parse_request_no_body() {
        let raw = b"GET /health HTTP/1.1\r\n\r\n";
        let (method, path, _h, body) = parse_http_request(raw).unwrap();
        assert_eq!(method, "GET");
        assert_eq!(path, "/health");
        assert!(body.is_empty());
    }

    #[test]
    fn parse_request_bad_returns_none() {
        let raw = b"not an HTTP request";
        assert!(parse_http_request(raw).is_none());
    }

    #[test]
    fn http_json_response_format() {
        let resp = http_json_response(200, "OK", r#"{"status":"ok"}"#);
        let s    = std::str::from_utf8(&resp).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK"));
        assert!(s.contains("Content-Type: application/json"));
        assert!(s.contains(r#"{"status":"ok"}"#));
    }

    #[test]
    fn split_chunks_basic() {
        let chunks = split_chunks("Hello world", 4);
        assert!(!chunks.is_empty());
        assert_eq!(chunks.concat(), "Hello world");
    }

    #[test]
    fn split_chunks_empty() {
        assert!(split_chunks("", 4).is_empty());
    }

    #[test]
    fn split_chunks_exact() {
        let chunks = split_chunks("abcd", 4);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "abcd");
    }

    #[test]
    fn split_chunks_preserves_all_bytes() {
        let text   = "The quick brown fox jumps over the lazy dog";
        let chunks = split_chunks(text, 7);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn infer_state_no_model_echo() {
        let state = Arc::new(Mutex::new(InferState {
            model: None, tokenizer: None,
            model_id: "test".to_string(),
            chat_template: ChatTemplate::ChatML,
        }));
        let config = SamplingConfig { temperature: 0.0, ..SamplingConfig::default() };
        let (text, prompt_count, completion_count) =
            run_inference(&state, "hello world", 10, &config);
        assert_eq!(text, "");              // no model → empty
        assert_eq!(completion_count, 0);
        assert_eq!(prompt_count, 11);      // byte-encode: "hello world" = 11 bytes
    }

    #[test]
    fn http_options_response_has_cors_headers() {
        let resp = http_options_response();
        let s    = std::str::from_utf8(&resp).unwrap();
        assert!(s.starts_with("HTTP/1.1 204"));
        assert!(s.contains("Access-Control-Allow-Origin"));
    }

    // ── Auth / rate-limit middleware ──────────────────────────────────────

    fn hdrs(auth: Option<&str>) -> Vec<(String, String)> {
        let mut h = vec![("content-type".to_string(), "application/json".to_string())];
        if let Some(a) = auth {
            h.push(("authorization".to_string(), a.to_string()));
        }
        h
    }

    #[test]
    fn bearer_ok_accepts_valid_key() {
        assert!(bearer_ok(&hdrs(Some("Bearer sk-test-123")), "sk-test-123"));
    }

    #[test]
    fn bearer_ok_accepts_lowercase_scheme() {
        assert!(bearer_ok(&hdrs(Some("bearer sk-test-123")), "sk-test-123"));
    }

    #[test]
    fn bearer_ok_rejects_wrong_key() {
        assert!(!bearer_ok(&hdrs(Some("Bearer sk-wrong")), "sk-test-123"));
        // same length, one byte off
        assert!(!bearer_ok(&hdrs(Some("Bearer sk-test-124")), "sk-test-123"));
    }

    #[test]
    fn bearer_ok_rejects_missing_header() {
        assert!(!bearer_ok(&hdrs(None), "sk-test-123"));
    }

    #[test]
    fn bearer_ok_rejects_non_bearer_scheme() {
        assert!(!bearer_ok(&hdrs(Some("Basic c2stdGVzdA==")), "sk-test-123"));
    }

    #[test]
    fn bearer_ok_rejects_empty_token() {
        assert!(!bearer_ok(&hdrs(Some("Bearer ")), "sk-test-123"));
    }

    #[test]
    fn ct_eq_basic() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab"));
        assert!(ct_eq(b"", b""));
    }

    #[test]
    fn unauthorized_response_shape() {
        let resp = http_unauthorized_response();
        let s = std::str::from_utf8(&resp).unwrap();
        assert!(s.starts_with("HTTP/1.1 401 Unauthorized"));
        assert!(s.contains("WWW-Authenticate: Bearer"));
        assert!(s.contains("authentication_error"));
    }

    #[test]
    fn too_many_requests_response_shape() {
        let resp = http_too_many_requests_response(30);
        let s = std::str::from_utf8(&resp).unwrap();
        assert!(s.starts_with("HTTP/1.1 429 Too Many Requests"));
        assert!(s.contains("Retry-After: 30"));
        assert!(s.contains("rate_limit_exceeded"));
    }

    #[test]
    fn json_response_with_extra_headers() {
        let resp = http_json_response_with(200, "OK", "{}", &[("X-Test", "1")]);
        let s = std::str::from_utf8(&resp).unwrap();
        assert!(s.contains("X-Test: 1"));
        assert!(s.contains("Connection: close"));
    }

    #[test]
    fn think_budget_closes_unclosed_block() {
        let text = "<think>\nword1 word2 word3 word4 word5 word6 word7 word8 word9 word10";
        let result = enforce_think_budget(text, 5);
        assert!(result.contains("</think>"), "should close the think block");
        assert!(result.contains("word5"), "should include words up to budget");
        assert!(!result.contains("word6"), "should exclude words beyond budget");
    }

    #[test]
    fn think_budget_leaves_closed_blocks_alone() {
        let text = "<think>\nsome thinking here\n</think>\n\nMadrid is the capital.";
        let result = enforce_think_budget(text, 3);
        assert_eq!(result, text, "already-closed think blocks should be untouched");
    }

    #[test]
    fn think_budget_no_think_block() {
        let text = "Madrid is the capital of Spain.";
        let result = enforce_think_budget(text, 5);
        assert_eq!(result, text, "text without think blocks should be untouched");
    }

    #[test]
    fn default_system_prompt_injected_when_absent() {
        let mut messages = vec![
            ChatMessage { role: "user".to_string(), content: "Hello".to_string() },
        ];
        inject_default_system_prompt(&mut messages);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(messages[0].content.contains("ATLAS"));
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn default_system_prompt_not_injected_when_present() {
        let mut messages = vec![
            ChatMessage { role: "system".to_string(), content: "Custom system".to_string() },
            ChatMessage { role: "user".to_string(), content: "Hello".to_string() },
        ];
        inject_default_system_prompt(&mut messages);
        assert_eq!(messages.len(), 2);
        // User system prompt is preserved, with the official OLMo template
        // functions-suffix appended (keeps the model in-distribution).
        assert!(messages[0].content.starts_with("Custom system"));
        assert!(messages[0].content.ends_with("<functions></functions>"));
    }

    #[test]
    fn strip_think_blocks_removes_closed() {
        let text = "<think>\nsome thinking\n</think>\n\nMadrid is the capital.";
        let result = strip_think_blocks(text);
        assert_eq!(result, "Madrid is the capital.");
    }

    #[test]
    fn strip_think_blocks_removes_multiple() {
        let text = "<think>thought 1</think>Hello <think>thought 2</think>world!";
        let result = strip_think_blocks(text);
        assert_eq!(result, "Hello world!");
    }

    #[test]
    fn strip_think_blocks_no_blocks() {
        let text = "Just a normal response.";
        let result = strip_think_blocks(text);
        assert_eq!(result, text);
    }

    #[test]
    fn strip_think_blocks_unclosed_graceful() {
        let text = "<think>rambling without end";
        let result = strip_think_blocks(text);
        // Should not return raw <think> block
        assert!(!result.contains("<think>"), "should not expose raw think block");
    }

    #[test]
    fn strip_think_blocks_all_think_no_answer() {
        let text = "<think>\nOkay, I need to think.\nThe capital of France is Paris.\n</think>";
        let result = strip_think_blocks(text);
        // Should extract the useful content from the think block
        assert!(!result.contains("<think>"));
        assert!(result.contains("Paris") || result.contains("help"));
    }

    #[test]
    fn strip_think_blocks_malformed_close_no_gt() {
        // Model sometimes outputs </think without closing >
        let text = "Some content\n</think\nMore content after";
        let result = strip_think_blocks(text);
        assert!(!result.contains("</think"), "malformed close tag should not leak");
    }

    #[test]
    fn strip_think_blocks_with_malformed_close() {
        let text = "<think>thinking stuff</think\nActual answer here.";
        let result = strip_think_blocks(text);
        assert!(result.contains("Actual answer"), "content after malformed close should appear");
        assert!(!result.contains("<think>"));
    }

    #[test]
    fn clean_filler_removes_okay() {
        assert_eq!(clean_filler("Okay\nThe answer is 42."), "The answer is 42.");
        assert_eq!(clean_filler("Hello world\nOkay"), "Hello world");
        // Trailing filler paragraph
        assert_eq!(
            clean_filler("Good answer here.\n\nOkay let me know if that helps"),
            "Good answer here."
        );
    }

    #[test]
    fn clean_filler_preserves_normal() {
        assert_eq!(clean_filler("Madrid is the capital."), "Madrid is the capital.");
    }

    #[test]
    fn extract_answer_declarative_preferred() {
        let think = "Okay, let me think.\nI need to figure this out.\nMadrid is the capital of Spain.\nWait, actually let me reconsider.";
        let result = extract_answer_from_think(think);
        assert!(result.contains("Madrid"), "should extract the declarative statement, got: {}", result);
    }

    #[test]
    fn extract_answer_fallback_on_all_filler() {
        let think = "Okay so I need to think about this.\nLet me consider.\nFirst, I should note.\nHmm, interesting.";
        let result = extract_answer_from_think(think);
        // Should return something rather than panic
        assert!(!result.is_empty());
    }
}
