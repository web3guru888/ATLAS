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

use atlas_model::OlmoModel;
use atlas_tokenize::Tokenizer;

use crate::types::{
    json_string, gen_id, unix_ts,
    ChatCompletionRequest, ChatCompletionResponse, ChatTemplate,
    CompletionRequest, CompletionResponse,
    ErrorResponse, StreamChunk,
};

// ── Shared inference state ────────────────────────────────────────────────────

/// Model + tokenizer shared across HTTP worker threads.
pub struct InferState {
    /// Loaded model, or None in echo/test mode.
    pub model: Option<OlmoModel>,
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
    let len = body.len();
    format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
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

/// Write the chunked-transfer terminating chunk.
pub fn write_chunk_end(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
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

        // ── Chat completions ──────────────────────────────────────────────
        ("POST", "/v1/chat/completions") => {
            let body_str = match std::str::from_utf8(body) {
                Ok(s) => s,
                Err(_) => {
                    let err = ErrorResponse { message: "request body is not valid UTF-8".to_string(), error_type: "invalid_request_error", status: 400 };
                    stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
                    return;
                }
            };
            let req = match ChatCompletionRequest::parse(body_str) {
                Ok(r)  => r,
                Err(e) => {
                    let err = ErrorResponse { message: format!("{e}"), error_type: "invalid_request_error", status: 400 };
                    stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
                    return;
                }
            };
            let template = state.lock().unwrap().chat_template;
            let prompt = req.to_prompt_with(&template);
            let id     = gen_id("chatcmpl");
            if req.stream {
                handle_chat_stream(stream, &req, &prompt, &id, state);
            } else {
                handle_chat_nonstream(stream, &req, &prompt, &id, state);
            }
        }

        // ── Text completions ──────────────────────────────────────────────
        ("POST", "/v1/completions") => {
            let body_str = match std::str::from_utf8(body) {
                Ok(s)  => s,
                Err(_) => {
                    let err = ErrorResponse { message: "request body is not valid UTF-8".to_string(), error_type: "invalid_request_error", status: 400 };
                    stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
                    return;
                }
            };
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

/// Run model inference for `prompt`.
/// Returns `(generated_text, prompt_token_count, completion_token_count)`.
fn run_inference(
    state: &Arc<Mutex<InferState>>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> (String, usize, usize) {
    let mut st = state.lock().unwrap();

    // Encode prompt
    let prompt_tokens: Vec<u32> = if let Some(ref tok) = st.tokenizer {
        tok.encode(prompt)
    } else {
        prompt.bytes().map(|b| b as u32).collect()
    };
    let prompt_count = prompt_tokens.len();

    // Generate
    let new_tokens: Vec<u32> = if let Some(ref mut model) = st.model {
        model.reset();
        model.generate(&prompt_tokens, max_tokens, temperature)
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
    let (content, prompt_tokens, completion_tokens) =
        run_inference(state, prompt, req.max_tokens, req.temperature);
    let model_id = state.lock().unwrap().model_id.clone();
    let finish   = if completion_tokens >= req.max_tokens { "length" } else { "stop" };
    let resp = ChatCompletionResponse {
        id: id.to_string(), created: unix_ts(),
        model: model_id, content,
        prompt_tokens, completion_tokens, finish_reason: finish,
    };
    stream.write_all(&http_json_response(200, "OK", &resp.to_json())).ok();
}

fn handle_chat_stream(
    stream: &mut TcpStream,
    req: &ChatCompletionRequest,
    prompt: &str,
    id: &str,
    state: &Arc<Mutex<InferState>>,
) {
    let (content, _prompt_tokens, completion_tokens) =
        run_inference(state, prompt, req.max_tokens, req.temperature);
    let model_id = state.lock().unwrap().model_id.clone();
    let finish   = if completion_tokens >= req.max_tokens { "length" } else { "stop" };

    if stream.write_all(&http_sse_header()).is_err() { return; }

    // Stream in small text chunks for a smooth client UX
    let parts = split_chunks(&content, 4);
    let n     = parts.len();
    for (i, text) in parts.iter().enumerate() {
        let finish_reason = if i == n.saturating_sub(1) && n > 0 { Some(finish) } else { None };
        let chunk = StreamChunk {
            id: id.to_string(), model: model_id.clone(),
            delta: text.to_string(), done: false, finish_reason,
        };
        if write_sse_chunk(stream, &chunk.to_sse()).is_err() { return; }
    }
    // [DONE] sentinel
    let done = StreamChunk {
        id: id.to_string(), model: model_id,
        delta: String::new(), done: true, finish_reason: None,
    };
    write_sse_chunk(stream, &done.to_sse()).ok();
    write_chunk_end(stream).ok();
}

fn handle_completion(
    stream: &mut TcpStream,
    req: &CompletionRequest,
    id: &str,
    state: &Arc<Mutex<InferState>>,
) {
    let (text, prompt_tokens, completion_tokens) =
        run_inference(state, &req.prompt, req.max_tokens, req.temperature);
    let model_id = state.lock().unwrap().model_id.clone();
    let finish   = if completion_tokens >= req.max_tokens { "length" } else { "stop" };
    let resp = CompletionResponse {
        id: id.to_string(), created: unix_ts(),
        model: model_id, text,
        prompt_tokens, completion_tokens, finish_reason: finish,
    };
    stream.write_all(&http_json_response(200, "OK", &resp.to_json())).ok();
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
        let (text, prompt_count, completion_count) =
            run_inference(&state, "hello world", 10, 0.0);
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
}
