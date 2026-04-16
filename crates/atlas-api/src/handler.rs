//! HTTP/1.1 request parser and route handler.
//!
//! Parses raw HTTP requests from TcpStream bytes. Routes to appropriate
//! handler functions. Returns raw HTTP response bytes.

use std::io::Write;
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use atlas_model::OlmoModel;
use atlas_tokenize::Tokenizer;

use crate::types::{
    json_string, gen_id, unix_ts,
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    ErrorResponse, StreamChunk,
};

/// Shared inference state (model + tokenizer).
pub struct InferState {
    /// The loaded model (or None if no weights loaded — echo mode).
    pub model: Option<OlmoModel>,
    /// The tokenizer (or None if not available).
    pub tokenizer: Option<Tokenizer>,
    /// Model ID string.
    pub model_id: String,
}

/// Parse a raw HTTP/1.1 request from a byte slice.
/// Returns (method, path, headers, body).
pub fn parse_http_request(raw: &[u8]) -> Option<(String, String, Vec<(String, String)>, Vec<u8>)> {
    // Find header/body separator
    let sep = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let header_bytes = &raw[..sep];
    let body = raw[sep + 4..].to_vec();

    let header_str = std::str::from_utf8(header_bytes).ok()?;
    let mut lines = header_str.lines();

    // Parse request line: "GET /v1/models HTTP/1.1"
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path   = parts.next()?.to_string();

    // Parse headers
    let mut headers = Vec::new();
    let mut content_length: usize = 0;
    for line in lines {
        if let Some(i) = line.find(':') {
            let k = line[..i].trim().to_lowercase();
            let v = line[i+1..].trim().to_string();
            if k == "content-length" {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }

    // Trim body to Content-Length
    let body = body[..content_length.min(body.len())].to_vec();

    Some((method, path, headers, body))
}

/// Build an HTTP/1.1 response with JSON body.
pub fn http_json_response(status: u16, reason: &str, body: &str) -> Vec<u8> {
    let b = body.as_bytes();
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n{}",
        b.len(), body
    ).into_bytes()
}

/// Build a streaming SSE HTTP response header (chunked).
pub fn http_sse_header() -> Vec<u8> {
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n".as_bytes().to_vec()
}

/// Write a single SSE chunk using HTTP chunked transfer encoding.
pub fn write_sse_chunk(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    // Chunked encoding: "<hex-size>\r\n<data>\r\n"
    let bytes = data.as_bytes();
    stream.write_all(format!("{:x}\r\n", bytes.len()).as_bytes())?;
    stream.write_all(bytes)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

/// Write HTTP chunked transfer terminator.
pub fn write_chunk_end(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

/// Build a CORS preflight response.
pub fn http_options_response() -> Vec<u8> {
    "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".as_bytes().to_vec()
}

/// Route and handle a single HTTP request. Writes response to stream.
pub fn handle(
    stream: &mut TcpStream,
    method: &str,
    path: &str,
    body: &[u8],
    state: &Arc<Mutex<InferState>>,
) {
    // Strip query params from path
    let clean_path = path.split('?').next().unwrap_or(path);

    // CORS preflight
    if method == "OPTIONS" {
        stream.write_all(&http_options_response()).ok();
        return;
    }

    match (method, clean_path) {
        // ── Health check ──────────────────────────────────────────────────
        ("GET", "/health") | ("GET", "/") => {
            let body = r#"{"status":"ok","service":"atlas-api"}"#;
            stream.write_all(&http_json_response(200, "OK", body)).ok();
        }

        // ── List models ───────────────────────────────────────────────────
        ("GET", "/v1/models") => {
            let st = state.lock().unwrap();
            let ts = unix_ts();
            let body = format!(
                r#"{{"object":"list","data":[{{"id":{},"object":"model","created":{},"owned_by":"atlas-agi","permission":[],"root":{},"parent":null}}]}}"#,
                json_string(&st.model_id),
                ts,
                json_string(&st.model_id),
            );
            drop(st);
            stream.write_all(&http_json_response(200, "OK", &body)).ok();
        }

        // ── Chat completions ──────────────────────────────────────────────
        ("POST", "/v1/chat/completions") => {
            let body_str = match std::str::from_utf8(body) {
                Ok(s) => s,
                Err(_) => {
                    let err = ErrorResponse { message: "invalid UTF-8 body".to_string(), error_type: "invalid_request_error", status: 400 };
                    stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
                    return;
                }
            };
            let req = match ChatCompletionRequest::parse(body_str) {
                Ok(r) => r,
                Err(e) => {
                    let err = ErrorResponse { message: format!("parse error: {e}"), error_type: "invalid_request_error", status: 400 };
                    stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
                    return;
                }
            };

            let prompt = req.to_prompt();
            let id = gen_id("chatcmpl");

            if req.stream {
                handle_chat_stream(stream, &req, &prompt, &id, state);
            } else {
                handle_chat_nonstream(stream, &req, &prompt, &id, state);
            }
        }

        // ── Text completions ──────────────────────────────────────────────
        ("POST", "/v1/completions") => {
            let body_str = match std::str::from_utf8(body) {
                Ok(s) => s,
                Err(_) => {
                    let err = ErrorResponse { message: "invalid UTF-8 body".to_string(), error_type: "invalid_request_error", status: 400 };
                    stream.write_all(&http_json_response(400, "Bad Request", &err.to_json())).ok();
                    return;
                }
            };
            let req = match CompletionRequest::parse(body_str) {
                Ok(r) => r,
                Err(e) => {
                    let err = ErrorResponse { message: format!("parse error: {e}"), error_type: "invalid_request_error", status: 400 };
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

// ── Inference helpers ─────────────────────────────────────────────────────────

/// Run inference: encode prompt, generate tokens, decode.
/// Returns (generated_text, prompt_token_count, completion_token_count).
fn run_inference(
    state: &Arc<Mutex<InferState>>,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> (String, usize, usize) {
    let mut st = state.lock().unwrap();

    let prompt_tokens: Vec<u32> = if let Some(ref tok) = st.tokenizer {
        tok.encode(prompt)
    } else {
        prompt.bytes().map(|b| b as u32).collect()
    };
    let prompt_count = prompt_tokens.len();

    let new_tokens: Vec<u32> = if let Some(ref mut model) = st.model {
        model.reset();
        model.generate(&prompt_tokens, max_tokens, temperature)
    } else {
        // No model loaded — echo mode for testing
        vec![]
    };
    let completion_count = new_tokens.len();

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
    let finish = if completion_tokens >= req.max_tokens { "length" } else { "stop" };
    let resp = ChatCompletionResponse {
        id: id.to_string(),
        created: unix_ts(),
        model: model_id,
        content,
        prompt_tokens,
        completion_tokens,
        finish_reason: finish,
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
    // For streaming we generate all tokens first, then stream them back token-by-token.
    // This is simpler and correct — true per-token streaming would require unlocking
    // between tokens which needs architecture changes.
    let (content, _prompt_tokens, completion_tokens) =
        run_inference(state, prompt, req.max_tokens, req.temperature);

    let model_id = state.lock().unwrap().model_id.clone();
    let finish = if completion_tokens >= req.max_tokens { "length" } else { "stop" };

    // Write SSE header
    if stream.write_all(&http_sse_header()).is_err() { return; }

    // Stream content word-by-word for better UX (split on spaces preserving them)
    let chunks: Vec<&str> = split_into_chunks(&content);
    let n = chunks.len();
    for (i, chunk_text) in chunks.iter().enumerate() {
        let finish_reason = if i == n - 1 { Some(finish) } else { None };
        let chunk = StreamChunk {
            id: id.to_string(),
            model: model_id.clone(),
            delta: chunk_text.to_string(),
            done: false,
            finish_reason,
        };
        if write_sse_chunk(stream, &chunk.to_sse()).is_err() { return; }
    }

    // Send [DONE]
    let done_chunk = StreamChunk {
        id: id.to_string(), model: model_id,
        delta: String::new(), done: true, finish_reason: None,
    };
    write_sse_chunk(stream, &done_chunk.to_sse()).ok();
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
    let finish = if completion_tokens >= req.max_tokens { "length" } else { "stop" };
    let resp = CompletionResponse {
        id: id.to_string(),
        created: unix_ts(),
        model: model_id,
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason: finish,
    };
    stream.write_all(&http_json_response(200, "OK", &resp.to_json())).ok();
}

/// Split text into streaming chunks (by word, preserving whitespace).
fn split_into_chunks(text: &str) -> Vec<&str> {
    // Split into individual characters for smooth streaming feel,
    // but batch into ~4 char chunks for efficiency
    if text.is_empty() { return vec![]; }
    let mut chunks = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let chunk_size = 4;
    while start < bytes.len() {
        let end = (start + chunk_size).min(bytes.len());
        // Don't split UTF-8 sequences
        let mut e = end;
        while e > start && bytes[e-1] >= 0x80 && bytes[e-1] < 0xC0 { e -= 1; }
        if e == start { e = end; } // Fallback
        if let Ok(s) = std::str::from_utf8(&bytes[start..e]) {
            chunks.push(s);
            start = e;
        } else {
            start += 1;
        }
    }
    chunks
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
    fn parse_post_request_with_body() {
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
    fn parse_request_strips_query() {
        let raw = b"GET /v1/models?limit=10 HTTP/1.1\r\nHost: localhost\r\n\r\n";
        let (_, path, _, _) = parse_http_request(raw).unwrap();
        // path includes query, stripping happens in handle()
        assert_eq!(path, "/v1/models?limit=10");
    }

    #[test]
    fn http_json_response_format() {
        let resp = http_json_response(200, "OK", r#"{"status":"ok"}"#);
        let s = std::str::from_utf8(&resp).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK"));
        assert!(s.contains("Content-Type: application/json"));
        assert!(s.contains(r#"{"status":"ok"}"#));
    }

    #[test]
    fn split_chunks_basic() {
        let chunks = split_into_chunks("Hello world");
        assert!(!chunks.is_empty());
        assert_eq!(chunks.concat(), "Hello world");
    }

    #[test]
    fn split_chunks_empty() {
        let chunks = split_into_chunks("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn infer_state_no_model() {
        let state = Arc::new(Mutex::new(InferState {
            model: None,
            tokenizer: None,
            model_id: "test".to_string(),
        }));
        let (text, prompt_count, completion_count) = run_inference(&state, "hello", 10, 0.0);
        // No model → empty output
        assert_eq!(text, "");
        assert_eq!(completion_count, 0);
        assert!(prompt_count > 0); // byte-encoded prompt
    }
}
