//! OpenAI-compatible request/response types.
//!
//! All serialization is hand-rolled — no serde or external deps.

use atlas_json::Json;
use atlas_core::{AtlasError, Result};

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind host (e.g. "0.0.0.0").
    pub host: String,
    /// TCP port (default 8080).
    pub port: u16,
    /// Model identifier returned by /v1/models.
    pub model_id: String,
    /// Directory containing model weights and tokenizer.
    pub weights_dir: Option<String>,
    /// Max tokens per generation (hard cap).
    pub max_tokens: usize,
    /// Number of worker threads (currently 1 inference thread + N HTTP threads).
    pub workers: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            model_id: "atlas".to_string(),
            weights_dir: None,
            max_tokens: 2048,
            workers: 4,
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// "system", "user", or "assistant".
    pub role: String,
    /// Message content.
    pub content: String,
}

impl ChatMessage {
    /// Parse from a JSON object.
    pub fn from_json(v: &Json) -> Option<Self> {
        let role    = v.get("role")?.as_str()?.to_string();
        let content = v.get("content")?.as_str()?.to_string();
        Some(Self { role, content })
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"role":{},"content":{}}}"#,
            json_string(&self.role),
            json_string(&self.content),
        )
    }
}

/// POST /v1/chat/completions request body.
#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Max tokens to generate (default 256).
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Whether to stream tokens back via SSE.
    pub stream: bool,
}

impl ChatCompletionRequest {
    /// Parse from raw JSON string.
    pub fn parse(body: &str) -> Result<Self> {
        let v = Json::parse(body)
            .map_err(|e| AtlasError::Parse(format!("invalid JSON: {e}")))?;
        let model = v.get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("atlas")
            .to_string();
        let messages = v.get("messages")
            .and_then(|a| a.as_array())
            .map(|arr| arr.iter().filter_map(ChatMessage::from_json).collect())
            .unwrap_or_default();
        let max_tokens = v.get("max_tokens")
            .and_then(|x| x.as_usize())
            .unwrap_or(256);
        let temperature = v.get("temperature")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0) as f32;
        let stream = v.get("stream")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        Ok(Self { model, messages, max_tokens, temperature, stream })
    }

    /// Build a single prompt string from the messages (ChatML format).
    pub fn to_prompt(&self) -> String {
        let mut prompt = String::new();
        for msg in &self.messages {
            match msg.role.as_str() {
                "system"    => { prompt.push_str(&format!("<|system|>\n{}\n", msg.content)); }
                "user"      => { prompt.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", msg.content)); }
                "assistant" => { prompt.push_str(&format!("{}\n", msg.content)); }
                _           => { prompt.push_str(&format!("{}\n", msg.content)); }
            }
        }
        prompt
    }
}

/// POST /v1/completions request body.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Model identifier.
    pub model: String,
    /// Prompt text.
    pub prompt: String,
    /// Max tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Whether to stream.
    pub stream: bool,
}

impl CompletionRequest {
    /// Parse from raw JSON string.
    pub fn parse(body: &str) -> Result<Self> {
        let v = Json::parse(body)
            .map_err(|e| AtlasError::Parse(format!("invalid JSON: {e}")))?;
        let model = v.get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("atlas")
            .to_string();
        let prompt = v.get("prompt")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        let max_tokens = v.get("max_tokens")
            .and_then(|x| x.as_usize())
            .unwrap_or(256);
        let temperature = v.get("temperature")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.0) as f32;
        let stream = v.get("stream")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        Ok(Self { model, prompt, max_tokens, temperature, stream })
    }
}

// ── Response types ────────────────────────────────────────────────────────────

/// A chat completion response (non-streaming).
pub struct ChatCompletionResponse {
    /// Unique completion ID.
    pub id: String,
    /// Unix timestamp.
    pub created: u64,
    /// Model ID.
    pub model: String,
    /// Generated content.
    pub content: String,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of generated tokens.
    pub completion_tokens: usize,
    /// Finish reason.
    pub finish_reason: &'static str,
}

impl ChatCompletionResponse {
    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        let total = self.prompt_tokens + self.completion_tokens;
        format!(
            r#"{{"id":{},"object":"chat.completion","created":{},"model":{},"choices":[{{"index":0,"message":{{"role":"assistant","content":{}}},"finish_reason":{}}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
            json_string(&self.id),
            self.created,
            json_string(&self.model),
            json_string(&self.content),
            json_string(self.finish_reason),
            self.prompt_tokens,
            self.completion_tokens,
            total,
        )
    }
}

/// A streaming SSE chunk.
pub struct StreamChunk {
    /// Completion ID.
    pub id: String,
    /// Model ID.
    pub model: String,
    /// Token text (empty for [DONE]).
    pub delta: String,
    /// True if this is the final chunk.
    pub done: bool,
    /// Finish reason (None unless done).
    pub finish_reason: Option<&'static str>,
}

impl StreamChunk {
    /// Serialize to SSE data line.
    pub fn to_sse(&self) -> String {
        if self.done {
            "data: [DONE]\n\n".to_string()
        } else {
            let fr_json = match self.finish_reason {
                Some(r) => json_string(r),
                None    => "null".to_string(),
            };
            let data = format!(
                r#"{{"id":{},"object":"chat.completion.chunk","model":{},"choices":[{{"index":0,"delta":{{"role":"assistant","content":{}}},"finish_reason":{}}}]}}"#,
                json_string(&self.id),
                json_string(&self.model),
                json_string(&self.delta),
                fr_json,
            );
            format!("data: {data}\n\n")
        }
    }
}

/// A text completion response (non-streaming).
pub struct CompletionResponse {
    /// Unique completion ID.
    pub id: String,
    /// Unix timestamp.
    pub created: u64,
    /// Model ID.
    pub model: String,
    /// Generated text.
    pub text: String,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of generated tokens.
    pub completion_tokens: usize,
    /// Finish reason.
    pub finish_reason: &'static str,
}

impl CompletionResponse {
    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        let total = self.prompt_tokens + self.completion_tokens;
        format!(
            r#"{{"id":{},"object":"text_completion","created":{},"model":{},"choices":[{{"text":{},"index":0,"finish_reason":{}}}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
            json_string(&self.id),
            self.created,
            json_string(&self.model),
            json_string(&self.text),
            json_string(self.finish_reason),
            self.prompt_tokens,
            self.completion_tokens,
            total,
        )
    }
}

/// Error response body.
pub struct ErrorResponse {
    /// Error message.
    pub message: String,
    /// Error type.
    pub error_type: &'static str,
    /// HTTP status code.
    pub status: u16,
}

impl ErrorResponse {
    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"error":{{"message":{},"type":{},"code":{}}}}}"#,
            json_string(&self.message),
            json_string(self.error_type),
            self.status,
        )
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Escape a string for JSON — surround with quotes, escape special chars.
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"'  => out.push_str(r#"\""#),
            '\\' => out.push_str(r#"\\"#),
            '\n' => out.push_str(r#"\n"#),
            '\r' => out.push_str(r#"\r"#),
            '\t' => out.push_str(r#"\t"#),
            c if (c as u32) < 32 => {
                out.push_str(&format!(r#"\u{:04x}"#, c as u32));
            }
            c    => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Generate a pseudo-unique ID string.
pub fn gen_id(prefix: &str) -> String {
    // Simple deterministic ID based on time (no random deps).
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}-{}{}", prefix, t.as_secs(), t.subsec_nanos() / 1_000_000)
}

/// Current Unix timestamp in seconds.
pub fn unix_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_string_basic() {
        assert_eq!(json_string("hello"), r#""hello""#);
    }

    #[test]
    fn json_string_escapes() {
        assert_eq!(json_string("say \"hi\""), r#""say \"hi\"""#);
        assert_eq!(json_string("line\nnew"), r#""line\nnew""#);
        assert_eq!(json_string("tab\there"), r#""tab\there""#);
    }

    #[test]
    fn chat_message_round_trip() {
        let json = r#"{"role":"user","content":"Hello!"}"#;
        let v = atlas_json::Json::parse(json).unwrap();
        let msg = ChatMessage::from_json(&v).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn chat_request_parse() {
        let body = r#"{"model":"atlas","messages":[{"role":"user","content":"Hi"}],"max_tokens":50,"temperature":0.7}"#;
        let req = ChatCompletionRequest::parse(body).unwrap();
        assert_eq!(req.model, "atlas");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.max_tokens, 50);
        assert!((req.temperature - 0.7).abs() < 0.01);
        assert!(!req.stream);
    }

    #[test]
    fn chat_request_stream_flag() {
        let body = r#"{"model":"atlas","messages":[],"stream":true}"#;
        let req = ChatCompletionRequest::parse(body).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn chat_request_defaults() {
        let body = r#"{"messages":[]}"#;
        let req = ChatCompletionRequest::parse(body).unwrap();
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 0.0);
        assert!(!req.stream);
    }

    #[test]
    fn completion_request_parse() {
        let body = r#"{"model":"atlas","prompt":"Hello world","max_tokens":100}"#;
        let req = CompletionRequest::parse(body).unwrap();
        assert_eq!(req.prompt, "Hello world");
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn chat_completion_response_json() {
        let resp = ChatCompletionResponse {
            id: "cmpl-1".to_string(),
            created: 1000,
            model: "atlas".to_string(),
            content: "Hello!".to_string(),
            prompt_tokens: 5,
            completion_tokens: 1,
            finish_reason: "stop",
        };
        let json = resp.to_json();
        let v = atlas_json::Json::parse(&json).unwrap();
        assert_eq!(v.get("object").and_then(|x| x.as_str()), Some("chat.completion"));
        let choice = &v.get("choices").unwrap().as_array().unwrap()[0];
        assert_eq!(choice.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()), Some("Hello!"));
        let usage = v.get("usage").unwrap();
        assert_eq!(usage.get("total_tokens").and_then(|x| x.as_i64()), Some(6));
    }

    #[test]
    fn completion_response_json() {
        let resp = CompletionResponse {
            id: "cmpl-2".to_string(),
            created: 2000,
            model: "atlas".to_string(),
            text: "World".to_string(),
            prompt_tokens: 3,
            completion_tokens: 1,
            finish_reason: "stop",
        };
        let json = resp.to_json();
        let v = atlas_json::Json::parse(&json).unwrap();
        assert_eq!(v.get("object").and_then(|x| x.as_str()), Some("text_completion"));
        let choice = &v.get("choices").unwrap().as_array().unwrap()[0];
        assert_eq!(choice.get("text").and_then(|c| c.as_str()), Some("World"));
    }

    #[test]
    fn stream_chunk_sse_format() {
        let chunk = StreamChunk {
            id: "cmpl-1".to_string(),
            model: "atlas".to_string(),
            delta: "Hi".to_string(),
            done: false,
            finish_reason: None,
        };
        let sse = chunk.to_sse();
        assert!(sse.starts_with("data: {"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn stream_chunk_done() {
        let chunk = StreamChunk {
            id: "cmpl-1".to_string(),
            model: "atlas".to_string(),
            delta: String::new(),
            done: true,
            finish_reason: Some("stop"),
        };
        assert_eq!(chunk.to_sse(), "data: [DONE]\n\n");
    }

    #[test]
    fn to_prompt_chatml() {
        let req = ChatCompletionRequest {
            model: "atlas".to_string(),
            messages: vec![
                ChatMessage { role: "system".to_string(), content: "You are helpful.".to_string() },
                ChatMessage { role: "user".to_string(), content: "What is 2+2?".to_string() },
            ],
            max_tokens: 50,
            temperature: 0.0,
            stream: false,
        };
        let prompt = req.to_prompt();
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("<|assistant|>"));
    }

    #[test]
    fn error_response_json() {
        let err = ErrorResponse {
            message: "Model not found".to_string(),
            error_type: "invalid_request_error",
            status: 404,
        };
        let json = err.to_json();
        let v = atlas_json::Json::parse(&json).unwrap();
        let e = v.get("error").unwrap();
        assert_eq!(e.get("message").and_then(|x| x.as_str()), Some("Model not found"));
        assert_eq!(e.get("type").and_then(|x| x.as_str()), Some("invalid_request_error"));
    }
}
