//! OpenAI-compatible request/response types.
//!
//! All serialization is hand-rolled — no serde or external deps.
//! Types mirror the OpenAI v1 REST API spec so any OpenAI-compatible
//! client can talk to ATLAS out of the box.

use atlas_core::{AtlasError, Result};
use atlas_json::Json;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind host (e.g. "0.0.0.0").
    pub host: String,
    /// TCP port (default 8080).
    pub port: u16,
    /// Model identifier returned by /v1/models.
    pub model_id: String,
    /// Directory containing model weights and tokenizer.json.
    pub weights_dir: Option<String>,
    /// Max tokens per generation (hard cap).
    pub max_tokens: usize,
    /// Number of worker threads.
    pub workers: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host:        "0.0.0.0".to_string(),
            port:        8080,
            model_id:    "atlas".to_string(),
            weights_dir: None,
            max_tokens:  2048,
            workers:     4,
        }
    }
}

// ── Chat template ────────────────────────────────────────────────────────────

/// Chat template format for converting messages → prompt text.
///
/// Different model families expect different framing tokens around chat turns.
/// Using the wrong template produces garbage output even with a perfect model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>\n`
    /// Used by: OLMo-3, SmolLM2-Instruct, Qwen, many HF models.
    ChatML,
    /// Llama-3 style: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// Generic fallback: `<|role|>\ncontent\n`
    Generic,
}

impl Default for ChatTemplate {
    fn default() -> Self { Self::ChatML }
}

// ── Chat message ─────────────────────────────────────────────────────────────

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
}

// ── Request types ─────────────────────────────────────────────────────────────

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

    /// Build a prompt string using the default ChatML template.
    pub fn to_prompt(&self) -> String {
        self.to_prompt_with(&ChatTemplate::default())
    }

    /// Build a prompt string using the specified chat template.
    ///
    /// The template must match the model's training format — using the wrong
    /// one produces incoherent output even with a perfect model.
    pub fn to_prompt_with(&self, template: &ChatTemplate) -> String {
        let mut prompt = String::new();
        match template {
            ChatTemplate::ChatML => {
                // <|im_start|>role\ncontent<|im_end|>\n
                for msg in &self.messages {
                    prompt.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        msg.role, msg.content
                    ));
                }
                // Prompt for assistant response
                prompt.push_str("<|im_start|>assistant\n");
            }
            ChatTemplate::Llama3 => {
                // <|begin_of_text|> is typically prepended by the tokenizer.
                for msg in &self.messages {
                    prompt.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        msg.role, msg.content
                    ));
                }
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatTemplate::Generic => {
                for msg in &self.messages {
                    match msg.role.as_str() {
                        "system"    => prompt.push_str(&format!("<|system|>\n{}\n", msg.content)),
                        "user"      => prompt.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", msg.content)),
                        "assistant" => prompt.push_str(&format!("{}\n", msg.content)),
                        _           => prompt.push_str(&format!("{}\n", msg.content)),
                    }
                }
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
            .and_then(|p| p.as_str())
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
            concat!(
                r#"{{"id":{id},"object":"chat.completion","created":{created},"model":{model},"#,
                r#""choices":[{{"index":0,"message":{{"role":"assistant","content":{content}}},"#,
                r#""finish_reason":{finish}}}],"#,
                r#""usage":{{"prompt_tokens":{pt},"completion_tokens":{ct},"total_tokens":{tt}}}}}"#
            ),
            id      = json_string(&self.id),
            created = self.created,
            model   = json_string(&self.model),
            content = json_string(&self.content),
            finish  = json_string(self.finish_reason),
            pt      = self.prompt_tokens,
            ct      = self.completion_tokens,
            tt      = total,
        )
    }
}

/// A streaming SSE chunk (OpenAI format).
pub struct StreamChunk {
    /// Completion ID.
    pub id: String,
    /// Model ID.
    pub model: String,
    /// Token text (empty for [DONE]).
    pub delta: String,
    /// True if this is the final [DONE] sentinel.
    pub done: bool,
    /// Finish reason (present on last content chunk).
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
                concat!(
                    r#"{{"id":{id},"object":"chat.completion.chunk","model":{model},"#,
                    r#""choices":[{{"index":0,"delta":{{"role":"assistant","content":{content}}},"#,
                    r#""finish_reason":{finish}}}]}}"#
                ),
                id      = json_string(&self.id),
                model   = json_string(&self.model),
                content = json_string(&self.delta),
                finish  = fr_json,
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
            concat!(
                r#"{{"id":{id},"object":"text_completion","created":{created},"model":{model},"#,
                r#""choices":[{{"text":{text},"index":0,"finish_reason":{finish}}}],"#,
                r#""usage":{{"prompt_tokens":{pt},"completion_tokens":{ct},"total_tokens":{tt}}}}}"#
            ),
            id      = json_string(&self.id),
            created = self.created,
            model   = json_string(&self.model),
            text    = json_string(&self.text),
            finish  = json_string(self.finish_reason),
            pt      = self.prompt_tokens,
            ct      = self.completion_tokens,
            tt      = total,
        )
    }
}

/// API error response body.
pub struct ErrorResponse {
    /// Error message.
    pub message: String,
    /// OpenAI error type string.
    pub error_type: &'static str,
    /// HTTP status code.
    pub status: u16,
}

impl ErrorResponse {
    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"error":{{"message":{msg},"type":{tp},"code":{code}}}}}"#,
            msg  = json_string(&self.message),
            tp   = json_string(self.error_type),
            code = self.status,
        )
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Escape a string for JSON — surround with double-quotes, escape specials.
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 32 => { out.push_str(&format!("\\u{:04x}", c as u32)); }
            c    => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Generate a pseudo-unique ID string from wall-clock time.
pub fn gen_id(prefix: &str) -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}-{}{:06}", prefix, t.as_secs(), t.subsec_micros())
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
    fn json_string_escapes_quotes() {
        assert_eq!(json_string(r#"say "hi""#), r#""say \"hi\"""#);
    }

    #[test]
    fn json_string_escapes_newline() {
        assert_eq!(json_string("line\nnew"), r#""line\nnew""#);
    }

    #[test]
    fn json_string_escapes_tab() {
        assert_eq!(json_string("tab\there"), r#""tab\there""#);
    }

    #[test]
    fn json_string_escapes_backslash() {
        assert_eq!(json_string("a\\b"), r#""a\\b""#);
    }

    #[test]
    fn chat_message_from_json() {
        let raw = r#"{"role":"user","content":"Hello!"}"#;
        let v = Json::parse(raw).unwrap();
        let msg = ChatMessage::from_json(&v).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn chat_message_missing_role_returns_none() {
        let v = Json::parse(r#"{"content":"Hi"}"#).unwrap();
        assert!(ChatMessage::from_json(&v).is_none());
    }

    #[test]
    fn chat_request_parse_full() {
        let body = r#"{"model":"atlas","messages":[{"role":"user","content":"Hi"}],"max_tokens":50,"temperature":0.7,"stream":false}"#;
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
        let req = ChatCompletionRequest::parse(r#"{"messages":[]}"#).unwrap();
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 0.0);
        assert!(!req.stream);
    }

    #[test]
    fn chat_request_invalid_json_errors() {
        assert!(ChatCompletionRequest::parse("{bad json}").is_err());
    }

    #[test]
    fn completion_request_parse() {
        let body = r#"{"model":"atlas","prompt":"Hello","max_tokens":100}"#;
        let req = CompletionRequest::parse(body).unwrap();
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn to_prompt_chatml() {
        let req = ChatCompletionRequest {
            model: "atlas".to_string(),
            messages: vec![
                ChatMessage { role: "system".to_string(),    content: "You are helpful.".to_string() },
                ChatMessage { role: "user".to_string(),      content: "What is 2+2?".to_string() },
            ],
            max_tokens: 50, temperature: 0.0, stream: false,
        };
        // Default is ChatML
        let p = req.to_prompt();
        assert!(p.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(p.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn to_prompt_llama3() {
        let req = ChatCompletionRequest {
            model: "atlas".to_string(),
            messages: vec![
                ChatMessage { role: "user".to_string(), content: "Hi".to_string() },
            ],
            max_tokens: 50, temperature: 0.0, stream: false,
        };
        let p = req.to_prompt_with(&ChatTemplate::Llama3);
        assert!(p.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(p.contains("Hi<|eot_id|>"));
        assert!(p.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn to_prompt_generic() {
        let req = ChatCompletionRequest {
            model: "atlas".to_string(),
            messages: vec![
                ChatMessage { role: "system".to_string(), content: "You are helpful.".to_string() },
                ChatMessage { role: "user".to_string(),   content: "Hello".to_string() },
            ],
            max_tokens: 50, temperature: 0.0, stream: false,
        };
        let p = req.to_prompt_with(&ChatTemplate::Generic);
        assert!(p.contains("<|system|>"));
        assert!(p.contains("<|user|>"));
        assert!(p.contains("<|assistant|>"));
    }

    #[test]
    fn chat_completion_response_json_valid() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-1".to_string(),
            created: 1000,
            model: "atlas".to_string(),
            content: "Hello!".to_string(),
            prompt_tokens: 5,
            completion_tokens: 1,
            finish_reason: "stop",
        };
        let json = resp.to_json();
        let v = Json::parse(&json).unwrap();
        assert_eq!(v.get("object").and_then(|x| x.as_str()), Some("chat.completion"));
        let choices = v.get("choices").unwrap().as_array().unwrap();
        let msg = choices[0].get("message").unwrap();
        assert_eq!(msg.get("content").and_then(|c| c.as_str()), Some("Hello!"));
        assert_eq!(msg.get("role").and_then(|r| r.as_str()), Some("assistant"));
        let usage = v.get("usage").unwrap();
        assert_eq!(usage.get("total_tokens").and_then(|x| x.as_i64()), Some(6));
    }

    #[test]
    fn completion_response_json_valid() {
        let resp = CompletionResponse {
            id: "cmpl-1".to_string(),
            created: 2000,
            model: "atlas".to_string(),
            text: "World".to_string(),
            prompt_tokens: 3,
            completion_tokens: 1,
            finish_reason: "stop",
        };
        let json = resp.to_json();
        let v = Json::parse(&json).unwrap();
        assert_eq!(v.get("object").and_then(|x| x.as_str()), Some("text_completion"));
        let choices = v.get("choices").unwrap().as_array().unwrap();
        assert_eq!(choices[0].get("text").and_then(|t| t.as_str()), Some("World"));
    }

    #[test]
    fn stream_chunk_sse_format() {
        let chunk = StreamChunk {
            id: "chatcmpl-1".to_string(),
            model: "atlas".to_string(),
            delta: "Hi".to_string(),
            done: false,
            finish_reason: None,
        };
        let sse = chunk.to_sse();
        assert!(sse.starts_with("data: {"));
        assert!(sse.ends_with("\n\n"));
        let data_json = &sse["data: ".len()..sse.len()-2];
        let v = Json::parse(data_json).unwrap();
        assert_eq!(v.get("object").and_then(|x| x.as_str()), Some("chat.completion.chunk"));
    }

    #[test]
    fn stream_chunk_done_sentinel() {
        let chunk = StreamChunk {
            id: "x".to_string(), model: "atlas".to_string(),
            delta: String::new(), done: true, finish_reason: Some("stop"),
        };
        assert_eq!(chunk.to_sse(), "data: [DONE]\n\n");
    }

    #[test]
    fn error_response_json_valid() {
        let err = ErrorResponse {
            message: "Model not found".to_string(),
            error_type: "invalid_request_error",
            status: 404,
        };
        let json = err.to_json();
        let v = Json::parse(&json).unwrap();
        let e = v.get("error").unwrap();
        assert_eq!(e.get("message").and_then(|x| x.as_str()), Some("Model not found"));
        assert_eq!(e.get("type").and_then(|x| x.as_str()), Some("invalid_request_error"));
        assert_eq!(e.get("code").and_then(|x| x.as_i64()), Some(404));
    }

    #[test]
    fn gen_id_prefixed() {
        let id = gen_id("chatcmpl");
        assert!(id.starts_with("chatcmpl-"));
    }

    #[test]
    fn server_config_default_values() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.max_tokens, 2048);
        assert!(cfg.weights_dir.is_none());
    }
}
