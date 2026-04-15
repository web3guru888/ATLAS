//! MCP JSON-RPC 2.0 protocol types — zero external dependencies.
//!
//! All types use [`atlas_json::Json`] for parsing and serialization.
//! Implements the standard MCP lifecycle messages:
//! - `initialize` — Server capability negotiation
//! - `tools/list` — Return all tool definitions
//! - `tools/call` — Dispatch to palace handlers

use atlas_json::Json;

/// MCP protocol version.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Server name reported during initialization.
pub const SERVER_NAME: &str = "atlas-palace";

/// Server version.
pub const SERVER_VERSION: &str = "0.1.0";

// ─── JSON-RPC Types ───────────────────────────────────────────────────────

/// A parsed JSON-RPC 2.0 request.
#[derive(Debug, Clone)]
pub struct JsonRpcRequest {
    /// Must be "2.0".
    pub jsonrpc: String,
    /// Request id (integer or string), or None for notifications.
    pub id: Option<Json>,
    /// Method name (e.g. "initialize", "tools/list", "tools/call").
    pub method: String,
    /// Optional parameters.
    pub params: Option<Json>,
}

impl JsonRpcRequest {
    /// Parse a JSON-RPC request from a [`Json`] value.
    pub fn from_json(v: &Json) -> Option<Self> {
        let jsonrpc = v.get("jsonrpc")?.as_str()?.to_string();
        let method = v.get("method")?.as_str()?.to_string();
        let id = v.get("id").cloned();
        let params = v.get("params").cloned();
        Some(Self { jsonrpc, id, method, params })
    }
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone)]
pub struct JsonRpcResponse {
    /// Always "2.0".
    pub jsonrpc: String,
    /// Matches the request id.
    pub id: Option<Json>,
    /// Success result (present if no error).
    pub result: Option<Json>,
    /// Error object (present if no result).
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone)]
pub struct JsonRpcError {
    /// Standard error code.
    pub code: i64,
    /// Human-readable error message.
    pub message: String,
}

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: Option<Json>, result: Json) -> Self {
        Self { jsonrpc: "2.0".to_string(), id, result: Some(result), error: None }
    }

    /// Create an error response.
    pub fn error(id: Option<Json>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError { code, message: message.into() }),
        }
    }

    /// Create a method-not-found error (-32601).
    pub fn method_not_found(id: Option<Json>, method: &str) -> Self {
        Self::error(id, -32601, format!("Method not found: {method}"))
    }

    /// Create an invalid-params error (-32602).
    pub fn invalid_params(id: Option<Json>, msg: impl Into<String>) -> Self {
        Self::error(id, -32602, msg)
    }

    /// Create an internal error (-32603).
    pub fn internal_error(id: Option<Json>, msg: impl Into<String>) -> Self {
        Self::error(id, -32603, msg)
    }

    /// Serialize to a [`Json`] value.
    pub fn to_json(&self) -> Json {
        let mut pairs: Vec<(String, Json)> = Vec::new();
        pairs.push(("jsonrpc".to_string(), Json::Str(self.jsonrpc.clone())));
        pairs.push(("id".to_string(), self.id.clone().unwrap_or(Json::Null)));
        if let Some(ref r) = self.result {
            pairs.push(("result".to_string(), r.clone()));
        }
        if let Some(ref e) = self.error {
            let mut err_pairs: Vec<(String, Json)> = Vec::new();
            err_pairs.push(("code".to_string(), Json::Int(e.code)));
            err_pairs.push(("message".to_string(), Json::Str(e.message.clone())));
            pairs.push(("error".to_string(), Json::Object(err_pairs)));
        }
        Json::Object(pairs)
    }

    /// Serialize to a compact JSON string.
    pub fn to_json_string(&self) -> String {
        self.to_json().to_json()
    }
}

// ─── MCP Result Structures ────────────────────────────────────────────────

/// Build the `initialize` response result as [`Json`].
pub fn initialize_result() -> Json {
    Json::Object(vec![
        ("protocolVersion".to_string(), Json::Str(MCP_PROTOCOL_VERSION.to_string())),
        ("serverInfo".to_string(), Json::Object(vec![
            ("name".to_string(), Json::Str(SERVER_NAME.to_string())),
            ("version".to_string(), Json::Str(SERVER_VERSION.to_string())),
        ])),
        ("capabilities".to_string(), Json::Object(vec![
            ("tools".to_string(), Json::Object(vec![
                ("listChanged".to_string(), Json::Bool(false)),
            ])),
        ])),
    ])
}

/// A tool result item (MCP tools/call response content).
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    /// Content items (typically a single text entry).
    pub content: Vec<ToolResultContent>,
    /// Whether this result is an error.
    pub is_error: bool,
}

/// A single content entry in a tool result.
#[derive(Debug, Clone)]
pub struct ToolResultContent {
    /// Content type (always "text" for now).
    pub content_type: String,
    /// Content body.
    pub text: String,
}

impl ToolCallResult {
    /// Create a successful text result.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent {
                content_type: "text".to_string(),
                text: text.into(),
            }],
            is_error: false,
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolResultContent {
                content_type: "text".to_string(),
                text: message.into(),
            }],
            is_error: true,
        }
    }

    /// Convert to [`Json`] for embedding in a JSON-RPC response.
    pub fn to_json(&self) -> Json {
        let content_arr: Vec<Json> = self.content.iter().map(|c| {
            Json::Object(vec![
                ("type".to_string(), Json::Str(c.content_type.clone())),
                ("text".to_string(), Json::Str(c.text.clone())),
            ])
        }).collect();
        let mut pairs: Vec<(String, Json)> = Vec::new();
        pairs.push(("content".to_string(), Json::Array(content_arr)));
        if self.is_error {
            pairs.push(("isError".to_string(), Json::Bool(true)));
        }
        Json::Object(pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_initialize_request() {
        let json = Json::parse(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#).unwrap();
        let req = JsonRpcRequest::from_json(&json).unwrap();
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(Json::Int(1)));
        assert!(req.params.is_some());
    }

    #[test]
    fn parse_tools_list_request() {
        let json = Json::parse(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#).unwrap();
        let req = JsonRpcRequest::from_json(&json).unwrap();
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.id, Some(Json::Int(2)));
        assert!(req.params.is_none());
    }

    #[test]
    fn parse_tools_call_request() {
        let json = Json::parse(r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"palace_search","arguments":{"query":"test"}}}"#).unwrap();
        let req = JsonRpcRequest::from_json(&json).unwrap();
        assert_eq!(req.method, "tools/call");
        let params = req.params.unwrap();
        assert_eq!(params.get("name").unwrap().as_str(), Some("palace_search"));
        let args = params.get("arguments").unwrap();
        assert_eq!(args.get("query").unwrap().as_str(), Some("test"));
    }

    #[test]
    fn serialize_initialize_response() {
        let result = initialize_result();
        let resp = JsonRpcResponse::success(Some(Json::Int(1)), result);
        let s = resp.to_json_string();
        assert!(s.contains("protocolVersion"));
        assert!(s.contains(MCP_PROTOCOL_VERSION));
        assert!(s.contains(SERVER_NAME));
    }

    #[test]
    fn serialize_error_response() {
        let resp = JsonRpcResponse::method_not_found(Some(Json::Int(99)), "bogus");
        let s = resp.to_json_string();
        assert!(s.contains("-32601"));
        assert!(s.contains("bogus"));
    }

    #[test]
    fn tool_call_result_text() {
        let r = ToolCallResult::text("hello world");
        let j = r.to_json();
        let content = j.get("content").unwrap().as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0].get("text").unwrap().as_str(), Some("hello world"));
        assert!(j.get("isError").is_none());
    }

    #[test]
    fn tool_call_result_error() {
        let r = ToolCallResult::error("something broke");
        let j = r.to_json();
        assert_eq!(j.get("isError").unwrap().as_bool(), Some(true));
        let text = j.get("content").unwrap().index(0).unwrap().get("text").unwrap().as_str().unwrap();
        assert_eq!(text, "something broke");
    }
}
