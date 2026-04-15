//! MCP server implementation — synchronous stdin/stdout transport.
//!
//! Handles the standard MCP lifecycle:
//! - `initialize` → server capabilities
//! - `tools/list` → return all 28 tool definitions
//! - `tools/call` → dispatch to palace methods
//!
//! Uses [`atlas_json::Json`] for all JSON parsing and serialization.
//! No tokio, no async — simple synchronous line-delimited JSON.

use atlas_json::Json;
use atlas_palace::Palace;

use crate::protocol::{
    initialize_result, JsonRpcRequest, JsonRpcResponse, ToolCallResult,
};
use crate::tools::{tool_catalog, ToolDefinition};

// ─── MCP Server ───────────────────────────────────────────────────────────

/// The ATLAS MCP server.
///
/// Wraps a [`Palace`] and dispatches MCP tool calls to palace methods.
pub struct McpServer {
    /// The backing palace instance.
    palace: Palace,
    /// Tool catalog (28 tools).
    tools: Vec<ToolDefinition>,
    /// Whether the server has been initialized (received `initialize`).
    initialized: bool,
}

impl McpServer {
    /// Create a new MCP server wrapping a palace.
    pub fn new(palace: Palace) -> Self {
        Self {
            palace,
            tools: tool_catalog(),
            initialized: false,
        }
    }

    /// Whether the server has received an `initialize` message.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Mutable access to the underlying palace.
    pub fn palace_mut(&mut self) -> &mut Palace {
        &mut self.palace
    }

    /// Immutable access to the underlying palace.
    pub fn palace(&self) -> &Palace {
        &self.palace
    }

    /// Get the number of registered tools.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Get a list of all tool names.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    // ─── Message Handling ─────────────────────────────────────────────

    /// Process a raw JSON string and return a JSON response string.
    pub fn handle_json(&mut self, input: &str) -> String {
        match Json::parse(input) {
            Ok(v) => {
                match JsonRpcRequest::from_json(&v) {
                    Some(req) => {
                        let resp = self.handle_request(&req);
                        resp.to_json_string()
                    }
                    None => {
                        let resp = JsonRpcResponse::error(None, -32600, "Invalid JSON-RPC request");
                        resp.to_json_string()
                    }
                }
            }
            Err(e) => {
                let resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {e}"));
                resp.to_json_string()
            }
        }
    }

    /// Process a parsed JSON-RPC request and return a response.
    pub fn handle_request(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "initialized" => {
                // Notification — return success with null result
                JsonRpcResponse::success(request.id.clone(), Json::Null)
            }
            "tools/list" => self.handle_tools_list(request),
            "tools/call" => self.handle_tools_call(request),
            _ => JsonRpcResponse::method_not_found(request.id.clone(), &request.method),
        }
    }

    // ─── Method Handlers ──────────────────────────────────────────────

    fn handle_initialize(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        self.initialized = true;
        JsonRpcResponse::success(request.id.clone(), initialize_result())
    }

    fn handle_tools_list(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let tools_json: Vec<Json> = self.tools.iter().map(|t| {
            Json::Object(vec![
                ("name".to_string(), Json::Str(t.name.clone())),
                ("description".to_string(), Json::Str(t.description.clone())),
                ("inputSchema".to_string(), t.parameters.clone()),
            ])
        }).collect();
        let result = Json::Object(vec![
            ("tools".to_string(), Json::Array(tools_json)),
        ]);
        JsonRpcResponse::success(request.id.clone(), result)
    }

    fn handle_tools_call(&mut self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params = match &request.params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::invalid_params(
                    request.id.clone(),
                    "Missing params for tools/call",
                );
            }
        };
        let name = match params.get("name").and_then(|n| n.as_str()) {
            Some(n) => n,
            None => {
                return JsonRpcResponse::invalid_params(
                    request.id.clone(),
                    "Missing 'name' in tools/call params",
                );
            }
        };
        let arguments = params.get("arguments");
        let result = self.dispatch_tool(name, arguments);
        JsonRpcResponse::success(request.id.clone(), result.to_json())
    }

    // ─── Tool Dispatch ────────────────────────────────────────────────

    /// Dispatch a tool call to the appropriate palace method.
    pub fn dispatch_tool(&mut self, name: &str, arguments: Option<&Json>) -> ToolCallResult {
        // Verify tool exists
        if !self.tools.iter().any(|t| t.name == name) {
            return ToolCallResult::error(format!("Unknown tool: {name}"));
        }

        match name {
            // ── Palace Navigation (read) ────────────────────────────
            "palace_status" => {
                ToolCallResult::text(self.palace.status())
            }

            "palace_list_wings" => {
                let wings = self.palace.list_wings();
                let items: Vec<Json> = wings.iter().map(|(id, name)| {
                    Json::Object(vec![
                        ("id".to_string(), Json::Str(id.clone())),
                        ("name".to_string(), Json::Str(name.clone())),
                    ])
                }).collect();
                let result = Json::Object(vec![
                    ("wings".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_list_rooms" => {
                let wing_id = match get_str(arguments, "wing_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: wing_id"),
                };
                let rooms = self.palace.list_rooms(wing_id);
                let items: Vec<Json> = rooms.iter().map(|(id, name)| {
                    Json::Object(vec![
                        ("id".to_string(), Json::Str(id.clone())),
                        ("name".to_string(), Json::Str(name.clone())),
                    ])
                }).collect();
                let result = Json::Object(vec![
                    ("wing_id".to_string(), Json::Str(wing_id.to_string())),
                    ("rooms".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_get_taxonomy" => {
                // Build full taxonomy: wings → rooms → drawer counts
                let wings = self.palace.list_wings();
                let items: Vec<Json> = wings.iter().map(|(wid, wname)| {
                    let rooms = self.palace.list_rooms(wid);
                    let room_items: Vec<Json> = rooms.iter().map(|(rid, rname)| {
                        Json::Object(vec![
                            ("id".to_string(), Json::Str(rid.clone())),
                            ("name".to_string(), Json::Str(rname.clone())),
                        ])
                    }).collect();
                    Json::Object(vec![
                        ("id".to_string(), Json::Str(wid.clone())),
                        ("name".to_string(), Json::Str(wname.clone())),
                        ("rooms".to_string(), Json::Array(room_items)),
                    ])
                }).collect();
                let result = Json::Object(vec![
                    ("taxonomy".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_search" => {
                let query = match get_str(arguments, "query") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: query"),
                };
                let top_k = get_usize(arguments, "top_k").unwrap_or(10);
                let results = self.palace.search(query, top_k);
                let items: Vec<Json> = results.iter().map(|r| {
                    Json::Object(vec![
                        ("drawer_id".to_string(), Json::Str(r.drawer_id.clone())),
                        ("score".to_string(), Json::Float(r.score as f64)),
                        ("preview".to_string(), Json::Str(r.preview.clone())),
                    ])
                }).collect();
                let result = Json::Object(vec![
                    ("query".to_string(), Json::Str(query.to_string())),
                    ("results".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_navigate" => {
                let from_id = match get_str(arguments, "from_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: from_id"),
                };
                let to_id = match get_str(arguments, "to_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: to_id"),
                };
                let max_steps = get_usize(arguments, "max_steps").unwrap_or(20);
                let path = self.palace.navigate(from_id, to_id, max_steps);
                let result = Json::Object(vec![
                    ("from".to_string(), Json::Str(from_id.to_string())),
                    ("to".to_string(), Json::Str(to_id.to_string())),
                    ("path".to_string(), Json::Array(
                        path.iter().map(|s| Json::Str(s.clone())).collect()
                    )),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_find_similar" => {
                let drawer_id = match get_str(arguments, "drawer_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: drawer_id"),
                };
                let top_k = get_usize(arguments, "top_k").unwrap_or(10);
                let results = self.palace.find_similar(drawer_id, top_k);
                let items: Vec<Json> = results.iter().map(|r| {
                    Json::Object(vec![
                        ("drawer_id".to_string(), Json::Str(r.drawer_id.clone())),
                        ("score".to_string(), Json::Float(r.score as f64)),
                        ("preview".to_string(), Json::Str(r.preview.clone())),
                    ])
                }).collect();
                let result = Json::Object(vec![
                    ("similar_to".to_string(), Json::Str(drawer_id.to_string())),
                    ("results".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_graph_stats" => {
                let stats = self.palace.status_dict();
                let sim_edges = self.palace.similarity_edge_count();
                let pairs: Vec<(String, Json)> = stats.iter()
                    .map(|(k, v)| (k.clone(), Json::Int(*v as i64)))
                    .chain(std::iter::once(("similarity_edges".to_string(), Json::Int(sim_edges as i64))))
                    .collect();
                ToolCallResult::text(Json::Object(pairs).to_json())
            }

            // ── Palace Operations (write) ───────────────────────────
            "palace_add_wing" => {
                let name = match get_str(arguments, "name") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: name"),
                };
                let desc = get_str(arguments, "description").unwrap_or("");
                let id = self.palace.add_wing(name, desc);
                let result = Json::Object(vec![
                    ("wing_id".to_string(), Json::Str(id)),
                    ("status".to_string(), Json::Str("created".to_string())),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_add_room" => {
                let wing_id = match get_str(arguments, "wing_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: wing_id"),
                };
                let name = match get_str(arguments, "name") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: name"),
                };
                let desc = get_str(arguments, "description").unwrap_or("");
                match self.palace.add_room(wing_id, name, desc) {
                    Ok(id) => {
                        let result = Json::Object(vec![
                            ("room_id".to_string(), Json::Str(id)),
                            ("status".to_string(), Json::Str("created".to_string())),
                        ]);
                        ToolCallResult::text(result.to_json())
                    }
                    Err(e) => ToolCallResult::error(format!("{e}")),
                }
            }

            "palace_add_drawer" => {
                let room_id = match get_str(arguments, "room_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: room_id"),
                };
                let title = match get_str(arguments, "title") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: title"),
                };
                let content = match get_str(arguments, "content") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: content"),
                };
                match self.palace.add_drawer(room_id, title, content, &[]) {
                    Ok(id) => {
                        let result = Json::Object(vec![
                            ("drawer_id".to_string(), Json::Str(id)),
                            ("status".to_string(), Json::Str("created".to_string())),
                        ]);
                        ToolCallResult::text(result.to_json())
                    }
                    Err(e) => ToolCallResult::error(format!("{e}")),
                }
            }

            "palace_add_drawer_if_unique" => {
                let room_id = match get_str(arguments, "room_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: room_id"),
                };
                let title = match get_str(arguments, "title") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: title"),
                };
                let content = match get_str(arguments, "content") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: content"),
                };
                let threshold = get_f32(arguments, "threshold").unwrap_or(0.85);
                match self.palace.add_drawer_if_unique(room_id, title, content, &[], threshold) {
                    Ok((id, is_new)) => {
                        let result = Json::Object(vec![
                            ("drawer_id".to_string(), Json::Str(id)),
                            ("is_new".to_string(), Json::Bool(is_new)),
                        ]);
                        ToolCallResult::text(result.to_json())
                    }
                    Err(e) => ToolCallResult::error(format!("{e}")),
                }
            }

            "palace_check_duplicate" => {
                let room_id = match get_str(arguments, "room_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: room_id"),
                };
                let content = match get_str(arguments, "content") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: content"),
                };
                let threshold = get_f32(arguments, "threshold").unwrap_or(0.85);
                let dup = self.palace.check_duplicate(room_id, content, threshold);
                let result = Json::Object(vec![
                    ("duplicate_of".to_string(), match dup {
                        Some(id) => Json::Str(id),
                        None => Json::Null,
                    }),
                ]);
                ToolCallResult::text(result.to_json())
            }

            // ── Knowledge Graph ─────────────────────────────────────
            "palace_kg_add" => {
                let from = match get_str(arguments, "from") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: from"),
                };
                let to = match get_str(arguments, "to") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: to"),
                };
                let relation = match get_str(arguments, "relation") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: relation"),
                };
                let confidence = get_f32(arguments, "confidence").unwrap_or(1.0);
                self.palace.kg_add(from, to, relation, confidence);
                ToolCallResult::text("{\"status\":\"added\"}")
            }

            "palace_kg_add_temporal" => {
                let from = match get_str(arguments, "from") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: from"),
                };
                let to = match get_str(arguments, "to") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: to"),
                };
                let relation = match get_str(arguments, "relation") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: relation"),
                };
                let confidence = get_f32(arguments, "confidence").unwrap_or(1.0);
                let timestamp = get_usize(arguments, "timestamp").unwrap_or(0) as u64;
                self.palace.kg_add_temporal(from, to, relation, confidence, timestamp);
                ToolCallResult::text("{\"status\":\"added\"}")
            }

            "palace_kg_query" => {
                let from = match get_str(arguments, "from") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: from"),
                };
                let edges = self.palace.kg_query(from);
                let items: Vec<Json> = edges.iter().map(|e| {
                    let mut pairs = vec![
                        ("from".to_string(), Json::Str(e.from.clone())),
                        ("to".to_string(), Json::Str(e.to.clone())),
                        ("relation".to_string(), Json::Str(e.relation.clone())),
                        ("confidence".to_string(), Json::Float(e.confidence as f64)),
                    ];
                    if let Some(ts) = e.timestamp {
                        pairs.push(("timestamp".to_string(), Json::Int(ts as i64)));
                    }
                    Json::Object(pairs)
                }).collect();
                let result = Json::Object(vec![
                    ("from".to_string(), Json::Str(from.to_string())),
                    ("edges".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_kg_contradictions" => {
                let threshold = get_f32(arguments, "threshold").unwrap_or(1.0);
                let contras = self.palace.kg_contradictions(threshold);
                let items: Vec<Json> = contras.iter().map(|(from, to, desc)| {
                    Json::Object(vec![
                        ("from".to_string(), Json::Str(from.clone())),
                        ("to".to_string(), Json::Str(to.clone())),
                        ("description".to_string(), Json::Str(desc.clone())),
                    ])
                }).collect();
                let result = Json::Object(vec![
                    ("contradictions".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_kg_invalidate" => {
                let drawer_id = match get_str(arguments, "drawer_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: drawer_id"),
                };
                self.palace.kg_invalidate(drawer_id);
                ToolCallResult::text("{\"status\":\"invalidated\"}")
            }

            "palace_build_similarity_graph" => {
                let threshold = get_f32(arguments, "threshold").unwrap_or(0.5);
                let added = self.palace.build_similarity_graph(threshold);
                let result = Json::Object(vec![
                    ("edges_added".to_string(), Json::Int(added as i64)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_build_tunnels" => {
                let min_pheromone = get_f32(arguments, "min_pheromone").unwrap_or(0.3);
                let added = self.palace.build_tunnels(min_pheromone);
                let result = Json::Object(vec![
                    ("tunnels_built".to_string(), Json::Int(added as i64)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            // ── Stigmergy ───────────────────────────────────────────
            "palace_deposit_pheromones" => {
                let drawer_id = match get_str(arguments, "drawer_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: drawer_id"),
                };
                let value = get_f32(arguments, "value").unwrap_or(0.5);
                let decay = get_f32(arguments, "decay").unwrap_or(0.05);
                let tag = get_str(arguments, "tag").unwrap_or("default");
                self.palace.deposit_pheromones(drawer_id, value, decay, tag);
                ToolCallResult::text("{\"status\":\"deposited\"}")
            }

            "palace_decay_pheromones" => {
                self.palace.decay_pheromones();
                ToolCallResult::text("{\"status\":\"decayed\"}")
            }

            "palace_hot_paths" => {
                let tag = get_str(arguments, "tag").unwrap_or("");
                let top_k = get_usize(arguments, "top_k").unwrap_or(10);
                let paths = self.palace.hot_paths(tag, top_k);
                let items: Vec<Json> = paths.iter().map(|(id, val)| {
                    Json::Object(vec![
                        ("drawer_id".to_string(), Json::Str(id.clone())),
                        ("pheromone".to_string(), Json::Float(*val as f64)),
                    ])
                }).collect();
                let result = Json::Object(vec![
                    ("hot_paths".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_cold_spots" => {
                let threshold = get_f32(arguments, "threshold").unwrap_or(0.1);
                let top_k = get_usize(arguments, "top_k").unwrap_or(10);
                let spots = self.palace.cold_spots(threshold, top_k);
                let items: Vec<Json> = spots.iter().map(|id| Json::Str(id.clone())).collect();
                let result = Json::Object(vec![
                    ("cold_spots".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_pheromone_status" => {
                // Return overall status; if a specific drawer is given, include its pheromone info
                let status = self.palace.status();
                ToolCallResult::text(status)
            }

            // ── Agent Diary ─────────────────────────────────────────
            "palace_create_agent" => {
                let id = match get_str(arguments, "id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: id"),
                };
                let name = match get_str(arguments, "name") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: name"),
                };
                let role = match get_str(arguments, "role") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: role"),
                };
                let home_room = match get_str(arguments, "home_room") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: home_room"),
                };
                self.palace.create_agent(id, name, role, home_room);
                let result = Json::Object(vec![
                    ("agent_id".to_string(), Json::Str(id.to_string())),
                    ("status".to_string(), Json::Str("created".to_string())),
                ]);
                ToolCallResult::text(result.to_json())
            }

            "palace_diary_write" => {
                let agent_id = match get_str(arguments, "agent_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: agent_id"),
                };
                let text = match get_str(arguments, "text") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: text"),
                };
                match self.palace.diary_write(agent_id, text, &[]) {
                    Ok(()) => ToolCallResult::text("{\"status\":\"written\"}"),
                    Err(e) => ToolCallResult::error(format!("{e}")),
                }
            }

            "palace_diary_read" => {
                let agent_id = match get_str(arguments, "agent_id") {
                    Some(s) => s,
                    None => return ToolCallResult::error("Missing required parameter: agent_id"),
                };
                let n = get_usize(arguments, "n").unwrap_or(10);
                let entries = self.palace.diary_read(agent_id, n);
                let items: Vec<Json> = entries.iter().map(|e| {
                    Json::Object(vec![
                        ("agent_id".to_string(), Json::Str(e.agent_id.clone())),
                        ("text".to_string(), Json::Str(e.text.clone())),
                        ("timestamp".to_string(), Json::Int(e.timestamp as i64)),
                    ])
                }).collect();
                let result = Json::Object(vec![
                    ("agent_id".to_string(), Json::Str(agent_id.to_string())),
                    ("entries".to_string(), Json::Array(items)),
                ]);
                ToolCallResult::text(result.to_json())
            }

            _ => ToolCallResult::error(format!("Tool '{name}' exists but has no handler")),
        }
    }

    /// Run the MCP server on stdin/stdout.
    ///
    /// Reads line-delimited JSON from stdin, processes each message,
    /// and writes the response as a single line to stdout.
    pub fn run_stdio(&mut self) -> std::io::Result<()> {
        use std::io::{self, BufRead, Write};

        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();

        for line in stdin.lock().lines() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let response = self.handle_json(trimmed);
            writeln!(stdout, "{response}")?;
            stdout.flush()?;
        }
        Ok(())
    }
}

// ─── Argument helpers ─────────────────────────────────────────────────────

/// Extract a string argument from a JSON object.
fn get_str<'a>(args: Option<&'a Json>, key: &str) -> Option<&'a str> {
    args?.get(key)?.as_str()
}

/// Extract a usize argument from a JSON object.
fn get_usize(args: Option<&Json>, key: &str) -> Option<usize> {
    args?.get(key)?.as_usize()
}

/// Extract an f32 argument from a JSON object.
fn get_f32(args: Option<&Json>, key: &str) -> Option<f32> {
    args?.get(key)?.as_f64().map(|f| f as f32)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_server() -> McpServer {
        let mut palace = Palace::new("test", "/tmp/atlas-mcp-test");
        let w = palace.add_wing("research", "Research wing");
        let _ = palace.add_room(&w, "findings", "Research findings").unwrap();
        McpServer::new(palace)
    }

    fn make_request(id: i64, method: &str, params: Option<&str>) -> String {
        let params_str = match params {
            Some(p) => format!(",\"params\":{p}"),
            None => String::new(),
        };
        format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"{method}\"{params_str}}}")
    }

    #[test]
    fn server_creation() {
        let s = test_server();
        assert_eq!(s.tool_count(), 28);
        assert!(!s.is_initialized());
    }

    #[test]
    fn initialize_sets_flag() {
        let mut s = test_server();
        let req = make_request(1, "initialize", None);
        let resp = s.handle_json(&req);
        assert!(resp.contains("protocolVersion"));
        assert!(resp.contains("atlas-palace"));
        assert!(s.is_initialized());
    }

    #[test]
    fn serialize_tools_list_response() {
        let mut s = test_server();
        let req = make_request(2, "tools/list", None);
        let resp = s.handle_json(&req);
        let parsed = Json::parse(&resp).unwrap();
        let tools = parsed.get("result").unwrap().get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 28);
        // Each tool should have name, description, inputSchema
        for t in tools {
            assert!(t.get("name").unwrap().as_str().is_some());
            assert!(t.get("description").unwrap().as_str().is_some());
            assert!(t.get("inputSchema").is_some());
        }
    }

    #[test]
    fn serialize_tools_call_response() {
        let mut s = test_server();
        let req = make_request(3, "tools/call",
            Some(r#"{"name":"palace_status","arguments":{}}"#));
        let resp = s.handle_json(&req);
        let parsed = Json::parse(&resp).unwrap();
        let result = parsed.get("result").unwrap();
        let content = result.get("content").unwrap().as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0].get("type").unwrap().as_str(), Some("text"));
        // Status text should mention the palace name
        let text = content[0].get("text").unwrap().as_str().unwrap();
        assert!(text.contains("test"));
    }

    #[test]
    fn dispatch_add_wing_tool() {
        let mut s = test_server();
        let result = s.dispatch_tool("palace_add_wing", Some(&Json::Object(vec![
            ("name".to_string(), Json::Str("engineering".to_string())),
            ("description".to_string(), Json::Str("Engineering wing".to_string())),
        ])));
        assert!(!result.is_error);
        let text = &result.content[0].text;
        assert!(text.contains("wing_id"));
        assert!(text.contains("engineering"));
    }

    #[test]
    fn dispatch_search_tool() {
        let mut s = test_server();
        // Add some content to search
        let wing_id = s.palace_mut().add_wing("test-wing", "desc");
        let room_id = s.palace_mut().add_room(&wing_id, "test-room", "desc").unwrap();
        s.palace_mut().add_drawer(&room_id, "Hello", "Hello world content", &[]).unwrap();

        let result = s.dispatch_tool("palace_search", Some(&Json::Object(vec![
            ("query".to_string(), Json::Str("hello".to_string())),
            ("top_k".to_string(), Json::Int(5)),
        ])));
        assert!(!result.is_error);
        let text = &result.content[0].text;
        assert!(text.contains("results"));
        assert!(text.contains("hello"));
    }

    #[test]
    fn unknown_method_returns_error() {
        let mut s = test_server();
        let req = make_request(99, "bogus/method", None);
        let resp = s.handle_json(&req);
        assert!(resp.contains("-32601"));
        assert!(resp.contains("bogus/method"));
    }

    #[test]
    fn unknown_tool_returns_error() {
        let mut s = test_server();
        let result = s.dispatch_tool("nonexistent_tool", None);
        assert!(result.is_error);
        assert!(result.content[0].text.contains("Unknown tool"));
    }

    #[test]
    fn dispatch_kg_add_and_query() {
        let mut s = test_server();
        // Add KG edge
        let add_result = s.dispatch_tool("palace_kg_add", Some(&Json::Object(vec![
            ("from".to_string(), Json::Str("A".to_string())),
            ("to".to_string(), Json::Str("B".to_string())),
            ("relation".to_string(), Json::Str("causes".to_string())),
        ])));
        assert!(!add_result.is_error);

        // Query it
        let query_result = s.dispatch_tool("palace_kg_query", Some(&Json::Object(vec![
            ("from".to_string(), Json::Str("A".to_string())),
        ])));
        assert!(!query_result.is_error);
        let text = &query_result.content[0].text;
        assert!(text.contains("causes"));
        assert!(text.contains("\"B\""));
    }

    #[test]
    fn dispatch_pheromone_cycle() {
        let mut s = test_server();
        // Setup: add wing/room/drawer
        let wing_id = s.palace_mut().add_wing("ph-wing", "desc");
        let room_id = s.palace_mut().add_room(&wing_id, "ph-room", "desc").unwrap();
        let drawer_id = s.palace_mut().add_drawer(&room_id, "item", "pheromone test", &[]).unwrap();

        // Deposit pheromone
        let dep = s.dispatch_tool("palace_deposit_pheromones", Some(&Json::Object(vec![
            ("drawer_id".to_string(), Json::Str(drawer_id.clone())),
            ("value".to_string(), Json::Float(0.8)),
            ("tag".to_string(), Json::Str("test".to_string())),
        ])));
        assert!(!dep.is_error);

        // Hot paths should find it
        let hot = s.dispatch_tool("palace_hot_paths", Some(&Json::Object(vec![
            ("tag".to_string(), Json::Str("test".to_string())),
        ])));
        assert!(!hot.is_error);
        assert!(hot.content[0].text.contains(&drawer_id));

        // Decay
        let decay = s.dispatch_tool("palace_decay_pheromones", None);
        assert!(!decay.is_error);
    }

    #[test]
    fn dispatch_agent_lifecycle() {
        let mut s = test_server();
        // Create agent
        let create = s.dispatch_tool("palace_create_agent", Some(&Json::Object(vec![
            ("id".to_string(), Json::Str("scout-01".to_string())),
            ("name".to_string(), Json::Str("Scout".to_string())),
            ("role".to_string(), Json::Str("scanning".to_string())),
            ("home_room".to_string(), Json::Str("room:1".to_string())),
        ])));
        assert!(!create.is_error);

        // Write diary
        let write = s.dispatch_tool("palace_diary_write", Some(&Json::Object(vec![
            ("agent_id".to_string(), Json::Str("scout-01".to_string())),
            ("text".to_string(), Json::Str("Found interesting paper".to_string())),
        ])));
        assert!(!write.is_error);

        // Read diary
        let read = s.dispatch_tool("palace_diary_read", Some(&Json::Object(vec![
            ("agent_id".to_string(), Json::Str("scout-01".to_string())),
        ])));
        assert!(!read.is_error);
        assert!(read.content[0].text.contains("interesting paper"));
    }

    #[test]
    fn missing_params_returns_error() {
        let mut s = test_server();
        let req = make_request(5, "tools/call", None);
        let resp = s.handle_json(&req);
        assert!(resp.contains("-32602"));
        assert!(resp.contains("Missing params"));
    }

    #[test]
    fn invalid_json_returns_parse_error() {
        let mut s = test_server();
        let resp = s.handle_json("not valid json {{");
        assert!(resp.contains("-32700"));
    }

    #[test]
    fn initialized_notification() {
        let mut s = test_server();
        let req = make_request(1, "initialized", None);
        let resp = s.handle_json(&req);
        let parsed = Json::parse(&resp).unwrap();
        assert!(parsed.get("result").is_some());
    }

    #[test]
    fn all_28_tools_dispatchable() {
        let mut s = test_server();
        let names = s.tool_names();
        assert_eq!(names.len(), 28);
        for name in &names {
            let result = s.dispatch_tool(name, Some(&Json::Object(vec![])));
            // Should return something (either success or a param-missing error)
            assert!(!result.content.is_empty(), "tool {name} returned empty content");
        }
    }
}
