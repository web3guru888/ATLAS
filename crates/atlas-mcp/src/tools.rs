//! MCP tool definitions for the ATLAS memory palace.
//!
//! Defines 28 tools in five categories:
//! 1. **Palace Navigation** (read) — 8 tools
//! 2. **Palace Operations** (write) — 5 tools
//! 3. **Knowledge Graph** — 7 tools
//! 4. **Stigmergy** — 5 tools
//! 5. **Agent Diary** — 3 tools
//!
//! Each tool has a name, description, and JSON Schema for input parameters.
//! Uses [`atlas_json::Json`] for schema representation.

use atlas_json::Json;

/// Metadata descriptor for a single MCP tool.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Tool name (e.g. `"palace_search"`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema describing the tool's input parameters.
    pub parameters: Json,
}

/// Build a JSON Schema property entry.
fn prop(ptype: &str) -> Json {
    Json::Object(vec![
        ("type".to_string(), Json::Str(ptype.to_string())),
    ])
}

/// Build a JSON Schema object with given properties and required fields.
fn schema(properties: Vec<(&str, &str)>, required: Vec<&str>) -> Json {
    let props = Json::Object(
        properties.iter()
            .map(|(name, ptype)| (name.to_string(), prop(ptype)))
            .collect(),
    );
    let req = Json::Array(required.iter().map(|r| Json::Str(r.to_string())).collect());
    Json::Object(vec![
        ("type".to_string(), Json::Str("object".to_string())),
        ("properties".to_string(), props),
        ("required".to_string(), req),
    ])
}

/// Build a tool definition.
fn tool(name: &str, desc: &str, params: Json) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: desc.to_string(),
        parameters: params,
    }
}

/// Returns metadata for all 28 ATLAS MCP tools.
pub fn tool_catalog() -> Vec<ToolDefinition> {
    vec![
        // ── 1. Palace Navigation (read) — 8 tools ─────────────────────

        tool("palace_status",
             "Return high-level palace status (wings, rooms, drawer counts).",
             schema(vec![], vec![])),

        tool("palace_list_wings",
             "List every wing in the palace.",
             schema(vec![], vec![])),

        tool("palace_list_rooms",
             "List rooms within a specific wing.",
             schema(vec![("wing_id", "string")], vec!["wing_id"])),

        tool("palace_get_taxonomy",
             "Retrieve the full palace taxonomy (wings → rooms → drawers).",
             schema(vec![], vec![])),

        tool("palace_search",
             "Semantic search across all drawers.",
             schema(
                 vec![("query", "string"), ("top_k", "integer")],
                 vec!["query"],
             )),

        tool("palace_navigate",
             "Find pheromone-guided path between two drawers.",
             schema(
                 vec![("from_id", "string"), ("to_id", "string"), ("max_steps", "integer")],
                 vec!["from_id", "to_id"],
             )),

        tool("palace_find_similar",
             "Find drawers similar to a given drawer.",
             schema(
                 vec![("drawer_id", "string"), ("top_k", "integer")],
                 vec!["drawer_id"],
             )),

        tool("palace_graph_stats",
             "Return aggregate graph statistics (similarity edges, tunnels).",
             schema(vec![], vec![])),

        // ── 2. Palace Operations (write) — 5 tools ────────────────────

        tool("palace_add_wing",
             "Create a new wing.",
             schema(
                 vec![("name", "string"), ("description", "string")],
                 vec!["name", "description"],
             )),

        tool("palace_add_room",
             "Create a new room in an existing wing.",
             schema(
                 vec![("wing_id", "string"), ("name", "string"), ("description", "string")],
                 vec!["wing_id", "name"],
             )),

        tool("palace_add_drawer",
             "Store a new drawer (memory item) in a specific room.",
             schema(
                 vec![("room_id", "string"), ("title", "string"), ("content", "string")],
                 vec!["room_id", "title", "content"],
             )),

        tool("palace_add_drawer_if_unique",
             "Store a drawer only if no near-duplicate exists in the room.",
             schema(
                 vec![("room_id", "string"), ("title", "string"), ("content", "string"),
                      ("threshold", "number")],
                 vec!["room_id", "title", "content"],
             )),

        tool("palace_check_duplicate",
             "Check whether content is a near-duplicate of an existing drawer.",
             schema(
                 vec![("room_id", "string"), ("content", "string"), ("threshold", "number")],
                 vec!["room_id", "content"],
             )),

        // ── 3. Knowledge Graph — 7 tools ──────────────────────────────

        tool("palace_kg_add",
             "Add a directed KG edge (from, to, relation).",
             schema(
                 vec![("from", "string"), ("to", "string"), ("relation", "string"),
                      ("confidence", "number")],
                 vec!["from", "to", "relation"],
             )),

        tool("palace_kg_add_temporal",
             "Add a temporal KG edge with timestamp.",
             schema(
                 vec![("from", "string"), ("to", "string"), ("relation", "string"),
                      ("confidence", "number"), ("timestamp", "integer")],
                 vec!["from", "to", "relation"],
             )),

        tool("palace_kg_query",
             "Query KG edges from a source node.",
             schema(
                 vec![("from", "string")],
                 vec!["from"],
             )),

        tool("palace_kg_contradictions",
             "Detect contradictory triples exceeding a confidence threshold.",
             schema(
                 vec![("threshold", "number")],
                 vec![],
             )),

        tool("palace_kg_invalidate",
             "Invalidate (remove) all edges from/to a drawer id.",
             schema(
                 vec![("drawer_id", "string")],
                 vec!["drawer_id"],
             )),

        tool("palace_build_similarity_graph",
             "Build similarity edges between drawers with cosine > threshold.",
             schema(
                 vec![("threshold", "number")],
                 vec![],
             )),

        tool("palace_build_tunnels",
             "Build cross-wing shortcut edges between high-pheromone drawers.",
             schema(
                 vec![("min_pheromone", "number")],
                 vec![],
             )),

        // ── 4. Stigmergy — 5 tools ────────────────────────────────────

        tool("palace_deposit_pheromones",
             "Deposit pheromone on a drawer.",
             schema(
                 vec![("drawer_id", "string"), ("value", "number"),
                      ("decay", "number"), ("tag", "string")],
                 vec!["drawer_id", "value", "tag"],
             )),

        tool("palace_decay_pheromones",
             "Trigger an immediate pheromone decay pass.",
             schema(vec![], vec![])),

        tool("palace_hot_paths",
             "Return the hottest (most-pheromone) drawers.",
             schema(
                 vec![("tag", "string"), ("top_k", "integer")],
                 vec![],
             )),

        tool("palace_cold_spots",
             "Return cold spots (least-visited drawers).",
             schema(
                 vec![("threshold", "number"), ("top_k", "integer")],
                 vec![],
             )),

        tool("palace_pheromone_status",
             "Query overall pheromone statistics for a drawer.",
             schema(
                 vec![("drawer_id", "string")],
                 vec![],
             )),

        // ── 5. Agent Diary — 3 tools ──────────────────────────────────

        tool("palace_create_agent",
             "Register a new agent in the palace.",
             schema(
                 vec![("id", "string"), ("name", "string"),
                      ("role", "string"), ("home_room", "string")],
                 vec!["id", "name", "role", "home_room"],
             )),

        tool("palace_diary_write",
             "Append an entry to an agent's diary.",
             schema(
                 vec![("agent_id", "string"), ("text", "string")],
                 vec!["agent_id", "text"],
             )),

        tool("palace_diary_read",
             "Read entries from an agent's diary.",
             schema(
                 vec![("agent_id", "string"), ("n", "integer")],
                 vec!["agent_id"],
             )),
    ]
}

/// Total number of tools in the catalog.
pub const TOOL_COUNT: usize = 28;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_28_tools() {
        let catalog = tool_catalog();
        assert_eq!(catalog.len(), TOOL_COUNT);
    }

    #[test]
    fn tool_names_unique() {
        let catalog = tool_catalog();
        let mut names: Vec<&str> = catalog.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        for w in names.windows(2) {
            assert_ne!(w[0], w[1], "duplicate tool name: {}", w[0]);
        }
    }

    #[test]
    fn tool_definitions_have_names_and_schemas() {
        let catalog = tool_catalog();
        for t in &catalog {
            assert!(!t.name.is_empty(), "tool has empty name");
            assert!(!t.description.is_empty(), "tool {} has empty description", t.name);
            assert_eq!(t.parameters.get("type").unwrap().as_str(), Some("object"),
                "tool {} parameters is not an object schema", t.name);
        }
    }

    #[test]
    fn search_tool_has_required_query() {
        let catalog = tool_catalog();
        let search = catalog.iter().find(|t| t.name == "palace_search").unwrap();
        let required = search.parameters.get("required").unwrap().as_array().unwrap();
        assert!(required.iter().any(|r| r.as_str() == Some("query")));
    }

    #[test]
    fn all_tools_start_with_palace_prefix() {
        let catalog = tool_catalog();
        for t in &catalog {
            assert!(t.name.starts_with("palace_"), "tool {} missing palace_ prefix", t.name);
        }
    }
}
