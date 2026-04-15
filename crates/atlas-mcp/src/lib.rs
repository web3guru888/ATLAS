//! atlas-mcp — MCP (Model Context Protocol) server for the ATLAS memory palace.
//!
//! Exposes all 28 palace tools via JSON-RPC 2.0 over stdin/stdout.
//! Zero external dependencies — uses only atlas-core, atlas-json, atlas-palace, and std.
//!
//! # Architecture
//!
//! - **`protocol`** — JSON-RPC 2.0 message types (request, response, error)
//! - **`tools`** — 28 MCP tool definitions with JSON Schema parameters
//! - **`server`** — MCP server: routes `initialize`, `tools/list`, `tools/call`
//!
//! # Usage
//!
//! ```no_run
//! use atlas_palace::Palace;
//! use atlas_mcp::McpServer;
//!
//! let palace = Palace::new("my-palace", "/tmp/palace");
//! let mut server = McpServer::new(palace);
//! server.run_stdio().unwrap();
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod protocol;
pub mod server;
pub mod tools;

// Re-export the main entry point.
pub use server::McpServer;
