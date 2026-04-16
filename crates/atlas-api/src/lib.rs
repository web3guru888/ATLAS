//! atlas-api — OpenAI-compatible HTTP/1.1 inference server.
//!
//! Exposes a subset of the OpenAI v1 REST API so any OpenAI-compatible client
//! (Python `openai` library, `curl`, LangChain, etc.) can talk to an ATLAS
//! model out of the box.  Zero external crate dependencies: the HTTP server
//! uses `std::net::TcpListener`, JSON is handled by `atlas-json`.
//!
//! # Endpoints
//!
//! | Method | Path                     | Description                    |
//! |--------|--------------------------|--------------------------------|
//! | GET    | `/health`                | Health / liveness check        |
//! | GET    | `/v1/models`             | List available models          |
//! | POST   | `/v1/chat/completions`   | Chat completions (+ streaming) |
//! | POST   | `/v1/completions`        | Text completions               |
//! | `*`    | `OPTIONS *`              | CORS preflight (204)           |
//!
//! # Quick start
//!
//! ```no_run
//! use atlas_api::{ApiServer, types::ServerConfig};
//!
//! let cfg = ServerConfig {
//!     host:        "0.0.0.0".to_string(),
//!     port:        8080,
//!     model_id:    "smollm2-135m".to_string(),
//!     weights_dir: Some("/models/smollm2-135m".to_string()),
//!     max_tokens:  2048,
//!     workers:     4,
//! };
//! // Blocks until Ctrl-C:
//! // ApiServer::new(cfg).serve().unwrap();
//! ```
//!
//! # Without weights (echo / test mode)
//!
//! ```no_run
//! use atlas_api::{ApiServer, types::ServerConfig};
//! let cfg = ServerConfig { weights_dir: None, ..ServerConfig::default() };
//! // ApiServer::new(cfg).serve().unwrap();
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod handler;
pub mod server;
pub mod types;

pub use server::ApiServer;
