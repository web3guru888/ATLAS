//! atlas-api — OpenAI-compatible HTTP/1.1 inference server.
//!
//! Exposes a subset of the OpenAI REST API, allowing any OpenAI-compatible
//! client to interact with an ATLAS-trained model. Zero external crate
//! dependencies: HTTP parsing via std::net::TcpListener, JSON via atlas-json.
//!
//! # Endpoints
//!
//! | Method | Path                     | Description                  |
//! |--------|--------------------------|------------------------------|
//! | GET    | /health                  | Health check                 |
//! | GET    | /v1/models               | List available models        |
//! | POST   | /v1/chat/completions     | Chat completions (streaming) |
//! | POST   | /v1/completions          | Text completions             |
//!
//! # Example
//!
//! ```no_run
//! use atlas_api::ApiServer;
//! use atlas_api::types::ServerConfig;
//!
//! let cfg = ServerConfig {
//!     host: "0.0.0.0".to_string(),
//!     port: 8080,
//!     model_id: "atlas-smollm2-135m".to_string(),
//!     weights_dir: None,
//!     max_tokens: 2048,
//!     workers: 4,
//! };
//! // ApiServer::new(cfg).serve().unwrap();  // blocks
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod handler;
pub mod server;
pub mod types;

pub use server::ApiServer;
