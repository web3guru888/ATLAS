//! atlas-api — OpenAI-compatible HTTP/1.1 inference server.
//!
//! Exposes a subset of the OpenAI v1 REST API so any OpenAI-compatible client
//! (Python `openai` library, `curl`, LangChain, etc.) can talk to an ATLAS
//! model out of the box.  Zero external crate dependencies: the HTTP server
//! uses `std::net::TcpListener`, JSON is handled by `atlas-json`.
//!
//! # Endpoints
//!
//! | Method | Path                     | Description                    | Auth |
//! |--------|--------------------------|--------------------------------|------|
//! | GET    | `/health`                | Health / liveness check        | no   |
//! | GET    | `/v1/models`             | List available models          | no   |
//! | POST   | `/v1/chat/completions`   | Chat completions (+ streaming) | yes* |
//! | POST   | `/v1/completions`        | Text completions               | yes* |
//! | `*`    | `OPTIONS *`              | CORS preflight (204)           | no   |
//!
//! \* Bearer-key auth is enforced only when a key is configured — via
//! [`types::ServerConfig::api_key`] or the `ATLAS_API_KEY` /
//! `ATLAS_API_KEY_FILE` environment variables. The engine is single-stream:
//! concurrent inference requests beyond `max_inflight` receive an immediate
//! `429` with a `Retry-After` header instead of queueing.
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
//!     ..ServerConfig::default()
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
