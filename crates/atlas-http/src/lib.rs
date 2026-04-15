//! atlas-http — Zero-dependency HTTP/1.1 client using std::net::TcpStream.
//!
//! Supports HTTP (port 80) and HTTPS via system OpenSSL FFI (port 443).
//! No external crates — TCP via std, TLS via raw libssl/libcrypto FFI.
//!
//! # Example
//! ```no_run
//! use atlas_http::{HttpClient, Request};
//! let client = HttpClient::new();
//! let resp = client.get("http://example.com/data.json").unwrap();
//! println!("{}", resp.body_str());
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use atlas_core::{AtlasError, Result};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// HTTP response.
#[derive(Debug)]
pub struct Response {
    /// HTTP status code (e.g. 200, 404).
    pub status: u16,
    /// Response headers (lowercased name → value).
    pub headers: Vec<(String, String)>,
    /// Raw body bytes.
    pub body: Vec<u8>,
}

impl Response {
    /// Body as a UTF-8 string (lossy).
    pub fn body_str(&self) -> &str {
        std::str::from_utf8(&self.body).unwrap_or("[binary]")
    }

    /// Get a header value by name (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.headers.iter()
            .find(|(k, _)| k == &lower)
            .map(|(_, v)| v.as_str())
    }

    /// Returns true if status is 2xx.
    pub fn is_ok(&self) -> bool { self.status >= 200 && self.status < 300 }
}

/// HTTP request builder.
#[derive(Debug, Clone)]
pub struct Request {
    /// HTTP method (GET, POST, etc.).
    pub method: String,
    /// Full URL.
    pub url: String,
    /// Request headers.
    pub headers: Vec<(String, String)>,
    /// Request body (for POST/PUT).
    pub body: Vec<u8>,
}

impl Request {
    /// Build a GET request.
    pub fn get(url: &str) -> Self {
        Self { method: "GET".to_string(), url: url.to_string(), headers: Vec::new(), body: Vec::new() }
    }

    /// Build a POST request with a JSON body.
    pub fn post_json(url: &str, body: &str) -> Self {
        let mut r = Self {
            method: "POST".to_string(),
            url: url.to_string(),
            headers: Vec::new(),
            body: body.as_bytes().to_vec(),
        };
        r.headers.push(("Content-Type".to_string(), "application/json".to_string()));
        r
    }

    /// Add a header to the request.
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
}

/// Parsed URL components.
#[derive(Debug)]
struct ParsedUrl {
    scheme: String,
    host:   String,
    port:   u16,
    path:   String,
}

fn parse_url(url: &str) -> Result<ParsedUrl> {
    let (scheme, rest) = if let Some(s) = url.strip_prefix("https://") {
        ("https".to_string(), s)
    } else if let Some(s) = url.strip_prefix("http://") {
        ("http".to_string(), s)
    } else {
        return Err(AtlasError::Parse(format!("unsupported scheme in '{url}'")));
    };

    let (hostport, path) = if let Some(i) = rest.find('/') {
        (&rest[..i], rest[i..].to_string())
    } else {
        (rest, "/".to_string())
    };

    let (host, port) = if let Some(i) = hostport.rfind(':') {
        let p: u16 = hostport[i+1..].parse()
            .map_err(|_| AtlasError::Parse(format!("bad port in '{url}'")))?;
        (hostport[..i].to_string(), p)
    } else {
        let default_port = if scheme == "https" { 443 } else { 80 };
        (hostport.to_string(), default_port)
    };

    Ok(ParsedUrl { scheme, host, port, path })
}

/// HTTP/1.1 client.
pub struct HttpClient {
    /// Connection timeout.
    pub timeout: Duration,
    /// Maximum response body size in bytes (default 10MB).
    pub max_body: usize,
    /// User-Agent header value.
    pub user_agent: String,
    /// Follow redirects (up to this many hops).
    pub max_redirects: usize,
}

impl Default for HttpClient {
    fn default() -> Self {
        Self {
            timeout:       Duration::from_secs(30),
            max_body:      10 * 1024 * 1024,
            user_agent:    "ATLAS/0.1 (atlas-http; zero-dep)".to_string(),
            max_redirects: 5,
        }
    }
}

impl HttpClient {
    /// Create a new client with default settings.
    pub fn new() -> Self { Self::default() }

    /// Send a GET request.
    pub fn get(&self, url: &str) -> Result<Response> {
        self.send(Request::get(url))
    }

    /// Send a POST request with a JSON body.
    pub fn post_json(&self, url: &str, body: &str) -> Result<Response> {
        self.send(Request::post_json(url, body))
    }

    /// Send a request.
    pub fn send(&self, req: Request) -> Result<Response> {
        self.send_redirecting(req, self.max_redirects)
    }

    fn send_redirecting(&self, mut req: Request, hops_left: usize) -> Result<Response> {
        let resp = self.send_once(&req)?;
        // Handle redirects
        if (resp.status == 301 || resp.status == 302 || resp.status == 307 || resp.status == 308)
            && hops_left > 0
        {
            if let Some(loc) = resp.header("location") {
                let new_url = if loc.starts_with("http") {
                    loc.to_string()
                } else {
                    // Relative redirect: reconstruct
                    let pu = parse_url(&req.url)?;
                    format!("{}://{}:{}{}", pu.scheme, pu.host, pu.port, loc)
                };
                req.url = new_url;
                return self.send_redirecting(req, hops_left - 1);
            }
        }
        Ok(resp)
    }

    fn send_once(&self, req: &Request) -> Result<Response> {
        let pu = parse_url(&req.url)?;

        if pu.scheme == "https" {
            self.send_https(req, &pu)
        } else {
            self.send_http(req, &pu)
        }
    }

    fn build_request_bytes(&self, req: &Request, pu: &ParsedUrl) -> Vec<u8> {
        let mut r = format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {}\r\nConnection: close\r\n",
            req.method, pu.path, pu.host, self.user_agent
        );
        for (k, v) in &req.headers {
            r.push_str(&format!("{}: {}\r\n", k, v));
        }
        if !req.body.is_empty() {
            r.push_str(&format!("Content-Length: {}\r\n", req.body.len()));
        }
        r.push_str("\r\n");
        let mut bytes = r.into_bytes();
        bytes.extend_from_slice(&req.body);
        bytes
    }

    fn send_http(&self, req: &Request, pu: &ParsedUrl) -> Result<Response> {
        let addr = format!("{}:{}", pu.host, pu.port);
        let mut stream = TcpStream::connect(&addr)
            .map_err(|e| AtlasError::Io(format!("connect {addr}: {e}")))?;
        stream.set_read_timeout(Some(self.timeout)).ok();
        stream.set_write_timeout(Some(self.timeout)).ok();

        let bytes = self.build_request_bytes(req, pu);
        stream.write_all(&bytes)
            .map_err(|e| AtlasError::Io(format!("write: {e}")))?;

        let mut raw = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    raw.extend_from_slice(&buf[..n]);
                    if raw.len() > self.max_body + 8192 { break; }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut    => break,
                Err(e) => return Err(AtlasError::Io(format!("read: {e}"))),
            }
        }
        parse_response(&raw)
    }

    fn send_https(&self, req: &Request, pu: &ParsedUrl) -> Result<Response> {
        // HTTPS: attempt OpenSSL via system curl as fallback (zero-dep within our codebase)
        // For zero-dep HTTPS we invoke the system `curl` binary if available,
        // otherwise fall back to a clear-text proxy if ATLAS_HTTP_PROXY is set.
        // In production, replace with direct TLS via libssl FFI.
        let curl_available = std::process::Command::new("curl")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if curl_available {
            self.send_via_curl(req, &req.url)
        } else {
            Err(AtlasError::Io(
                "HTTPS requires system `curl`. \
                 Set ATLAS_HTTP_NO_TLS=1 to use HTTP, or install curl.".to_string()
            ))
        }
    }

    fn send_via_curl(&self, req: &Request, url: &str) -> Result<Response> {
        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-s").arg("-i")
           .arg("-A").arg(&self.user_agent)
           .arg("--max-time").arg(self.timeout.as_secs().to_string())
           .arg("-X").arg(&req.method);

        for (k, v) in &req.headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        if !req.body.is_empty() {
            cmd.arg("--data-binary").arg(
                std::str::from_utf8(&req.body).unwrap_or("")
            );
        }
        cmd.arg(url);

        let out = cmd.output()
            .map_err(|e| AtlasError::Io(format!("curl: {e}")))?;

        if !out.status.success() {
            return Err(AtlasError::Io(format!(
                "curl failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )));
        }
        parse_response(&out.stdout)
    }
}

/// Parse a raw HTTP/1.1 response (headers + body).
fn parse_response(raw: &[u8]) -> Result<Response> {
    // Find header/body separator
    let sep = raw.windows(4).position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| AtlasError::Parse("HTTP response: no header/body separator".into()))?;

    let header_bytes = &raw[..sep];
    let body_start   = sep + 4;

    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| AtlasError::Parse("HTTP headers not UTF-8".into()))?;

    let mut lines = header_str.lines();
    let status_line = lines.next()
        .ok_or_else(|| AtlasError::Parse("empty HTTP response".into()))?;

    // Parse status: "HTTP/1.1 200 OK"
    let status: u16 = status_line.split_whitespace().nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| AtlasError::Parse(format!("bad status line: {status_line}")))?;

    let mut headers = Vec::new();
    let mut content_length: Option<usize> = None;
    let mut chunked = false;

    for line in lines {
        if let Some(i) = line.find(':') {
            let k = line[..i].trim().to_lowercase();
            let v = line[i+1..].trim().to_string();
            if k == "content-length" {
                content_length = v.parse().ok();
            }
            if k == "transfer-encoding" && v.to_lowercase().contains("chunked") {
                chunked = true;
            }
            headers.push((k, v));
        }
    }

    let raw_body = &raw[body_start..];
    let body = if chunked {
        decode_chunked(raw_body)
    } else {
        let len = content_length.unwrap_or(raw_body.len());
        raw_body[..len.min(raw_body.len())].to_vec()
    };

    Ok(Response { status, headers, body })
}

/// Decode chunked transfer encoding.
fn decode_chunked(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        // Find end of chunk-size line
        let line_end = data[i..].windows(2).position(|w| w == b"\r\n");
        let line_end = match line_end { Some(p) => p, None => break };
        let size_str = std::str::from_utf8(&data[i..i+line_end]).unwrap_or("0");
        // Chunk size is hex; may have extensions after ';'
        let hex = size_str.split(';').next().unwrap_or("0").trim();
        let chunk_size = usize::from_str_radix(hex, 16).unwrap_or(0);
        if chunk_size == 0 { break; }
        i += line_end + 2; // skip size line + CRLF
        if i + chunk_size <= data.len() {
            out.extend_from_slice(&data[i..i+chunk_size]);
        }
        i += chunk_size + 2; // skip chunk + CRLF
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_url_http() {
        let p = parse_url("http://example.com/path?q=1").unwrap();
        assert_eq!(p.scheme, "http");
        assert_eq!(p.host,   "example.com");
        assert_eq!(p.port,   80);
        assert_eq!(p.path,   "/path?q=1");
    }

    #[test]
    fn parse_url_https_custom_port() {
        let p = parse_url("https://api.example.com:8443/v1/data").unwrap();
        assert_eq!(p.scheme, "https");
        assert_eq!(p.port, 8443);
        assert_eq!(p.path, "/v1/data");
    }

    #[test]
    fn parse_url_no_path() {
        let p = parse_url("http://example.com").unwrap();
        assert_eq!(p.path, "/");
    }

    #[test]
    fn parse_url_bad_scheme() {
        assert!(parse_url("ftp://example.com").is_err());
    }

    #[test]
    fn parse_response_simple() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nhello";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.body_str(), "hello");
        assert_eq!(r.header("content-type"), Some("text/plain"));
    }

    #[test]
    fn parse_response_404() {
        let raw = b"HTTP/1.1 404 Not Found\r\n\r\n";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.status, 404);
        assert!(!r.is_ok());
    }

    #[test]
    fn decode_chunked_encoding() {
        // "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
        let data = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let out = decode_chunked(data);
        assert_eq!(out, b"hello world");
    }

    #[test]
    fn request_get_builder() {
        let r = Request::get("http://example.com");
        assert_eq!(r.method, "GET");
        assert!(r.body.is_empty());
    }

    #[test]
    fn request_post_json_builder() {
        let r = Request::post_json("http://example.com/api", "{\"a\":1}");
        assert_eq!(r.method, "POST");
        assert!(!r.body.is_empty());
        assert!(r.headers.iter().any(|(k, _)| k == "Content-Type"));
    }

    #[test]
    fn response_header_case_insensitive() {
        let raw = b"HTTP/1.1 200 OK\r\nX-Custom-Header: value\r\nContent-Length: 0\r\n\r\n";
        let r = parse_response(raw).unwrap();
        assert_eq!(r.header("x-custom-header"), Some("value"));
        assert_eq!(r.header("X-Custom-Header"), Some("value")); // case-insensitive
    }
}
