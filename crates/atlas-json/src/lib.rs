//! atlas-json — Zero-dependency recursive-descent JSON parser.
//!
//! Supports all JSON types. Used by atlas-tokenize (vocab loading)
//! and atlas-model (safetensors header, config files).
//!
//! # Example
//! ```
//! use atlas_json::Json;
//! let v = Json::parse(r#"{"hello": 42}"#).unwrap();
//! assert_eq!(v.get("hello").and_then(|x| x.as_i64()), Some(42));
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use std::fmt;

/// A parsed JSON value.
#[derive(Clone, PartialEq)]
pub enum Json {
    /// JSON `null`
    Null,
    /// JSON boolean
    Bool(bool),
    /// JSON integer (i64)
    Int(i64),
    /// JSON float (f64) — only when the literal contains `.` or `e/E`
    Float(f64),
    /// JSON string
    Str(String),
    /// JSON array
    Array(Vec<Json>),
    /// JSON object (preserves insertion order)
    Object(Vec<(String, Json)>),
}

/// Parse error with position information.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// Human-readable message.
    pub message: String,
    /// Byte offset in the input string.
    pub position: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JSON parse error at position {}: {}", self.position, self.message)
    }
}

impl std::error::Error for ParseError {}

// ── Parser internals ──────────────────────────────────────────────────────────

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self { src: s.as_bytes(), pos: 0 }
    }

    fn err(&self, msg: &str) -> ParseError {
        ParseError { message: msg.to_string(), position: self.pos }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() { self.pos += 1; }
        b
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect_bytes(&mut self, s: &[u8]) -> Result<(), ParseError> {
        for &b in s {
            match self.advance() {
                Some(c) if c == b => {}
                _ => return Err(self.err(&format!("expected '{}'", std::str::from_utf8(s).unwrap_or("?")))),
            }
        }
        Ok(())
    }

    fn parse_value(&mut self) -> Result<Json, ParseError> {
        self.skip_ws();
        match self.peek() {
            None         => Err(self.err("unexpected end of input")),
            Some(b'n')   => { self.expect_bytes(b"null")?;  Ok(Json::Null) }
            Some(b't')   => { self.expect_bytes(b"true")?;  Ok(Json::Bool(true)) }
            Some(b'f')   => { self.expect_bytes(b"false")?; Ok(Json::Bool(false)) }
            Some(b'"')   => Ok(Json::Str(self.parse_string()?)),
            Some(b'[')   => self.parse_array(),
            Some(b'{')   => self.parse_object(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(c)      => Err(self.err(&format!("unexpected byte {c:#04x}"))),
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        // consume opening "
        self.advance();
        let mut out = Vec::new();
        loop {
            match self.advance() {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => break,
                Some(b'\\') => match self.advance() {
                    Some(b'"')  => out.push(b'"'),
                    Some(b'\\') => out.push(b'\\'),
                    Some(b'/')  => out.push(b'/'),
                    Some(b'n')  => out.push(b'\n'),
                    Some(b'r')  => out.push(b'\r'),
                    Some(b't')  => out.push(b'\t'),
                    Some(b'b')  => out.push(8),
                    Some(b'f')  => out.push(12),
                    Some(b'u')  => {
                        let cp = self.parse_unicode_escape()?;
                        self.encode_utf8(cp, &mut out);
                    }
                    _ => return Err(self.err("invalid escape sequence")),
                },
                Some(b) => out.push(b),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("invalid UTF-8 in string"))
    }

    fn parse_unicode_escape(&mut self) -> Result<u32, ParseError> {
        let mut code: u32 = 0;
        for _ in 0..4 {
            let b = self.advance().ok_or_else(|| self.err("unexpected end in \\uXXXX"))?;
            let nibble = match b {
                b'0'..=b'9' => (b - b'0') as u32,
                b'a'..=b'f' => (b - b'a') as u32 + 10,
                b'A'..=b'F' => (b - b'A') as u32 + 10,
                _ => return Err(self.err("invalid hex digit in \\uXXXX")),
            };
            code = (code << 4) | nibble;
        }
        // Handle surrogate pairs (high surrogate: D800-DBFF)
        if (0xD800..=0xDBFF).contains(&code) {
            // expect \uXXXX
            self.expect_bytes(b"\\u")?;
            let mut low: u32 = 0;
            for _ in 0..4 {
                let b = self.advance().ok_or_else(|| self.err("unexpected end in surrogate pair"))?;
                let nibble = match b {
                    b'0'..=b'9' => (b - b'0') as u32,
                    b'a'..=b'f' => (b - b'a') as u32 + 10,
                    b'A'..=b'F' => (b - b'A') as u32 + 10,
                    _ => return Err(self.err("invalid hex in surrogate pair")),
                };
                low = (low << 4) | nibble;
            }
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        }
        Ok(code)
    }

    fn encode_utf8(&self, cp: u32, out: &mut Vec<u8>) {
        if cp < 0x80 {
            out.push(cp as u8);
        } else if cp < 0x800 {
            out.push(0xC0 | (cp >> 6) as u8);
            out.push(0x80 | (cp & 0x3F) as u8);
        } else if cp < 0x10000 {
            out.push(0xE0 | (cp >> 12) as u8);
            out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
            out.push(0x80 | (cp & 0x3F) as u8);
        } else {
            out.push(0xF0 | (cp >> 18) as u8);
            out.push(0x80 | ((cp >> 12) & 0x3F) as u8);
            out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
            out.push(0x80 | (cp & 0x3F) as u8);
        }
    }

    fn parse_number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        // optional minus
        if self.peek() == Some(b'-') { self.pos += 1; }
        // integer part
        if self.peek() == Some(b'0') {
            self.pos += 1;
        } else {
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
        }
        let mut is_float = false;
        // fractional
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
        }
        // exponent
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.pos += 1; }
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
        }
        let s = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| self.err("non-UTF8 in number"))?;
        if is_float {
            s.parse::<f64>()
             .map(Json::Float)
             .map_err(|_| self.err(&format!("invalid float: {s}")))
        } else {
            // Try i64 first, fall back to f64 for large numbers
            if let Ok(i) = s.parse::<i64>() {
                Ok(Json::Int(i))
            } else {
                s.parse::<f64>()
                 .map(Json::Float)
                 .map_err(|_| self.err(&format!("invalid number: {s}")))
            }
        }
    }

    fn parse_array(&mut self) -> Result<Json, ParseError> {
        self.advance(); // consume '['
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') { self.advance(); return Ok(Json::Array(arr)); }
        loop {
            arr.push(self.parse_value()?);
            self.skip_ws();
            match self.advance() {
                Some(b']') => break,
                Some(b',') => {}
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
        Ok(Json::Array(arr))
    }

    fn parse_object(&mut self) -> Result<Json, ParseError> {
        self.advance(); // consume '{'
        let mut obj = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') { self.advance(); return Ok(Json::Object(obj)); }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected '\"' for object key"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect_bytes(b":")?;
            let val = self.parse_value()?;
            obj.push((key, val));
            self.skip_ws();
            match self.advance() {
                Some(b'}') => break,
                Some(b',') => {}
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
        Ok(Json::Object(obj))
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

impl Json {
    /// Parse a JSON string. Returns the root value.
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let mut p = Parser::new(input);
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.src.len() {
            return Err(p.err("trailing data after JSON value"));
        }
        Ok(v)
    }

    /// Look up a key in a JSON object. Returns `None` if not an object or key absent.
    pub fn get(&self, key: &str) -> Option<&Json> {
        if let Json::Object(pairs) = self {
            pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }

    /// Index into a JSON array.
    pub fn index(&self, i: usize) -> Option<&Json> {
        if let Json::Array(arr) = self { arr.get(i) } else { None }
    }

    /// Returns `Some(&str)` if this is a `Json::Str`.
    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s.as_str()) } else { None }
    }

    /// Returns `Some(i64)` for both `Int` and `Float` variants.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(n) => Some(*n),
            Json::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Returns `Some(usize)` when the value is a non-negative integer.
    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64().and_then(|n| if n >= 0 { Some(n as usize) } else { None })
    }

    /// Returns `Some(f64)` for both `Int` and `Float` variants.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(n) => Some(*n as f64),
            Json::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns `Some(bool)` if this is a `Json::Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self { Some(*b) } else { None }
    }

    /// Returns `Some(&[Json])` if this is a `Json::Array`.
    pub fn as_array(&self) -> Option<&[Json]> {
        if let Json::Array(a) = self { Some(a.as_slice()) } else { None }
    }

    /// Returns `Some(&[(String,Json)])` if this is a `Json::Object`.
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        if let Json::Object(o) = self { Some(o.as_slice()) } else { None }
    }

    /// Returns `true` if this is `Json::Null`.
    pub fn is_null(&self) -> bool { matches!(self, Json::Null) }

    /// Serialize back to a compact JSON string.
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        self.write_json(&mut s);
        s
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Json::Null        => out.push_str("null"),
            Json::Bool(b)     => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(n)      => out.push_str(&n.to_string()),
            Json::Float(f)    => out.push_str(&format!("{f}")),
            Json::Str(s)      => { out.push('"'); Self::escape_str(s, out); out.push('"'); }
            Json::Array(arr)  => {
                out.push('[');
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    v.write_json(out);
                }
                out.push(']');
            }
            Json::Object(obj) => {
                out.push('{');
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    out.push('"'); Self::escape_str(k, out); out.push('"');
                    out.push(':');
                    v.write_json(out);
                }
                out.push('}');
            }
        }
    }

    fn escape_str(s: &str, out: &mut String) {
        for c in s.chars() {
            match c {
                '"'  => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
    }
}

impl fmt::Debug for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json())
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_primitives() {
        assert_eq!(Json::parse("null").unwrap(), Json::Null);
        assert_eq!(Json::parse("true").unwrap(), Json::Bool(true));
        assert_eq!(Json::parse("false").unwrap(), Json::Bool(false));
        assert_eq!(Json::parse("42").unwrap(), Json::Int(42));
        assert_eq!(Json::parse("-7").unwrap(), Json::Int(-7));
        assert_eq!(Json::parse("3.14").unwrap().as_f64().unwrap().abs() < 0.001 + 3.14, true);
        assert_eq!(Json::parse("1e3").unwrap().as_f64().unwrap() as i64, 1000);
    }

    #[test]
    fn parse_string() {
        assert_eq!(Json::parse(r#""hello""#).unwrap().as_str(), Some("hello"));
        assert_eq!(Json::parse(r#""a\nb""#).unwrap().as_str(), Some("a\nb"));
        assert_eq!(Json::parse(r#""a\\b""#).unwrap().as_str(), Some("a\\b"));
        assert_eq!(Json::parse(r#""\u0041""#).unwrap().as_str(), Some("A"));
    }

    #[test]
    fn parse_array() {
        let v = Json::parse("[1,2,3]").unwrap();
        assert_eq!(v.as_array().unwrap().len(), 3);
        assert_eq!(v.index(1).unwrap().as_i64(), Some(2));
    }

    #[test]
    fn parse_object() {
        let v = Json::parse(r#"{"x":1,"y":"hi"}"#).unwrap();
        assert_eq!(v.get("x").unwrap().as_i64(), Some(1));
        assert_eq!(v.get("y").unwrap().as_str(), Some("hi"));
        assert!(v.get("z").is_none());
    }

    #[test]
    fn parse_nested() {
        let v = Json::parse(r#"{"a":[1,{"b":null}]}"#).unwrap();
        let inner = v.get("a").unwrap().index(1).unwrap();
        assert!(inner.get("b").unwrap().is_null());
    }

    #[test]
    fn empty_structures() {
        assert_eq!(Json::parse("[]").unwrap().as_array().unwrap().len(), 0);
        assert_eq!(Json::parse("{}").unwrap().as_object().unwrap().len(), 0);
    }

    #[test]
    fn roundtrip() {
        let orig = r#"{"a":1,"b":[2,3],"c":true,"d":null}"#;
        let v = Json::parse(orig).unwrap();
        assert_eq!(v.to_json(), orig);
    }

    #[test]
    fn unicode_surrogate_pair() {
        // 𝄞 = U+1D11E (musical symbol G-clef) encoded as \uD834\uDD1E
        let v = Json::parse(r#""\uD834\uDD1E""#).unwrap();
        assert_eq!(v.as_str(), Some("𝄞"));
    }

    #[test]
    fn whitespace_tolerance() {
        let v = Json::parse("  { \"k\" : [ 1 , 2 ]  }  ").unwrap();
        assert_eq!(v.get("k").unwrap().as_array().unwrap().len(), 2);
    }

    #[test]
    fn trailing_data_error() {
        assert!(Json::parse("42 extra").is_err());
    }

    #[test]
    fn large_number_fallback() {
        // 2^63 overflows i64, should parse as float
        let v = Json::parse("9223372036854775808").unwrap();
        assert!(v.as_f64().is_some());
    }
}
