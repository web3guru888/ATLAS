//! atlas-tokenize — BPE tokenizer (GPT-2/GPT-NeoX byte-level BPE).
//!
//! Zero external crate dependencies. Loads from HuggingFace `tokenizer.json`
//! format using atlas-json.
//!
//! Supports OLMo, LLaMA 3, GPT-2, and any byte-level BPE tokenizer whose
//! vocab/merges are stored in the standard HuggingFace JSON format.
//!
//! # Example
//! ```no_run
//! use atlas_tokenize::Tokenizer;
//! let tok = Tokenizer::from_file("tokenizer.json").unwrap();
//! let ids = tok.encode("Hello, world!");
//! let text = tok.decode(&ids);
//! assert_eq!(text, "Hello, world!");
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use atlas_core::AtlasError;
use atlas_json::Json;
use std::collections::HashMap;

/// Pre-tokenizer type detected from tokenizer.json configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum PreTokenizerType {
    /// GPT-4/OLMo-3/LLaMA-3 regex pattern (handles contractions, numbers, punctuation)
    Gpt4Regex,
    /// Simple byte-level (naive space splitting — legacy fallback)
    Simple,
}

/// A byte-level BPE tokenizer compatible with GPT-2/OLMo/LLaMA-3.
pub struct Tokenizer {
    /// token id → token string (byte repr)
    vocab: Vec<String>,
    /// token string → token id
    vocab_map: HashMap<String, u32>,
    /// BPE merges in order: (left, right) → merged
    merges: Vec<(String, String)>,
    /// Map from merged string → rank (index in merges list, lower = earlier merge)
    merge_rank: HashMap<(String, String), usize>,
    /// byte → unicode character (GPT-2 byte encoding table)
    byte_encoder: [char; 256],
    /// unicode character → byte
    byte_decoder: HashMap<char, u8>,
    /// special tokens: string → id
    special_tokens: HashMap<String, u32>,
    /// BOS token id
    pub bos_token_id: Option<u32>,
    /// EOS token id
    pub eos_token_id: Option<u32>,
    /// PAD token id
    pub pad_token_id: Option<u32>,
    /// Pre-tokenizer type detected from tokenizer.json
    pre_tokenizer_type: PreTokenizerType,
}

impl Tokenizer {
    // ── GPT-2 byte ↔ unicode encoding ──────────────────────────────────────

    /// Build the GPT-2 byte encoder: maps each byte 0-255 to a printable unicode char.
    fn build_byte_encoder() -> ([char; 256], HashMap<char, u8>) {
        // Printable ASCII that don't need remapping
        let mut encoder = ['\0'; 256];
        let mut decoder = HashMap::new();

        // Collect bytes already in the "printable" range
        let mut bs: Vec<u8> = (b'!'..=b'~')
            .chain(b'\xa1'..=b'\xac')
            .chain(b'\xae'..=b'\xff')
            .collect();

        // chars for those bytes: directly to unicode codepoint
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();

        // For remaining bytes (0-32, 127-160, 172-173), map to codepoints starting at 256
        let mut n: u32 = 256;
        for b in 0u8..=255u8 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(n);
                n += 1;
            }
        }

        for (&b, &c) in bs.iter().zip(cs.iter()) {
            let ch = char::from_u32(c).unwrap_or('\u{FFFD}');
            encoder[b as usize] = ch;
            decoder.insert(ch, b);
        }
        (encoder, decoder)
    }

    /// Load a tokenizer from a HuggingFace `tokenizer.json` file.
    pub fn from_file(path: &str) -> Result<Self, AtlasError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| AtlasError::Io(format!("tokenizer.json: {e}")))?;
        Self::from_json_str(&raw)
    }

    /// Load a tokenizer from a JSON string (useful for embedding tokenizer inline).
    pub fn from_json_str(s: &str) -> Result<Self, AtlasError> {
        let root = Json::parse(s)
            .map_err(|e| AtlasError::Parse(format!("tokenizer JSON: {e}")))?;

        let (byte_encoder, byte_decoder) = Self::build_byte_encoder();

        // ── vocab ──────────────────────────────────────────────────────────
        let model = root.get("model")
            .ok_or_else(|| AtlasError::Parse("missing 'model' key".into()))?;

        let vocab_obj = model.get("vocab")
            .ok_or_else(|| AtlasError::Parse("missing 'model.vocab'".into()))?
            .as_object()
            .ok_or_else(|| AtlasError::Parse("'model.vocab' must be object".into()))?;

        let vocab_size = vocab_obj.len();
        let mut vocab = vec![String::new(); vocab_size];
        let mut vocab_map = HashMap::with_capacity(vocab_size);

        for (token, id_json) in vocab_obj {
            let id = id_json.as_usize()
                .ok_or_else(|| AtlasError::Parse(format!("non-int vocab id for '{token}'")))?
                as u32;
            if id as usize >= vocab_size {
                // extend if vocab_size was underestimated
                vocab.resize((id as usize) + 1, String::new());
            }
            vocab[id as usize] = token.clone();
            vocab_map.insert(token.clone(), id);
        }

        // ── merges ─────────────────────────────────────────────────────────
        let merges_arr = model.get("merges")
            .ok_or_else(|| AtlasError::Parse("missing 'model.merges'".into()))?
            .as_array()
            .ok_or_else(|| AtlasError::Parse("'model.merges' must be array".into()))?;

        let mut merges = Vec::with_capacity(merges_arr.len());
        let mut merge_rank = HashMap::with_capacity(merges_arr.len());

        for (rank, entry) in merges_arr.iter().enumerate() {
            let s = entry.as_str()
                .ok_or_else(|| AtlasError::Parse("merge entry must be string".into()))?;
            // Format: "tokenA tokenB" (space-separated)
            let mid = s.find(' ')
                .ok_or_else(|| AtlasError::Parse(format!("invalid merge: '{s}'")))?;
            let left  = s[..mid].to_string();
            let right = s[mid+1..].to_string();
            merge_rank.insert((left.clone(), right.clone()), rank);
            merges.push((left, right));
        }

        // ── special tokens ─────────────────────────────────────────────────
        let mut special_tokens = HashMap::new();
        let mut bos_token_id = None;
        let mut eos_token_id = None;
        let mut pad_token_id = None;

        if let Some(added) = root.get("added_tokens").and_then(|v| v.as_array()) {
            for tok in added {
                if let (Some(s), Some(id)) = (
                    tok.get("content").and_then(|x| x.as_str()),
                    tok.get("id").and_then(|x| x.as_usize()),
                ) {
                    let id = id as u32;
                    special_tokens.insert(s.to_string(), id);
                    // Extend vocab if needed
                    if id as usize >= vocab.len() {
                        vocab.resize((id as usize) + 1, String::new());
                    }
                    vocab[id as usize] = s.to_string();
                    vocab_map.insert(s.to_string(), id);
                }
            }
        }

        // Common special token names
        for (field, id_field, target) in [
            ("bos_token", "bos_token_id", &mut bos_token_id),
            ("eos_token", "eos_token_id", &mut eos_token_id),
        ] {
            if let Some(id) = root.get(field)
                .and_then(|v| v.as_str())
                .and_then(|s| vocab_map.get(s)) {
                *target = Some(*id);
            }
            if let Some(id) = root.get(id_field).and_then(|v| v.as_usize()) {
                *target = Some(id as u32);
            }
        }
        if let Some(id) = root.get("pad_token_id").and_then(|v| v.as_usize()) {
            pad_token_id = Some(id as u32);
        }

        // ── pre-tokenizer detection ────────────────────────────────────────
        let pre_tokenizer_type = Self::detect_pre_tokenizer(&root);

        Ok(Self {
            vocab,
            vocab_map,
            merges,
            merge_rank,
            byte_encoder,
            byte_decoder,
            special_tokens,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            pre_tokenizer_type,
        })
    }

    /// Detect pre-tokenizer type from tokenizer.json structure.
    fn detect_pre_tokenizer(root: &Json) -> PreTokenizerType {
        if let Some(pt) = root.get("pre_tokenizer") {
            // Check for Sequence → Split → Regex pattern (OLMo-3, LLaMA-3)
            if let Some(pretoks) = pt.get("pretokenizers").and_then(|v| v.as_array()) {
                for p in pretoks {
                    if let Some(typ) = p.get("type").and_then(|v| v.as_str()) {
                        if typ == "Split" {
                            if let Some(pattern) = p.get("pattern") {
                                if pattern.get("Regex").is_some() {
                                    return PreTokenizerType::Gpt4Regex;
                                }
                            }
                        }
                    }
                }
            }
            // Check for ByteLevel (GPT-2 style — use GPT-4 regex, compatible superset)
            if let Some(typ) = pt.get("type").and_then(|v| v.as_str()) {
                if typ == "ByteLevel" {
                    return PreTokenizerType::Gpt4Regex;
                }
                // Also check top-level Sequence type
                if typ == "Sequence" {
                    if let Some(pretoks) = pt.get("pretokenizers").and_then(|v| v.as_array()) {
                        for p in pretoks {
                            if let Some(inner_typ) = p.get("type").and_then(|v| v.as_str()) {
                                if inner_typ == "ByteLevel" {
                                    return PreTokenizerType::Gpt4Regex;
                                }
                            }
                        }
                    }
                }
            }
        }
        PreTokenizerType::Simple
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    // ── Encoding ───────────────────────────────────────────────────────────

    /// Encode a text string into token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Split on special tokens first, then BPE-encode each piece
        let pieces = self.split_on_special(text);
        let mut out = Vec::new();
        for (piece, is_special) in pieces {
            if is_special {
                if let Some(&id) = self.special_tokens.get(piece) {
                    out.push(id);
                }
            } else {
                out.extend(self.bpe_encode(piece));
            }
        }
        out
    }

    /// Decode token ids back to a UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let tok = self.id_to_token(id);
            // Each char in the token string is a GPT-2 byte-encoded character
            for c in tok.chars() {
                if let Some(&b) = self.byte_decoder.get(&c) {
                    bytes.push(b);
                }
                // Unknown chars: skip
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Convert token id to its string representation.
    pub fn id_to_token(&self, id: u32) -> &str {
        self.vocab.get(id as usize).map(|s| s.as_str()).unwrap_or("<unk>")
    }

    /// Convert token string to id, or `None` if unknown.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab_map.get(token).copied()
    }

    // ── BPE internals ────────────────────────────────────────────────────────

    fn split_on_special<'a>(&self, text: &'a str) -> Vec<(&'a str, bool)> {
        if self.special_tokens.is_empty() {
            return vec![(text, false)];
        }
        // Simple greedy split: scan for earliest/longest special token
        let mut result = Vec::new();
        let mut i = 0;
        let bytes = text.as_bytes();
        'outer: while i < text.len() {
            for (special, _) in &self.special_tokens {
                let sb = special.as_bytes();
                if bytes[i..].starts_with(sb) {
                    result.push((&text[i..i+sb.len()], true));
                    i += sb.len();
                    continue 'outer;
                }
            }
            // find next special token start
            let next = self.special_tokens.keys()
                .filter_map(|s| {
                    let sb = s.as_bytes();
                    bytes[i+1..].windows(sb.len()).position(|w| w == sb).map(|p| p + i + 1)
                })
                .min();
            let end = next.unwrap_or(text.len());
            result.push((&text[i..end], false));
            i = end;
        }
        result
    }

    fn bpe_encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() { return Vec::new(); }

        // Pre-tokenize: split into word-level pieces with byte encoding
        let words = self.pretokenize(text);
        let mut all_ids = Vec::new();
        for word_bytes in words {
            all_ids.extend(self.bpe_word(&word_bytes));
        }
        all_ids
    }

    /// Pre-tokenization: split text into word-level pieces and byte-encode.
    ///
    /// Dispatches to GPT-4 regex pattern or legacy space splitting based on
    /// the pre-tokenizer type detected from tokenizer.json.
    fn pretokenize(&self, text: &str) -> Vec<Vec<String>> {
        match self.pre_tokenizer_type {
            PreTokenizerType::Gpt4Regex => {
                gpt4_pretokenize_str(text).into_iter().map(|piece| {
                    piece.bytes()
                        .map(|b| self.byte_encoder[b as usize].to_string())
                        .collect()
                }).collect()
            }
            PreTokenizerType::Simple => self.simple_pretokenize(text),
        }
    }

    /// Legacy pre-tokenization: split on whitespace, prepend Ġ to non-first words.
    fn simple_pretokenize(&self, text: &str) -> Vec<Vec<String>> {
        let mut words = Vec::new();
        let mut current_word = Vec::new();
        let mut after_space = true; // start-of-text is treated as "after space"

        for c in text.chars() {
            if c == ' ' {
                if !current_word.is_empty() {
                    words.push(current_word.clone());
                    current_word.clear();
                }
                after_space = true;
            } else {
                if after_space && !words.is_empty() {
                    // Add space character to word start (Ġ = encoded space)
                    current_word.push(self.byte_encoder[b' ' as usize].to_string());
                }
                for b in c.to_string().bytes() {
                    current_word.push(self.byte_encoder[b as usize].to_string());
                }
                after_space = false;
            }
        }
        if !current_word.is_empty() {
            words.push(current_word);
        }
        words
    }

    /// Apply BPE merges to a single pre-tokenized word (list of byte-char strings).
    fn bpe_word(&self, word: &[String]) -> Vec<u32> {
        if word.is_empty() { return Vec::new(); }
        if word.len() == 1 {
            return vec![self.vocab_map.get(&word[0]).copied().unwrap_or(0)];
        }

        let mut tokens: Vec<String> = word.to_vec();

        loop {
            // Find the lowest-rank merge pair
            let mut best_rank = usize::MAX;
            let mut best_i = usize::MAX;

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i+1].clone());
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = i;
                    }
                }
            }

            if best_rank == usize::MAX { break; } // no more merges

            // Apply the merge
            let merged = format!("{}{}", tokens[best_i], tokens[best_i+1]);
            tokens[best_i] = merged;
            tokens.remove(best_i + 1);
        }

        tokens.iter()
            .map(|t| self.vocab_map.get(t).copied().unwrap_or(0))
            .collect()
    }
}

// ── GPT-4 regex pre-tokenizer (zero-dependency) ─────────────────────────────

/// GPT-4/OLMo-3/LLaMA-3 regex pre-tokenization (zero-dependency implementation).
///
/// Implements the pattern:
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
/// | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// All 7 alternatives are tried left-to-right at each position, matching
/// the standard NFA regex semantics. Handles UTF-8 correctly via `char_indices`.
pub fn gpt4_pretokenize_str(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut pos = 0;

    while pos < text.len() {
        let rest = &text[pos..];

        // Alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if let Some(len) = try_contraction(rest) {
            result.push(&text[pos..pos + len]);
            pos += len;
            continue;
        }

        // Alt 2: [^\r\n\p{L}\p{N}]?\p{L}+
        if let Some(len) = try_letters_with_prefix(rest) {
            result.push(&text[pos..pos + len]);
            pos += len;
            continue;
        }

        // Alt 3: \p{N}{1,3}
        if let Some(len) = try_digits(rest) {
            result.push(&text[pos..pos + len]);
            pos += len;
            continue;
        }

        // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        if let Some(len) = try_punctuation(rest) {
            result.push(&text[pos..pos + len]);
            pos += len;
            continue;
        }

        // Alt 5: \s*[\r\n]+
        if let Some(len) = try_newlines(rest) {
            result.push(&text[pos..pos + len]);
            pos += len;
            continue;
        }

        // Alt 6: \s+(?!\S)
        if let Some(len) = try_trailing_whitespace(rest) {
            result.push(&text[pos..pos + len]);
            pos += len;
            continue;
        }

        // Alt 7: \s+
        if let Some(len) = try_whitespace(rest) {
            result.push(&text[pos..pos + len]);
            pos += len;
            continue;
        }

        // Safety fallback: advance by one char (should never reach here —
        // the 7 alternatives cover all possible characters)
        let c = rest.chars().next().unwrap();
        let clen = c.len_utf8();
        result.push(&text[pos..pos + clen]);
        pos += clen;
    }

    result
}

/// Alt 1: `(?i:'s|'t|'re|'ve|'m|'ll|'d)` — contractions (case-insensitive).
fn try_contraction(s: &str) -> Option<usize> {
    let mut chars = s.chars();
    let first = chars.next()?;
    if first != '\'' { return None; }
    let flen = first.len_utf8();

    let second = chars.next()?;
    let slen = second.len_utf8();
    match second.to_ascii_lowercase() {
        's' | 't' | 'm' | 'd' => Some(flen + slen),
        'r' | 'v' | 'l' => {
            let third = chars.next()?;
            let tlen = third.len_utf8();
            match (second.to_ascii_lowercase(), third.to_ascii_lowercase()) {
                ('r', 'e') | ('v', 'e') | ('l', 'l') => Some(flen + slen + tlen),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Alt 2: `[^\r\n\p{L}\p{N}]?\p{L}+` — optional non-letter/digit/newline prefix + letters.
fn try_letters_with_prefix(s: &str) -> Option<usize> {
    let mut chars = s.chars();
    let first = chars.next()?;
    let mut byte_len = 0;

    let is_prefix_char = first != '\r' && first != '\n'
        && !first.is_alphabetic() && !first.is_numeric();

    if is_prefix_char {
        byte_len += first.len_utf8();
    } else if first.is_alphabetic() {
        byte_len += first.len_utf8();
    } else {
        return None;
    }

    // Count consecutive letters
    let mut letter_count: usize = if first.is_alphabetic() { 1 } else { 0 };
    for c in chars {
        if c.is_alphabetic() {
            byte_len += c.len_utf8();
            letter_count += 1;
        } else {
            break;
        }
    }

    if letter_count == 0 {
        return None; // Had prefix but no following letters
    }

    Some(byte_len)
}

/// Alt 3: `\p{N}{1,3}` — 1 to 3 digits.
fn try_digits(s: &str) -> Option<usize> {
    let mut byte_len = 0;
    let mut count = 0;
    for c in s.chars() {
        if c.is_numeric() && count < 3 {
            byte_len += c.len_utf8();
            count += 1;
        } else {
            break;
        }
    }
    if count > 0 { Some(byte_len) } else { None }
}

/// Alt 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*` — optional space + punctuation + optional newlines.
fn try_punctuation(s: &str) -> Option<usize> {
    let mut byte_len = 0;
    let mut it = s.chars().peekable();

    // Optional leading space (literal 0x20)
    if it.peek() == Some(&' ') {
        byte_len += 1;
        it.next();
    }

    // Required: 1+ chars that are NOT whitespace, NOT letter, NOT digit
    let mut punct_count = 0;
    while let Some(&c) = it.peek() {
        if !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric() {
            byte_len += c.len_utf8();
            punct_count += 1;
            it.next();
        } else {
            break;
        }
    }

    if punct_count == 0 {
        return None;
    }

    // Optional trailing \r or \n
    while let Some(&c) = it.peek() {
        if c == '\r' || c == '\n' {
            byte_len += c.len_utf8();
            it.next();
        } else {
            break;
        }
    }

    Some(byte_len)
}

/// Alt 5: `\s*[\r\n]+` — whitespace ending in newlines (with backtracking).
///
/// Finds the full whitespace run, then matches through the last `\r` or `\n`.
fn try_newlines(s: &str) -> Option<usize> {
    let mut ws_byte_len = 0;
    let mut last_newline_end: Option<usize> = None;

    for c in s.chars() {
        if c.is_whitespace() {
            ws_byte_len += c.len_utf8();
            if c == '\r' || c == '\n' {
                last_newline_end = Some(ws_byte_len);
            }
        } else {
            break;
        }
    }

    last_newline_end
}

/// Alt 6: `\s+(?!\S)` — trailing whitespace with negative lookahead (with backtracking).
///
/// - If run extends to end of string → match entire run.
/// - If run is 2+ chars → match (run_length − 1) chars.
/// - If run is exactly 1 char → no match (fall through to Alt 7).
fn try_trailing_whitespace(s: &str) -> Option<usize> {
    let mut ws_byte_len = 0;
    let mut ws_count: usize = 0;
    for c in s.chars() {
        if c.is_whitespace() {
            ws_byte_len += c.len_utf8();
            ws_count += 1;
        } else {
            break;
        }
    }

    if ws_count == 0 { return None; }

    // If run extends to end of string → match entire run
    if ws_byte_len == s.len() {
        return Some(ws_byte_len);
    }

    // If run is 2+ chars → match all but the last whitespace char
    if ws_count >= 2 {
        let last_char = s[..ws_byte_len].chars().next_back().unwrap();
        return Some(ws_byte_len - last_char.len_utf8());
    }

    // Run is exactly 1 char followed by non-whitespace → no match
    None
}

/// Alt 7: `\s+` — any whitespace run (greedy).
fn try_whitespace(s: &str) -> Option<usize> {
    let mut byte_len = 0;
    for c in s.chars() {
        if c.is_whitespace() {
            byte_len += c.len_utf8();
        } else {
            break;
        }
    }
    if byte_len > 0 { Some(byte_len) } else { None }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal tokenizer with explicit vocab and merges for testing.
    fn tiny_tokenizer() -> Tokenizer {
        // Build byte encoder
        let (byte_encoder, byte_decoder) = Tokenizer::build_byte_encoder();
        // Vocab: individual bytes + one merged token
        let mut vocab_map = HashMap::new();
        let mut vocab = Vec::new();
        // Add all 256 byte characters as individual tokens
        for i in 0u8..=255 {
            let s = byte_encoder[i as usize].to_string();
            vocab_map.insert(s.clone(), i as u32);
            vocab.push(s);
        }
        // Add a merged token "He" (H=72, e=101 in ASCII, encoded as themselves)
        let h_char = byte_encoder[b'H' as usize].to_string();
        let e_char = byte_encoder[b'e' as usize].to_string();
        let he = format!("{h_char}{e_char}");
        let he_id = 256u32;
        vocab_map.insert(he.clone(), he_id);
        vocab.push(he.clone());

        let l_char = byte_encoder[b'l' as usize].to_string();
        let ll = format!("{l_char}{l_char}");
        let ll_id = 257u32;
        vocab_map.insert(ll.clone(), ll_id);
        vocab.push(ll.clone());

        let mut merge_rank = HashMap::new();
        let merges = vec![
            (h_char.clone(), e_char.clone()),
            (l_char.clone(), l_char.clone()),
        ];
        for (i, (a, b)) in merges.iter().enumerate() {
            merge_rank.insert((a.clone(), b.clone()), i);
        }

        Tokenizer {
            vocab,
            vocab_map,
            merges,
            merge_rank,
            byte_encoder,
            byte_decoder,
            special_tokens: HashMap::new(),
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            pre_tokenizer_type: PreTokenizerType::Simple,
        }
    }

    #[test]
    fn byte_encoder_roundtrip() {
        let (enc, dec) = Tokenizer::build_byte_encoder();
        for b in 0u8..=255 {
            let c = enc[b as usize];
            assert_eq!(dec[&c], b, "roundtrip failed for byte {b}");
        }
    }

    #[test]
    fn tiny_tok_encode_decode() {
        let tok = tiny_tokenizer();
        // "Hello" → H(72) e(101) l(108) l(108) o(111)
        // After merges: "He" + "ll" + "o"
        let ids = tok.bpe_word(
            &["H", "e", "l", "l", "o"].map(|s| {
                tok.byte_encoder[s.as_bytes()[0] as usize].to_string()
            })
        );
        // He=256, ll=257, o=111
        assert_eq!(ids[0], 256, "He should be merged");
        assert_eq!(ids[1], 257, "ll should be merged");
    }

    #[test]
    fn decode_single_bytes() {
        let tok = tiny_tokenizer();
        // Encode "hi" → h=104, i=105 as byte ids
        let h_id = tok.vocab_map[&tok.byte_encoder[b'h' as usize].to_string()];
        let i_id = tok.vocab_map[&tok.byte_encoder[b'i' as usize].to_string()];
        let decoded = tok.decode(&[h_id, i_id]);
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn vocab_size() {
        let tok = tiny_tokenizer();
        assert_eq!(tok.vocab_size(), 258); // 256 bytes + 2 merges
    }

    #[test]
    fn special_token_split() {
        let tok = tiny_tokenizer();
        // With no special tokens, the whole string is one piece
        let pieces = tok.split_on_special("hello world");
        assert_eq!(pieces.len(), 1);
        assert!(!pieces[0].1);
    }

    // ── GPT-4 pre-tokenizer tests ────────────────────────────────────────────

    #[test]
    fn gpt4_pretokenize_basic() {
        let pieces = gpt4_pretokenize_str("The capital of France is");
        assert_eq!(pieces, vec!["The", " capital", " of", " France", " is"]);
    }

    #[test]
    fn gpt4_pretokenize_contractions() {
        let pieces = gpt4_pretokenize_str("What's the weather like?");
        assert_eq!(pieces, vec!["What", "'s", " the", " weather", " like", "?"]);
    }

    #[test]
    fn gpt4_pretokenize_numbers() {
        let pieces = gpt4_pretokenize_str("1234 tokens");
        assert_eq!(pieces, vec!["123", "4", " tokens"]);
    }

    #[test]
    fn gpt4_pretokenize_punctuation() {
        let pieces = gpt4_pretokenize_str("Hello, world!");
        assert_eq!(pieces, vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn gpt4_pretokenize_leading_spaces() {
        let pieces = gpt4_pretokenize_str("  leading spaces");
        assert_eq!(pieces, vec![" ", " leading", " spaces"]);
    }

    #[test]
    fn gpt4_pretokenize_newline() {
        let pieces = gpt4_pretokenize_str("newline\nhere");
        assert_eq!(pieces, vec!["newline", "\n", "here"]);
    }

    #[test]
    fn gpt4_pretokenize_math() {
        let pieces = gpt4_pretokenize_str("x=42+3");
        assert_eq!(pieces, vec!["x", "=", "42", "+", "3"]);
    }

    // ── OLMo-3 integration test (requires tokenizer.json on disk) ────────

    #[test]
    #[ignore] // Run with: cargo test -p atlas-tokenize -- --ignored --nocapture
    fn olmo3_encode_reference() {
        // Try standard model location, or /tmp for CI
        let paths = [
            format!("{}/models/olmo3-7b-think/tokenizer.json",
                    std::env::var("HOME").unwrap_or_else(|_| "/home/robindey".into())),
            "/tmp/olmo3_tokenizer.json".to_string(),
        ];
        let tok = paths.iter()
            .find_map(|p| Tokenizer::from_file(p).ok())
            .expect("OLMo-3 tokenizer.json not found — copy to /tmp/olmo3_tokenizer.json");

        eprintln!("  OLMo-3 vocab size: {}", tok.vocab_size());
        assert_eq!(tok.vocab_size(), 100278, "unexpected vocab size");
        assert_eq!(tok.pre_tokenizer_type, PreTokenizerType::Gpt4Regex,
            "should detect GPT-4 regex pre-tokenizer");

        // Reference encodings from HuggingFace `tokenizers` library
        let cases: &[(&str, &[u32])] = &[
            ("The capital of France is", &[791, 6864, 315, 9822, 374]),
            ("Hello, world!", &[9906, 11, 1917, 0]),
            ("1234 tokens", &[4513, 19, 11460]),
        ];

        for (text, expected) in cases {
            let ids = tok.encode(text);
            eprintln!("  encode({:30}) → {:?}", format!("{text:?}"), ids);
            assert_eq!(&ids, expected,
                "encoding mismatch for {:?}: got {:?}, expected {:?}",
                text, ids, expected);
        }

        // Round-trip decode
        for (text, ids) in cases {
            let decoded = tok.decode(ids);
            assert_eq!(decoded, *text,
                "decode mismatch for {:?}: got {:?}", ids, decoded);
        }

        eprintln!("  ✅ OLMo-3 encode/decode: all reference cases pass");
    }

    #[test]
    #[ignore]
    fn smollm2_encode_reference() {
        let paths = [
            format!("{}/models/smollm2-1b7/tokenizer.json",
                    std::env::var("HOME").unwrap_or_else(|_| "/home/robindey".into())),
        ];
        let tok = match paths.iter().find_map(|p| Tokenizer::from_file(p).ok()) {
            Some(t) => t,
            None => { eprintln!("  SKIP: SmolLM2 tokenizer.json not found"); return; }
        };

        eprintln!("  SmolLM2 vocab size: {}", tok.vocab_size());

        let cases: &[(&str, &[u32])] = &[
            ("The capital of France is", &[504, 3575, 282, 4649, 314]),
            ("Hello, world!", &[19556, 28, 905, 17]),
        ];

        for (text, expected) in cases {
            let ids = tok.encode(text);
            eprintln!("  encode({:30}) → {:?}", format!("{text:?}"), ids);
            assert_eq!(&ids, expected,
                "encoding mismatch for {:?}: got {:?}, expected {:?}",
                text, ids, expected);
        }
        eprintln!("  ✅ SmolLM2 encode/decode: all reference cases pass");
    }
}
