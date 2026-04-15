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
        })
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    // ── Encoding ────────────────────────────────────────────────────────────

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

        // Convert each byte to its GPT-2 unicode representation
        let byte_chars: Vec<String> = text.bytes()
            .map(|b| self.byte_encoder[b as usize].to_string())
            .collect();

        // Pre-tokenize: split on spaces to create word boundaries
        // GPT-2 style: prepend 'Ġ' (unicode 0x0120) to tokens following a space
        let words = self.pretokenize(text);
        let mut all_ids = Vec::new();
        for word_bytes in words {
            all_ids.extend(self.bpe_word(&word_bytes));
        }
        all_ids
    }

    /// GPT-2 pre-tokenization: split on whitespace, prepend Ġ to non-first words.
    fn pretokenize(&self, text: &str) -> Vec<Vec<String>> {
        // Simplified: split on spaces, prepend byte-encoded space (Ġ) to each word
        // that follows whitespace. This matches HuggingFace ByteLevelBPETokenizer.
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
}
