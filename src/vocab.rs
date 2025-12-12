/*
 * @file vocab.rs
 * @brief Vocabulary management
 * @author Kevin Thomas
 * @date 2025
 *
 * MIT License
 *
 * Copyright (c) 2025 Kevin Thomas
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//! FILE: vocab.rs
//!
//! DESCRIPTION:
//! Vocabulary Management for TinyGPT.
//!
//! BRIEF:
//! Provides word-to-index and index-to-word mappings.
//! Handles tokenization and detokenization of text.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use std::collections::{HashMap, HashSet};

/// Vocabulary structure for token management.
///
/// # Details
/// Manages bidirectional mapping between words and indices.
/// Stores tokenized training data.
///
/// # Fields
/// * `w2i` - Word to index mapping
/// * `i2w` - Index to word mapping
/// * `data` - Tokenized training data
/// * `size` - Vocabulary size
pub struct Vocab {
    pub w2i: HashMap<String, usize>,
    pub i2w: HashMap<usize, String>,
    pub data: Vec<usize>,
    pub size: usize,
}

impl Vocab {
    /// Creates vocabulary from corpus.
    ///
    /// # Details
    /// Builds word-to-index mappings from training corpus.
    /// Appends END token to each sentence.
    ///
    /// # Arguments
    /// * `corpus` - Array of training sentences
    ///
    /// # Returns
    /// * `Self` - Vocabulary instance
    pub fn from_corpus(corpus: &[String]) -> Self {
        let text = corpus
            .iter()
            .map(|s| format!("{} <END>", s))
            .collect::<Vec<_>>()
            .join(" ");
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique: Vec<_> = words
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let w2i: HashMap<_, _> = unique
            .iter()
            .enumerate()
            .map(|(i, &w)| (w.to_string(), i))
            .collect();
        let i2w: HashMap<_, _> = w2i.iter().map(|(w, &i)| (i, w.clone())).collect();
        let data: Vec<usize> = words.iter().map(|w| w2i[*w]).collect();
        Self {
            size: unique.len(),
            w2i,
            i2w,
            data,
        }
    }

    /// Encodes word to index.
    ///
    /// # Details
    /// Converts word string to vocabulary index.
    ///
    /// # Arguments
    /// * `word` - Word to encode
    ///
    /// # Returns
    /// * `usize` - Vocabulary index
    pub fn encode(&self, word: &str) -> usize {
        self.w2i[word]
    }

    /// Decodes token sequence to text.
    ///
    /// # Details
    /// Converts array of indices back to words.
    /// Joins words with spaces.
    ///
    /// # Arguments
    /// * `tokens` - Array of token indices
    ///
    /// # Returns
    /// * `String` - Decoded text
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&i| self.i2w[&i].as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests vocabulary from corpus.
    #[test]
    fn test_vocab_from_corpus() {
        let corpus = vec!["hello world".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        assert!(vocab.size >= 3); // hello, world, <END>
    }

    /// Tests vocabulary encode.
    #[test]
    fn test_vocab_encode() {
        let corpus = vec!["foo bar".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        let foo_idx = vocab.encode("foo");
        let bar_idx = vocab.encode("bar");
        assert_ne!(foo_idx, bar_idx);
    }

    /// Tests vocabulary decode.
    #[test]
    fn test_vocab_decode() {
        let corpus = vec!["abc def".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        let abc_idx = vocab.encode("abc");
        let decoded = vocab.decode(&[abc_idx]);
        assert_eq!(decoded, "abc");
    }

    /// Tests encode/decode roundtrip.
    #[test]
    fn test_vocab_roundtrip() {
        let corpus = vec!["the quick brown".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        let idx = vocab.encode("quick");
        let decoded = vocab.decode(&[idx]);
        assert_eq!(decoded, "quick");
    }

    /// Tests vocabulary contains END token.
    #[test]
    fn test_vocab_end_token() {
        let corpus = vec!["test".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        assert!(vocab.w2i.contains_key("<END>"));
    }

    /// Tests vocabulary data field.
    #[test]
    fn test_vocab_data() {
        let corpus = vec!["a b".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        assert!(!vocab.data.is_empty());
    }

    /// Tests decode multiple tokens.
    #[test]
    fn test_vocab_decode_multiple() {
        let corpus = vec!["one two three".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        let one = vocab.encode("one");
        let two = vocab.encode("two");
        let decoded = vocab.decode(&[one, two]);
        assert!(decoded.contains("one"));
        assert!(decoded.contains("two"));
    }

    /// Tests vocabulary handles duplicates.
    #[test]
    fn test_vocab_duplicates() {
        let corpus = vec!["hello hello hello".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        // Should have "hello" and "<END>" only
        assert!(vocab.size >= 2);
        assert!(vocab.size <= 3); // hello, <END>, and maybe whitespace artifact
    }

    /// Tests i2w and w2i consistency.
    #[test]
    fn test_vocab_bidirectional() {
        let corpus = vec!["alpha beta".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        for (word, &idx) in &vocab.w2i {
            assert_eq!(&vocab.i2w[&idx], word);
        }
    }

    /// Tests vocabulary size matches unique words.
    #[test]
    fn test_vocab_size() {
        let corpus = vec!["a b c".to_string()];
        let vocab = Vocab::from_corpus(&corpus);
        assert_eq!(vocab.size, vocab.w2i.len());
        assert_eq!(vocab.size, vocab.i2w.len());
    }
}
