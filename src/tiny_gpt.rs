/*
 * @file tiny_gpt.rs
 * @brief TinyGPT model implementation
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

//! FILE: tiny_gpt.rs
//!
//! DESCRIPTION:
//! TinyGPT Model Implementation.
//!
//! BRIEF:
//! Provides complete GPT architecture with transformer blocks.
//! Implements forward pass, backpropagation, and text generation.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::block::Block;
use crate::config::CFG;
use crate::embedding::Embedding;
use crate::layer_norm::LayerNorm;
use crate::linear::Linear;
use crate::math::softmax1d;
use crate::sampling::sample;
use ndarray::{Array2, Array3, s};
use rand::rngs::ThreadRng;

/// TinyGPT language model.
///
/// # Details
/// Complete GPT architecture with embeddings, transformer blocks, and output head.
/// Supports training and autoregressive text generation.
///
/// # Fields
/// * `tok` - Token embedding layer
/// * `pos` - Position embedding layer
/// * `blocks` - Stack of transformer blocks
/// * `ln` - Final layer normalization
/// * `head` - Output projection to vocabulary
pub struct TinyGPT {
    pub(crate) tok: Embedding,
    pub(crate) pos: Embedding,
    pub(crate) blocks: Vec<Block>,
    pub(crate) ln: LayerNorm,
    pub(crate) head: Linear,
}

impl TinyGPT {
    /// Creates new TinyGPT model.
    ///
    /// # Details
    /// Initializes all model components based on configuration.
    ///
    /// # Arguments
    /// * `v` - Vocabulary size
    ///
    /// # Returns
    /// * `Self` - TinyGPT instance
    pub fn new(v: usize) -> Self {
        Self {
            tok: Embedding::new(v, CFG.embed_dim),
            pos: Embedding::new(CFG.block_size, CFG.embed_dim),
            blocks: (0..CFG.n_layers)
                .map(|_| Block::new(CFG.embed_dim, CFG.n_heads))
                .collect(),
            ln: LayerNorm::new(CFG.embed_dim),
            head: Linear::new(CFG.embed_dim, v),
        }
    }

    /// Computes forward pass for single sequence.
    ///
    /// # Details
    /// Runs token through embeddings, transformer blocks, and output head.
    ///
    /// # Arguments
    /// * `idx` - Token indices
    ///
    /// # Returns
    /// * `Array2<f32>` - Logits for each position
    pub fn forward(&self, idx: &[usize]) -> Array2<f32> {
        let mut x = &self.tok.forward(idx) + &self.pos.forward(&(0..idx.len()).collect::<Vec<_>>());
        for b in &self.blocks {
            x = b.forward(&x);
        }
        self.head.forward(&self.ln.forward(&x))
    }

    /// Computes forward pass for batch.
    ///
    /// # Details
    /// Processes multiple sequences in parallel.
    ///
    /// # Arguments
    /// * `batch` - Batch of token indices
    ///
    /// # Returns
    /// * `Array3<f32>` - Logits for each batch and position
    pub fn forward_batch(&self, batch: &Array2<usize>) -> Array3<f32> {
        let (b, t, v) = (batch.shape()[0], batch.shape()[1], self.head.w.shape()[1]);
        let mut out = Array3::zeros((b, t, v));
        for i in 0..b {
            out.slice_mut(s![i, .., ..])
                .assign(&self.forward(&batch.row(i).to_vec()));
        }
        out
    }

    /// Computes hidden states before output head.
    ///
    /// # Details
    /// Returns transformer output before final projection.
    ///
    /// # Arguments
    /// * `seq` - Token indices
    ///
    /// # Returns
    /// * `Array2<f32>` - Hidden states
    fn hidden(&self, seq: &[usize]) -> Array2<f32> {
        let mut x = &self.tok.forward(seq) + &self.pos.forward(&(0..seq.len()).collect::<Vec<_>>());
        for b in &self.blocks {
            x = b.forward(&x);
        }
        self.ln.forward(&x)
    }

    /// Computes gradients for training.
    ///
    /// # Details
    /// Performs simplified backpropagation through output head and embeddings.
    /// Uses cross-entropy loss gradient.
    ///
    /// # Arguments
    /// * `batch` - Input token indices
    /// * `targets` - Target token indices
    pub fn backward(&mut self, batch: &Array2<usize>, targets: &Array2<usize>) {
        let (b, t, v) = (batch.shape()[0], batch.shape()[1], self.head.w.shape()[1]);
        let logits = self.forward_batch(batch);
        for i in 0..b {
            let seq = batch.row(i).to_vec();
            let h = self.hidden(&seq);
            for j in 0..t {
                let mut grad = softmax1d(&logits.slice(s![i, j, ..]));
                grad[targets[[i, j]]] -= 1.0;
                grad /= (b * t) as f32;
                let hj = h.row(j);
                for k in 0..v {
                    for l in 0..CFG.embed_dim {
                        self.head.dw[[l, k]] += hj[l] * grad[k];
                    }
                    self.head.db[k] += grad[k];
                }
                let gh = self.head.w.dot(&grad);
                for l in 0..CFG.embed_dim {
                    self.tok.dw[[seq[j], l]] += gh[l];
                    self.pos.dw[[j, l]] += gh[l];
                }
            }
        }
    }

    /// Generates text autoregressively.
    ///
    /// # Details
    /// Samples tokens one at a time from model predictions.
    /// Uses context window limited by block size.
    ///
    /// # Arguments
    /// * `start` - Starting token index
    /// * `n` - Number of tokens to generate
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// * `Vec<usize>` - Generated token sequence
    pub fn generate(&self, start: usize, n: usize, rng: &mut ThreadRng) -> Vec<usize> {
        let mut out = vec![start];
        for _ in 0..n {
            let ctx: Vec<_> = out
                .iter()
                .rev()
                .take(CFG.block_size)
                .rev()
                .copied()
                .collect();
            let logits = self.forward(&ctx);
            out.push(sample(&softmax1d(&logits.row(logits.shape()[0] - 1)), rng));
        }
        out
    }

    /// Zeros all gradient accumulators.
    ///
    /// # Details
    /// Resets gradients for embeddings, blocks, and output head.
    pub fn zero_grad(&mut self) {
        self.tok.zero_grad();
        self.pos.zero_grad();
        self.head.zero_grad();
        self.blocks.iter_mut().for_each(|b| b.zero_grad());
    }

    /// Updates all parameters with gradient descent.
    ///
    /// # Details
    /// Applies gradient update to embeddings, blocks, and output head.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.tok.step(lr);
        self.pos.step(lr);
        self.head.step(lr);
        self.blocks.iter_mut().for_each(|b| b.step(lr));
    }
}

/// Computes cross-entropy loss.
///
/// # Details
/// Calculates average negative log probability of targets.
///
/// # Arguments
/// * `logits` - Model output logits
/// * `tgt` - Target token indices
///
/// # Returns
/// * `f32` - Average cross-entropy loss
pub fn cross_entropy(logits: &Array3<f32>, tgt: &Array2<usize>) -> f32 {
    let (b, t) = (logits.shape()[0], logits.shape()[1]);
    let mut loss = 0.0;
    for i in 0..b {
        for j in 0..t {
            loss -= softmax1d(&logits.slice(s![i, j, ..]))[tgt[[i, j]]].ln();
        }
    }
    loss / (b * t) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};
    use rand::rng;

    /// Tests TinyGPT creation.
    #[test]
    fn test_tiny_gpt_new() {
        let model = TinyGPT::new(100);
        assert_eq!(model.head.w.shape()[1], 100);
    }

    /// Tests forward pass shape.
    #[test]
    fn test_tiny_gpt_forward_shape() {
        let model = TinyGPT::new(50);
        let result = model.forward(&[0, 1, 2]);
        assert_eq!(result.shape(), &[3, 50]);
    }

    /// Tests forward pass produces finite values.
    #[test]
    fn test_tiny_gpt_forward_finite() {
        let model = TinyGPT::new(20);
        let result = model.forward(&[0, 1]);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    /// Tests forward_batch shape.
    #[test]
    fn test_tiny_gpt_forward_batch_shape() {
        let model = TinyGPT::new(30);
        let batch = arr2(&[[0, 1, 2], [3, 4, 5]]);
        let result = model.forward_batch(&batch);
        assert_eq!(result.shape(), &[2, 3, 30]);
    }

    /// Tests generate output length.
    #[test]
    fn test_tiny_gpt_generate_length() {
        let model = TinyGPT::new(20);
        let mut rng = rng();
        let result = model.generate(0, 5, &mut rng);
        assert_eq!(result.len(), 6); // start + 5 generated
    }

    /// Tests generate produces valid tokens.
    #[test]
    fn test_tiny_gpt_generate_valid() {
        let model = TinyGPT::new(15);
        let mut rng = rng();
        let result = model.generate(0, 3, &mut rng);
        assert!(result.iter().all(|&t| t < 15));
    }

    /// Tests zero_grad clears gradients.
    #[test]
    fn test_tiny_gpt_zero_grad() {
        let mut model = TinyGPT::new(10);
        model.head.dw.fill(1.0);
        model.tok.dw.fill(1.0);
        model.zero_grad();
        assert!(model.head.dw.iter().all(|&x| x == 0.0));
        assert!(model.tok.dw.iter().all(|&x| x == 0.0));
    }

    /// Tests step updates parameters.
    #[test]
    fn test_tiny_gpt_step() {
        let mut model = TinyGPT::new(10);
        let before = model.head.w[[0, 0]];
        model.head.dw.fill(1.0);
        model.step(0.1);
        assert!((model.head.w[[0, 0]] - (before - 0.1)).abs() < 1e-5);
    }

    /// Tests backward computes gradients.
    #[test]
    fn test_tiny_gpt_backward() {
        let mut model = TinyGPT::new(10);
        let batch = arr2(&[[0, 1], [2, 3]]);
        let targets = arr2(&[[1, 2], [3, 4]]);
        model.zero_grad();
        model.backward(&batch, &targets);
        // At least some gradient should be non-zero
        let has_grad = model.head.dw.iter().any(|&x| x != 0.0);
        assert!(has_grad);
    }

    /// Tests cross_entropy is positive.
    #[test]
    fn test_cross_entropy_positive() {
        let logits = arr3(&[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        let tgt = arr2(&[[0, 1]]);
        let loss = cross_entropy(&logits, &tgt);
        assert!(loss > 0.0);
    }

    /// Tests cross_entropy decreases with correct predictions.
    #[test]
    fn test_cross_entropy_correct() {
        let logits_good = arr3(&[[[10.0, 0.0, 0.0]]]);
        let logits_bad = arr3(&[[[0.0, 0.0, 10.0]]]);
        let tgt = arr2(&[[0]]);
        let loss_good = cross_entropy(&logits_good, &tgt);
        let loss_bad = cross_entropy(&logits_bad, &tgt);
        assert!(loss_good < loss_bad);
    }

    /// Tests cross_entropy is finite.
    #[test]
    fn test_cross_entropy_finite() {
        let logits = arr3(&[[[1.0, 2.0], [3.0, 4.0]]]);
        let tgt = arr2(&[[0, 1]]);
        let loss = cross_entropy(&logits, &tgt);
        assert!(loss.is_finite());
    }

    /// Tests single token forward.
    #[test]
    fn test_tiny_gpt_single_token() {
        let model = TinyGPT::new(10);
        let result = model.forward(&[0]);
        assert_eq!(result.shape(), &[1, 10]);
    }
}
