/*
 * @file block.rs
 * @brief Transformer block implementation
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

//! FILE: block.rs
//!
//! DESCRIPTION:
//! Transformer Block Implementation for TinyGPT.
//!
//! BRIEF:
//! Provides complete transformer block with attention and feed-forward.
//! Implements pre-norm architecture with residual connections.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::attention::MultiHeadAttention;
use crate::feed_forward::FeedForward;
use crate::layer_norm::LayerNorm;
use ndarray::Array2;

/// Transformer block module.
///
/// # Details
/// Combines multi-head attention and feed-forward network.
/// Uses pre-normalization and residual connections.
///
/// # Fields
/// * `sa` - Self-attention module
/// * `ffn` - Feed-forward network
/// * `ln1` - Layer norm before attention
/// * `ln2` - Layer norm before feed-forward
#[derive(Clone)]
pub struct Block {
    pub(crate) sa: MultiHeadAttention,
    pub(crate) ffn: FeedForward,
    pub(crate) ln1: LayerNorm,
    pub(crate) ln2: LayerNorm,
}

impl Block {
    /// Creates new transformer block.
    ///
    /// # Details
    /// Initializes attention, feed-forward, and layer norms.
    ///
    /// # Arguments
    /// * `d` - Embedding dimension
    /// * `nh` - Number of attention heads
    ///
    /// # Returns
    /// * `Self` - Block instance
    pub fn new(d: usize, nh: usize) -> Self {
        Self {
            sa: MultiHeadAttention::new(d, nh),
            ffn: FeedForward::new(d),
            ln1: LayerNorm::new(d),
            ln2: LayerNorm::new(d),
        }
    }

    /// Computes block output.
    ///
    /// # Details
    /// Applies attention and feed-forward with residual connections.
    /// Uses pre-norm architecture: LN -> sublayer -> residual.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - Block output
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let x = x + &self.sa.forward(&self.ln1.forward(x));
        &x + &self.ffn.forward(&self.ln2.forward(&x))
    }

    /// Zeros gradient accumulators.
    ///
    /// # Details
    /// Resets gradients for attention and feed-forward.
    pub fn zero_grad(&mut self) {
        self.sa.zero_grad();
        self.ffn.zero_grad();
    }

    /// Updates parameters with gradient descent.
    ///
    /// # Details
    /// Applies gradient update to attention and feed-forward.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.sa.step(lr);
        self.ffn.step(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests block creation.
    #[test]
    fn test_block_new() {
        let b = Block::new(16, 4);
        assert_eq!(b.ln1.g.len(), 16);
        assert_eq!(b.ln2.g.len(), 16);
    }

    /// Tests block forward shape.
    #[test]
    fn test_block_forward_shape() {
        let b = Block::new(8, 2);
        let x = arr2(&[[1.0; 8], [2.0; 8]]);
        let result = b.forward(&x);
        assert_eq!(result.shape(), &[2, 8]);
    }

    /// Tests block forward produces finite values.
    #[test]
    fn test_block_forward_finite() {
        let b = Block::new(8, 2);
        let x = arr2(&[[1.0; 8], [2.0; 8]]);
        let result = b.forward(&x);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    /// Tests block residual connection.
    #[test]
    fn test_block_residual() {
        let b = Block::new(4, 1);
        let x = arr2(&[[1.0; 4]]);
        let result = b.forward(&x);
        // Output should be finite
        assert!(result.iter().all(|v| v.is_finite()));
    }

    /// Tests block zero_grad.
    #[test]
    fn test_block_zero_grad() {
        let mut b = Block::new(8, 2);
        b.sa.proj.dw.fill(1.0);
        b.ffn.l1.dw.fill(1.0);
        b.zero_grad();
        assert!(b.sa.proj.dw.iter().all(|&x| x == 0.0));
        assert!(b.ffn.l1.dw.iter().all(|&x| x == 0.0));
    }

    /// Tests block step.
    #[test]
    fn test_block_step() {
        let mut b = Block::new(8, 2);
        let before = b.sa.proj.w[[0, 0]];
        b.sa.proj.dw.fill(1.0);
        b.step(0.1);
        assert!((b.sa.proj.w[[0, 0]] - (before - 0.1)).abs() < 1e-5);
    }

    /// Tests block clone.
    #[test]
    fn test_block_clone() {
        let b1 = Block::new(8, 2);
        let b2 = b1.clone();
        assert_eq!(b1.ln1.g.len(), b2.ln1.g.len());
    }

    /// Tests dimension is preserved through block.
    #[test]
    fn test_block_dimension_preserved() {
        let b = Block::new(16, 4);
        let x = arr2(&[[1.0; 16], [2.0; 16], [3.0; 16]]);
        let result = b.forward(&x);
        assert_eq!(result.shape(), &[3, 16]);
    }
}
