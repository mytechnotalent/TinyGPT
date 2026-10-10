/*
 * @file attention.rs
 * @brief Multi-head attention implementation
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

//! FILE: attention.rs
//!
//! DESCRIPTION:
//! Multi-Head Attention Implementation for TinyGPT.
//!
//! BRIEF:
//! Provides parallel attention heads with output projection.
//! Concatenates head outputs and projects to embedding dimension.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::head::Head;
use crate::linear::Linear;
use ndarray::{Array2, Axis};

/// Multi-head attention module.
///
/// # Details
/// Implements parallel attention heads with concatenation and projection.
/// Each head attends to different representation subspaces.
///
/// # Fields
/// * `heads` - Vector of attention heads
/// * `proj` - Output projection layer
#[derive(Clone)]
pub struct MultiHeadAttention {
    pub(crate) heads: Vec<Head>,
    pub(crate) proj: Linear,
}

impl MultiHeadAttention {
    /// Creates new multi-head attention.
    ///
    /// # Details
    /// Initializes specified number of attention heads.
    /// Head size is embedding dimension divided by number of heads.
    ///
    /// # Arguments
    /// * `d` - Embedding dimension
    /// * `nh` - Number of attention heads
    ///
    /// # Returns
    /// * `Self` - MultiHeadAttention instance
    pub fn new(d: usize, nh: usize) -> Self {
        Self {
            heads: (0..nh).map(|_| Head::new(d, d / nh)).collect(),
            proj: Linear::new(d, d),
        }
    }

    /// Computes multi-head attention output.
    ///
    /// # Details
    /// Runs all heads in parallel, concatenates outputs.
    /// Projects concatenated output back to embedding dimension.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - Attention output
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let o: Vec<_> = self.heads.iter().map(|h| h.forward(x)).collect();
        let v: Vec<_> = o.iter().map(|a| a.view()).collect();
        self.proj
            .forward(&ndarray::concatenate(Axis(1), &v).unwrap())
    }

    /// Zeros gradient accumulators.
    ///
    /// # Details
    /// Resets gradients for all heads and projection.
    pub fn zero_grad(&mut self) {
        self.heads.iter_mut().for_each(|h| h.zero_grad());
        self.proj.zero_grad();
    }

    /// Updates parameters with gradient descent.
    ///
    /// # Details
    /// Applies gradient update to all heads and projection.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.heads.iter_mut().for_each(|h| h.step(lr));
        self.proj.step(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests multi-head attention creation.
    #[test]
    fn test_mha_new() {
        let mha = MultiHeadAttention::new(16, 4);
        assert_eq!(mha.heads.len(), 4);
    }

    /// Tests multi-head attention forward shape.
    #[test]
    fn test_mha_forward_shape() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = arr2(&[[1.0; 8], [2.0; 8], [3.0; 8]]);
        let result = mha.forward(&x);
        assert_eq!(result.shape(), &[3, 8]);
    }

    /// Tests multi-head attention produces finite values.
    #[test]
    fn test_mha_forward_finite() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = arr2(&[[1.0; 8], [2.0; 8]]);
        let result = mha.forward(&x);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    /// Tests multi-head attention zero_grad.
    #[test]
    fn test_mha_zero_grad() {
        let mut mha = MultiHeadAttention::new(8, 2);
        for h in &mut mha.heads {
            h.key.dw.fill(1.0);
        }
        mha.proj.dw.fill(1.0);
        mha.zero_grad();
        for h in &mha.heads {
            assert!(h.key.dw.iter().all(|&x| x == 0.0));
        }
        assert!(mha.proj.dw.iter().all(|&x| x == 0.0));
    }

    /// Tests multi-head attention step.
    #[test]
    fn test_mha_step() {
        let mut mha = MultiHeadAttention::new(8, 2);
        let before = mha.proj.w[[0, 0]];
        mha.proj.dw.fill(1.0);
        mha.step(0.1);
        assert!((mha.proj.w[[0, 0]] - (before - 0.1)).abs() < 1e-5);
    }

    /// Tests multi-head attention clone.
    #[test]
    fn test_mha_clone() {
        let mha1 = MultiHeadAttention::new(8, 4);
        let mha2 = mha1.clone();
        assert_eq!(mha1.heads.len(), mha2.heads.len());
    }

    /// Tests single head configuration.
    #[test]
    fn test_mha_single_head() {
        let mha = MultiHeadAttention::new(4, 1);
        let x = arr2(&[[1.0; 4]]);
        let result = mha.forward(&x);
        assert_eq!(result.shape(), &[1, 4]);
    }

    /// Tests output projection maintains dimension.
    #[test]
    fn test_mha_dimension_preserved() {
        let mha = MultiHeadAttention::new(16, 4);
        let x = arr2(&[[1.0; 16], [2.0; 16]]);
        let result = mha.forward(&x);
        assert_eq!(result.shape()[1], 16);
    }
}
