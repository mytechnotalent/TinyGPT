/*
 * @file embedding.rs
 * @brief Embedding layer implementation
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

//! FILE: embedding.rs
//!
//! DESCRIPTION:
//! Embedding Layer Implementation for TinyGPT.
//!
//! BRIEF:
//! Provides token and position embedding lookup table.
//! Maps discrete indices to continuous vectors.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::math::randn;
use ndarray::Array2;

/// Embedding layer for token lookup.
///
/// # Details
/// Implements learnable embedding table.
/// Stores embedding weights and gradients.
///
/// # Fields
/// * `w` - Embedding weight matrix
/// * `dw` - Embedding gradient matrix
#[derive(Clone)]
pub struct Embedding {
    pub w: Array2<f32>,
    pub dw: Array2<f32>,
}

impl Embedding {
    /// Creates new embedding layer.
    ///
    /// # Details
    /// Initializes embeddings with random normal distribution.
    ///
    /// # Arguments
    /// * `n` - Number of embeddings (vocabulary size)
    /// * `d` - Embedding dimension
    ///
    /// # Returns
    /// * `Self` - Embedding layer instance
    pub fn new(n: usize, d: usize) -> Self {
        Self {
            w: randn(n, d),
            dw: Array2::zeros((n, d)),
        }
    }

    /// Looks up embeddings for indices.
    ///
    /// # Details
    /// Gathers embedding vectors for input indices.
    ///
    /// # Arguments
    /// * `idx` - Array of indices to look up
    ///
    /// # Returns
    /// * `Array2<f32>` - Stacked embedding vectors
    pub fn forward(&self, idx: &[usize]) -> Array2<f32> {
        Array2::from_shape_fn((idx.len(), self.w.shape()[1]), |(i, j)| self.w[[idx[i], j]])
    }

    /// Zeros gradient accumulators.
    ///
    /// # Details
    /// Resets embedding gradients to zero.
    pub fn zero_grad(&mut self) {
        self.dw.fill(0.0);
    }

    /// Updates parameters with gradient descent.
    ///
    /// # Details
    /// Applies gradient update: param = param - lr * grad.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.w = &self.w - lr * &self.dw;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests embedding creation.
    #[test]
    fn test_embedding_new() {
        let e = Embedding::new(10, 4);
        assert_eq!(e.w.shape(), &[10, 4]);
        assert_eq!(e.dw.shape(), &[10, 4]);
    }

    /// Tests embedding forward single index.
    #[test]
    fn test_embedding_forward_single() {
        let e = Embedding::new(10, 4);
        let result = e.forward(&[0]);
        assert_eq!(result.shape(), &[1, 4]);
    }

    /// Tests embedding forward multiple indices.
    #[test]
    fn test_embedding_forward_multiple() {
        let e = Embedding::new(10, 4);
        let result = e.forward(&[0, 1, 2]);
        assert_eq!(result.shape(), &[3, 4]);
    }

    /// Tests embedding lookup correctness.
    #[test]
    fn test_embedding_lookup() {
        let mut e = Embedding::new(3, 2);
        e.w = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let result = e.forward(&[1]);
        assert!((result[[0, 0]] - 3.0).abs() < 1e-5);
        assert!((result[[0, 1]] - 4.0).abs() < 1e-5);
    }

    /// Tests embedding zero_grad.
    #[test]
    fn test_embedding_zero_grad() {
        let mut e = Embedding::new(5, 3);
        e.dw.fill(1.0);
        e.zero_grad();
        assert!(e.dw.iter().all(|&x| x == 0.0));
    }

    /// Tests embedding step.
    #[test]
    fn test_embedding_step() {
        let mut e = Embedding::new(2, 2);
        e.w = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        e.dw = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        e.step(0.1);
        assert!((e.w[[0, 0]] - 0.9).abs() < 1e-5);
    }

    /// Tests embedding clone.
    #[test]
    fn test_embedding_clone() {
        let e1 = Embedding::new(5, 3);
        let e2 = e1.clone();
        assert_eq!(e1.w.shape(), e2.w.shape());
    }

    /// Tests embedding forward with repeated indices.
    #[test]
    fn test_embedding_repeated_indices() {
        let mut e = Embedding::new(3, 2);
        e.w = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let result = e.forward(&[0, 0, 0]);
        assert_eq!(result.shape(), &[3, 2]);
        for i in 0..3 {
            assert!((result[[i, 0]] - 1.0).abs() < 1e-5);
        }
    }

    /// Tests embedding with last index.
    #[test]
    fn test_embedding_last_index() {
        let mut e = Embedding::new(5, 2);
        e.w = arr2(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [9.0, 9.0]]);
        let result = e.forward(&[4]);
        assert!((result[[0, 0]] - 9.0).abs() < 1e-5);
    }
}
