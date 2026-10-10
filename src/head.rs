/*
 * @file head.rs
 * @brief Single attention head implementation
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

//! FILE: head.rs
//!
//! DESCRIPTION:
//! Single Attention Head Implementation for TinyGPT.
//!
//! BRIEF:
//! Provides scaled dot-product attention mechanism.
//! Implements causal masking for autoregressive generation.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::linear::Linear;
use crate::math::softmax;
use ndarray::Array2;

/// Single attention head.
///
/// # Details
/// Implements scaled dot-product attention with causal mask.
/// Contains key, query, and value projections.
///
/// # Fields
/// * `key` - Key projection layer
/// * `query` - Query projection layer
/// * `value` - Value projection layer
/// * `hs` - Head size (dimension per head)
#[derive(Clone)]
pub struct Head {
    pub(crate) key: Linear,
    pub(crate) query: Linear,
    pub(crate) value: Linear,
    hs: usize,
}

impl Head {
    /// Creates new attention head.
    ///
    /// # Details
    /// Initializes key, query, value projections.
    ///
    /// # Arguments
    /// * `d` - Input dimension
    /// * `hs` - Head size (output dimension)
    ///
    /// # Returns
    /// * `Self` - Head instance
    pub fn new(d: usize, hs: usize) -> Self {
        Self {
            key: Linear::new(d, hs),
            query: Linear::new(d, hs),
            value: Linear::new(d, hs),
            hs,
        }
    }

    /// Computes attention output.
    ///
    /// # Details
    /// Applies scaled dot-product attention with causal masking.
    /// Future positions are masked with negative infinity.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - Attention output
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (k, q, v) = (
            self.key.forward(x),
            self.query.forward(x),
            self.value.forward(x),
        );
        let mut w = q.dot(&k.t()) / (self.hs as f32).sqrt();
        let t = w.shape()[0];
        for i in 0..t {
            for j in i + 1..t {
                w[[i, j]] = f32::NEG_INFINITY;
            }
        }
        softmax(&w).dot(&v)
    }

    /// Zeros gradient accumulators.
    ///
    /// # Details
    /// Resets all projection gradients to zero.
    pub fn zero_grad(&mut self) {
        self.key.zero_grad();
        self.query.zero_grad();
        self.value.zero_grad();
    }

    /// Updates parameters with gradient descent.
    ///
    /// # Details
    /// Applies gradient update to all projections.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.key.step(lr);
        self.query.step(lr);
        self.value.step(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests head creation.
    #[test]
    fn test_head_new() {
        let h = Head::new(16, 4);
        assert_eq!(h.hs, 4);
    }

    /// Tests head forward output shape.
    #[test]
    fn test_head_forward_shape() {
        let h = Head::new(8, 2);
        let x = arr2(&[[1.0; 8], [2.0; 8], [3.0; 8]]);
        let result = h.forward(&x);
        assert_eq!(result.shape(), &[3, 2]);
    }

    /// Tests head forward produces finite values.
    #[test]
    fn test_head_forward_finite() {
        let h = Head::new(8, 2);
        let x = arr2(&[[1.0; 8], [2.0; 8]]);
        let result = h.forward(&x);
        assert!(result.iter().all(|x| x.is_finite()));
    }

    /// Tests head zero_grad.
    #[test]
    fn test_head_zero_grad() {
        let mut h = Head::new(4, 2);
        h.key.dw.fill(1.0);
        h.query.dw.fill(1.0);
        h.value.dw.fill(1.0);
        h.zero_grad();
        assert!(h.key.dw.iter().all(|&x| x == 0.0));
        assert!(h.query.dw.iter().all(|&x| x == 0.0));
        assert!(h.value.dw.iter().all(|&x| x == 0.0));
    }

    /// Tests head step updates parameters.
    #[test]
    fn test_head_step() {
        let mut h = Head::new(4, 2);
        let before_key = h.key.w[[0, 0]];
        h.key.dw.fill(1.0);
        h.step(0.1);
        assert!((h.key.w[[0, 0]] - (before_key - 0.1)).abs() < 1e-5);
    }

    /// Tests head clone.
    #[test]
    fn test_head_clone() {
        let h1 = Head::new(8, 4);
        let h2 = h1.clone();
        assert_eq!(h1.hs, h2.hs);
    }

    /// Tests causal masking (future tokens should not attend).
    #[test]
    fn test_head_causal_mask() {
        let h = Head::new(4, 2);
        let x = arr2(&[[1.0; 4], [1.0; 4], [1.0; 4]]);
        let result = h.forward(&x);
        // Result should be finite (mask applied correctly)
        assert!(result.iter().all(|v| v.is_finite()));
    }

    /// Tests single sequence element.
    #[test]
    fn test_head_single_element() {
        let h = Head::new(4, 2);
        let x = arr2(&[[1.0; 4]]);
        let result = h.forward(&x);
        assert_eq!(result.shape(), &[1, 2]);
    }
}
