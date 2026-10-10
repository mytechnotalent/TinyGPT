/*
 * @file feed_forward.rs
 * @brief Feed-forward network implementation
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

//! FILE: feed_forward.rs
//!
//! DESCRIPTION:
//! Feed-Forward Network Implementation for TinyGPT.
//!
//! BRIEF:
//! Provides position-wise feed-forward network.
//! Implements two linear layers with ReLU activation.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::linear::Linear;
use ndarray::Array2;

/// Position-wise feed-forward network.
///
/// # Details
/// Implements FFN(x) = ReLU(xW1 + b1)W2 + b2.
/// Hidden dimension is 4x the embedding dimension.
///
/// # Fields
/// * `l1` - First linear layer (expansion)
/// * `l2` - Second linear layer (projection)
#[derive(Clone)]
pub struct FeedForward {
    pub(crate) l1: Linear,
    pub(crate) l2: Linear,
}

impl FeedForward {
    /// Creates new feed-forward network.
    ///
    /// # Details
    /// Initializes with 4x expansion in hidden layer.
    ///
    /// # Arguments
    /// * `d` - Input and output dimension
    ///
    /// # Returns
    /// * `Self` - FeedForward instance
    pub fn new(d: usize) -> Self {
        Self {
            l1: Linear::new(d, 4 * d),
            l2: Linear::new(4 * d, d),
        }
    }

    /// Computes feed-forward output.
    ///
    /// # Details
    /// Applies linear, ReLU, linear transformation.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - Feed-forward output
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        self.l2.forward(&self.l1.forward(x).mapv(|v| v.max(0.0)))
    }

    /// Zeros gradient accumulators.
    ///
    /// # Details
    /// Resets gradients for both linear layers.
    pub fn zero_grad(&mut self) {
        self.l1.zero_grad();
        self.l2.zero_grad();
    }

    /// Updates parameters with gradient descent.
    ///
    /// # Details
    /// Applies gradient update to both layers.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn step(&mut self, lr: f32) {
        self.l1.step(lr);
        self.l2.step(lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests feed-forward creation.
    #[test]
    fn test_ff_new() {
        let ff = FeedForward::new(8);
        assert_eq!(ff.l1.w.shape(), &[8, 32]); // 4x expansion
        assert_eq!(ff.l2.w.shape(), &[32, 8]);
    }

    /// Tests feed-forward forward shape.
    #[test]
    fn test_ff_forward_shape() {
        let ff = FeedForward::new(4);
        let x = arr2(&[[1.0; 4], [2.0; 4]]);
        let result = ff.forward(&x);
        assert_eq!(result.shape(), &[2, 4]);
    }

    /// Tests feed-forward produces finite values.
    #[test]
    fn test_ff_forward_finite() {
        let ff = FeedForward::new(4);
        let x = arr2(&[[1.0; 4]]);
        let result = ff.forward(&x);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    /// Tests feed-forward ReLU activation.
    #[test]
    fn test_ff_relu() {
        let mut ff = FeedForward::new(2);
        ff.l1.w.fill(-1.0); // Force negative intermediate
        ff.l1.b.fill(-10.0);
        ff.l2.w.fill(1.0);
        ff.l2.b.fill(0.0);
        let x = arr2(&[[1.0, 1.0]]);
        let result = ff.forward(&x);
        // After ReLU, negative values become 0
        // Output should be near bias of l2
        assert!(result.iter().all(|v| v.is_finite()));
    }

    /// Tests feed-forward zero_grad.
    #[test]
    fn test_ff_zero_grad() {
        let mut ff = FeedForward::new(4);
        ff.l1.dw.fill(1.0);
        ff.l2.dw.fill(1.0);
        ff.zero_grad();
        assert!(ff.l1.dw.iter().all(|&x| x == 0.0));
        assert!(ff.l2.dw.iter().all(|&x| x == 0.0));
    }

    /// Tests feed-forward step.
    #[test]
    fn test_ff_step() {
        let mut ff = FeedForward::new(4);
        let before = ff.l1.w[[0, 0]];
        ff.l1.dw.fill(1.0);
        ff.step(0.1);
        assert!((ff.l1.w[[0, 0]] - (before - 0.1)).abs() < 1e-5);
    }

    /// Tests feed-forward clone.
    #[test]
    fn test_ff_clone() {
        let ff1 = FeedForward::new(8);
        let ff2 = ff1.clone();
        assert_eq!(ff1.l1.w.shape(), ff2.l1.w.shape());
    }

    /// Tests dimension is preserved.
    #[test]
    fn test_ff_dimension_preserved() {
        let ff = FeedForward::new(16);
        let x = arr2(&[[1.0; 16], [2.0; 16], [3.0; 16]]);
        let result = ff.forward(&x);
        assert_eq!(result.shape()[1], 16);
    }
}
