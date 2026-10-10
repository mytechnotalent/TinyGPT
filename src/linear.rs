/*
 * @file linear.rs
 * @brief Linear layer implementation
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

//! FILE: linear.rs
//!
//! DESCRIPTION:
//! Linear Layer Implementation for TinyGPT.
//!
//! BRIEF:
//! Provides fully connected linear layer with weight and bias.
//! Supports forward pass and gradient accumulation.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::math::randn;
use ndarray::{Array1, Array2};

/// Linear layer for neural network.
///
/// # Details
/// Implements y = xW + b transformation.
/// Stores weights, biases, and their gradients.
///
/// # Fields
/// * `w` - Weight matrix
/// * `b` - Bias vector
/// * `dw` - Weight gradient
/// * `db` - Bias gradient
#[derive(Clone)]
pub struct Linear {
    pub w: Array2<f32>,
    pub b: Array1<f32>,
    pub dw: Array2<f32>,
    pub db: Array1<f32>,
}

impl Linear {
    /// Creates new linear layer.
    ///
    /// # Details
    /// Initializes weights with random normal distribution.
    /// Biases initialized to zero.
    ///
    /// # Arguments
    /// * `i` - Input dimension
    /// * `o` - Output dimension
    ///
    /// # Returns
    /// * `Self` - Linear layer instance
    pub fn new(i: usize, o: usize) -> Self {
        Self {
            w: randn(i, o),
            b: Array1::zeros(o),
            dw: Array2::zeros((i, o)),
            db: Array1::zeros(o),
        }
    }

    /// Computes forward pass.
    ///
    /// # Details
    /// Applies linear transformation: y = xW + b.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - Output tensor
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.w) + &self.b
    }

    /// Zeros gradient accumulators.
    ///
    /// # Details
    /// Resets weight and bias gradients to zero.
    pub fn zero_grad(&mut self) {
        self.dw.fill(0.0);
        self.db.fill(0.0);
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
        self.b = &self.b - lr * &self.db;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests linear layer creation.
    #[test]
    fn test_linear_new() {
        let l = Linear::new(4, 8);
        assert_eq!(l.w.shape(), &[4, 8]);
        assert_eq!(l.b.shape(), &[8]);
        assert_eq!(l.dw.shape(), &[4, 8]);
        assert_eq!(l.db.shape(), &[8]);
    }

    /// Tests linear forward output shape.
    #[test]
    fn test_linear_forward_shape() {
        let l = Linear::new(4, 8);
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let y = l.forward(&x);
        assert_eq!(y.shape(), &[2, 8]);
    }

    /// Tests linear forward computation.
    #[test]
    fn test_linear_forward_computation() {
        let mut l = Linear::new(2, 2);
        l.w = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        l.b = Array1::from_vec(vec![1.0, 2.0]);
        let x = arr2(&[[1.0, 2.0]]);
        let y = l.forward(&x);
        assert!((y[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((y[[0, 1]] - 4.0).abs() < 1e-5);
    }

    /// Tests zero_grad clears gradients.
    #[test]
    fn test_linear_zero_grad() {
        let mut l = Linear::new(2, 2);
        l.dw.fill(1.0);
        l.db.fill(1.0);
        l.zero_grad();
        assert!(l.dw.iter().all(|&x| x == 0.0));
        assert!(l.db.iter().all(|&x| x == 0.0));
    }

    /// Tests step updates weights.
    #[test]
    fn test_linear_step() {
        let mut l = Linear::new(2, 2);
        l.w = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        l.b = Array1::from_vec(vec![1.0, 1.0]);
        l.dw = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        l.db = Array1::from_vec(vec![1.0, 1.0]);
        l.step(0.1);
        assert!((l.w[[0, 0]] - 0.9).abs() < 1e-5);
        assert!((l.b[0] - 0.9).abs() < 1e-5);
    }

    /// Tests linear layer clone.
    #[test]
    fn test_linear_clone() {
        let l1 = Linear::new(3, 4);
        let l2 = l1.clone();
        assert_eq!(l1.w.shape(), l2.w.shape());
        assert_eq!(l1.b.shape(), l2.b.shape());
    }

    /// Tests bias is added correctly.
    #[test]
    fn test_linear_bias() {
        let mut l = Linear::new(2, 3);
        l.w = Array2::zeros((2, 3));
        l.b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x = arr2(&[[0.0, 0.0]]);
        let y = l.forward(&x);
        assert!((y[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((y[[0, 1]] - 2.0).abs() < 1e-5);
        assert!((y[[0, 2]] - 3.0).abs() < 1e-5);
    }
}
