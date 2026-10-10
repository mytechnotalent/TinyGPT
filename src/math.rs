/*
 * @file math.rs
 * @brief Mathematical utility functions
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

//! FILE: math.rs
//!
//! DESCRIPTION:
//! Mathematical Utility Functions for TinyGPT.
//!
//! BRIEF:
//! Provides core mathematical operations for neural network computations.
//! Contains random initialization, softmax, and layer normalization functions.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use ndarray::{Array1, Array2, Axis};
use rand::rng;
use rand_distr::StandardNormal;

/// Generates random 2D array with standard normal distribution.
///
/// # Details
/// Initializes weights using scaled normal distribution.
/// Scaling factor of 0.02 provides stable training initialization.
///
/// # Arguments
/// * `r` - Number of rows
/// * `c` - Number of columns
///
/// # Returns
/// * `Array2<f32>` - Randomly initialized 2D array
pub fn randn(r: usize, c: usize) -> Array2<f32> {
    use rand::prelude::*;
    Array2::from_shape_fn((r, c), |_| rng().sample::<f32, _>(StandardNormal) * 0.02)
}

/// Computes softmax activation for 2D array.
///
/// # Details
/// Applies softmax along rows with numerical stability.
/// Subtracts maximum value before exponentiation to prevent overflow.
///
/// # Arguments
/// * `x` - Input 2D array
///
/// # Returns
/// * `Array2<f32>` - Softmax probabilities
pub fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let e = x.mapv(|v| (v - x.fold(f32::NEG_INFINITY, |a, &b| a.max(b))).exp());
    &e / &e.sum_axis(Axis(1)).insert_axis(Axis(1))
}

/// Computes softmax activation for 1D array view.
///
/// # Details
/// Applies softmax with numerical stability for single vector.
/// Used for probability distribution over vocabulary.
///
/// # Arguments
/// * `x` - Input 1D array view
///
/// # Returns
/// * `Array1<f32>` - Softmax probabilities
pub fn softmax1d(x: &ndarray::ArrayView1<f32>) -> Array1<f32> {
    let mx = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e: Array1<f32> = x.mapv(|v| (v - mx).exp());
    &e / e.sum()
}

/// Applies layer normalization to input tensor.
///
/// # Details
/// Normalizes across last dimension with learnable scale and bias.
/// Stabilizes training by normalizing activations.
///
/// # Arguments
/// * `x` - Input 2D array to normalize
/// * `g` - Gamma (scale) parameter
/// * `b` - Beta (bias) parameter
///
/// # Returns
/// * `Array2<f32>` - Normalized output
pub fn layer_norm(x: &Array2<f32>, g: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let m = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let v = x
        .mapv(|v| v * v)
        .mean_axis(Axis(1))
        .unwrap()
        .insert_axis(Axis(1))
        - &m * &m;
    &((x - &m) / v.mapv(|v| (v + 1e-5).sqrt())) * g + b
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests randn generates correct shape.
    #[test]
    fn test_randn_shape() {
        let arr = randn(3, 4);
        assert_eq!(arr.shape(), &[3, 4]);
    }

    /// Tests randn generates different values.
    #[test]
    fn test_randn_random() {
        let arr = randn(10, 10);
        let first = arr[[0, 0]];
        let has_different = arr.iter().any(|&x| (x - first).abs() > 1e-10);
        assert!(has_different);
    }

    /// Tests randn values are scaled.
    #[test]
    fn test_randn_scaled() {
        let arr = randn(100, 100);
        let max_abs = arr.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b));
        assert!(max_abs < 1.0, "Values should be scaled by 0.02");
    }

    /// Tests softmax produces valid probabilities.
    #[test]
    fn test_softmax_probabilities() {
        let x = arr2(&[[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]);
        let result = softmax(&x);
        for row in result.rows() {
            let sum: f32 = row.sum();
            assert!((sum - 1.0).abs() < 1e-5, "Row should sum to 1");
            for val in row {
                assert!(*val >= 0.0, "Probabilities should be non-negative");
                assert!(*val <= 1.0, "Probabilities should be <= 1");
            }
        }
    }

    /// Tests softmax with equal values.
    #[test]
    fn test_softmax_equal_values() {
        let x = arr2(&[[1.0, 1.0, 1.0]]);
        let result = softmax(&x);
        let expected = 1.0 / 3.0;
        for val in result.iter() {
            assert!((val - expected).abs() < 1e-5);
        }
    }

    /// Tests softmax numerical stability with large values.
    #[test]
    fn test_softmax_large_values() {
        let x = arr2(&[[1000.0, 1001.0, 1002.0]]);
        let result = softmax(&x);
        assert!(result.iter().all(|x| x.is_finite()));
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// Tests softmax1d produces valid probabilities.
    #[test]
    fn test_softmax1d_probabilities() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = softmax1d(&x.view());
        let sum: f32 = result.sum();
        assert!((sum - 1.0).abs() < 1e-5);
        for val in result.iter() {
            assert!(*val >= 0.0);
            assert!(*val <= 1.0);
        }
    }

    /// Tests softmax1d with negative values.
    #[test]
    fn test_softmax1d_negative() {
        let x = Array1::from_vec(vec![-1.0, -2.0, -3.0]);
        let result = softmax1d(&x.view());
        let sum: f32 = result.sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// Tests softmax1d preserves ordering.
    #[test]
    fn test_softmax1d_ordering() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = softmax1d(&x.view());
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    /// Tests layer_norm produces zero mean.
    #[test]
    fn test_layer_norm_zero_mean() {
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let g = Array1::ones(4);
        let b = Array1::zeros(4);
        let result = layer_norm(&x, &g, &b);
        for row in result.rows() {
            let mean: f32 = row.sum() / row.len() as f32;
            assert!(mean.abs() < 1e-5, "Mean should be approximately 0");
        }
    }

    /// Tests layer_norm with gamma scaling.
    #[test]
    fn test_layer_norm_gamma() {
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0]]);
        let g = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0]);
        let b = Array1::zeros(4);
        let result1 = layer_norm(&x, &Array1::ones(4), &b);
        let result2 = layer_norm(&x, &g, &b);
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((b - a * 2.0).abs() < 1e-5);
        }
    }

    /// Tests layer_norm with beta offset.
    #[test]
    fn test_layer_norm_beta() {
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0]]);
        let g = Array1::ones(4);
        let b = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let result = layer_norm(&x, &g, &b);
        let mean: f32 = result.row(0).sum() / 4.0;
        assert!((mean - 1.0).abs() < 1e-5);
    }

    /// Tests layer_norm output shape.
    #[test]
    fn test_layer_norm_shape() {
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let g = Array1::ones(2);
        let b = Array1::zeros(2);
        let result = layer_norm(&x, &g, &b);
        assert_eq!(result.shape(), &[3, 2]);
    }
}
