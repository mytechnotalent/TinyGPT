/*
 * @file layer_norm.rs
 * @brief Layer normalization implementation
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

//! FILE: layer_norm.rs
//!
//! DESCRIPTION:
//! Layer Normalization Implementation for TinyGPT.
//!
//! BRIEF:
//! Provides layer normalization with learnable parameters.
//! Stabilizes training by normalizing activations.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::math::layer_norm;
use ndarray::{Array1, Array2};

/// Layer normalization module.
///
/// # Details
/// Implements layer normalization with scale and shift.
/// Normalizes across the embedding dimension.
///
/// # Fields
/// * `g` - Gamma (scale) parameter
/// * `b` - Beta (shift) parameter
#[derive(Clone)]
pub struct LayerNorm {
    pub g: Array1<f32>,
    pub b: Array1<f32>,
}

impl LayerNorm {
    /// Creates new layer normalization.
    ///
    /// # Details
    /// Initializes gamma to ones and beta to zeros.
    ///
    /// # Arguments
    /// * `d` - Dimension to normalize over
    ///
    /// # Returns
    /// * `Self` - LayerNorm instance
    pub fn new(d: usize) -> Self {
        Self {
            g: Array1::ones(d),
            b: Array1::zeros(d),
        }
    }

    /// Applies layer normalization.
    ///
    /// # Details
    /// Normalizes input and applies learned scale and shift.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    ///
    /// # Returns
    /// * `Array2<f32>` - Normalized output
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        layer_norm(x, &self.g, &self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Tests layer norm creation.
    #[test]
    fn test_layer_norm_new() {
        let ln = LayerNorm::new(4);
        assert_eq!(ln.g.shape(), &[4]);
        assert_eq!(ln.b.shape(), &[4]);
        assert!(ln.g.iter().all(|&x| x == 1.0));
        assert!(ln.b.iter().all(|&x| x == 0.0));
    }

    /// Tests layer norm forward output shape.
    #[test]
    fn test_layer_norm_forward_shape() {
        let ln = LayerNorm::new(3);
        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = ln.forward(&x);
        assert_eq!(result.shape(), &[2, 3]);
    }

    /// Tests layer norm normalizes mean.
    #[test]
    fn test_layer_norm_mean() {
        let ln = LayerNorm::new(4);
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0]]);
        let result = ln.forward(&x);
        let mean: f32 = result.row(0).sum() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    /// Tests layer norm clone.
    #[test]
    fn test_layer_norm_clone() {
        let ln1 = LayerNorm::new(5);
        let ln2 = ln1.clone();
        assert_eq!(ln1.g.shape(), ln2.g.shape());
        assert_eq!(ln1.b.shape(), ln2.b.shape());
    }

    /// Tests layer norm with custom gamma.
    #[test]
    fn test_layer_norm_custom_gamma() {
        let mut ln = LayerNorm::new(2);
        ln.g = Array1::from_vec(vec![2.0, 2.0]);
        let x = arr2(&[[1.0, 3.0]]);
        let result1 = LayerNorm::new(2).forward(&x);
        let result2 = ln.forward(&x);
        for (a, b) in result1.iter().zip(result2.iter()) {
            assert!((b - a * 2.0).abs() < 1e-5);
        }
    }

    /// Tests layer norm with custom beta.
    #[test]
    fn test_layer_norm_custom_beta() {
        let mut ln = LayerNorm::new(3);
        ln.b = Array1::from_vec(vec![5.0, 5.0, 5.0]);
        let x = arr2(&[[1.0, 2.0, 3.0]]);
        let result = ln.forward(&x);
        let mean: f32 = result.row(0).sum() / 3.0;
        assert!((mean - 5.0).abs() < 1e-5);
    }

    /// Tests layer norm with single row.
    #[test]
    fn test_layer_norm_single_row() {
        let ln = LayerNorm::new(3);
        let x = arr2(&[[10.0, 20.0, 30.0]]);
        let result = ln.forward(&x);
        assert_eq!(result.shape(), &[1, 3]);
        assert!(result.iter().all(|x| x.is_finite()));
    }

    /// Tests layer norm with constant input.
    #[test]
    fn test_layer_norm_constant_input() {
        let ln = LayerNorm::new(3);
        let x = arr2(&[[5.0, 5.0, 5.0]]);
        let result = ln.forward(&x);
        assert!(result.iter().all(|x| x.is_finite()));
    }
}
