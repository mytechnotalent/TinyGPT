/*
 * @file sampling.rs
 * @brief Probability sampling utilities
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

//! FILE: sampling.rs
//!
//! DESCRIPTION:
//! Probability Sampling Utilities for TinyGPT.
//!
//! BRIEF:
//! Provides sampling function for token generation.
//! Implements categorical sampling from probability distribution.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use ndarray::Array1;
use rand::{Rng, rngs::ThreadRng};

/// Samples index from probability distribution.
///
/// # Details
/// Performs categorical sampling using cumulative distribution.
/// Used for next token selection during text generation.
///
/// # Arguments
/// * `probs` - Probability distribution array
/// * `rng` - Random number generator
///
/// # Returns
/// * `usize` - Sampled index
pub fn sample(probs: &Array1<f32>, rng: &mut ThreadRng) -> usize {
    let (r, mut c) = (rng.random::<f32>(), 0.0);
    probs
        .iter()
        .position(|&p| {
            c += p;
            r < c
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    /// Tests sample returns valid index.
    #[test]
    fn test_sample_valid_index() {
        let probs = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let mut rng = rng();
        for _ in 0..100 {
            let idx = sample(&probs, &mut rng);
            assert!(idx < 4);
        }
    }

    /// Tests sample with deterministic distribution.
    #[test]
    fn test_sample_deterministic() {
        let probs = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let mut rng = rng();
        for _ in 0..10 {
            let idx = sample(&probs, &mut rng);
            assert_eq!(idx, 2);
        }
    }

    /// Tests sample respects distribution.
    #[test]
    fn test_sample_distribution() {
        let probs = Array1::from_vec(vec![0.9, 0.1, 0.0, 0.0]);
        let mut rng = rng();
        let mut counts = [0; 4];
        for _ in 0..1000 {
            counts[sample(&probs, &mut rng)] += 1;
        }
        assert!(counts[0] > counts[1]);
        assert_eq!(counts[2], 0);
        assert_eq!(counts[3], 0);
    }

    /// Tests sample with single element.
    #[test]
    fn test_sample_single() {
        let probs = Array1::from_vec(vec![1.0]);
        let mut rng = rng();
        assert_eq!(sample(&probs, &mut rng), 0);
    }

    /// Tests sample fallback behavior.
    #[test]
    fn test_sample_edge_case() {
        let probs = Array1::from_vec(vec![0.5, 0.5]);
        let mut rng = rng();
        let idx = sample(&probs, &mut rng);
        assert!(idx < 2);
    }
}
