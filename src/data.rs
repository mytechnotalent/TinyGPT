/*
 * @file data.rs
 * @brief Data loading utilities
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

//! FILE: data.rs
//!
//! DESCRIPTION:
//! Data Loading Utilities for TinyGPT.
//!
//! BRIEF:
//! Provides batch generation for training.
//! Handles random sampling of training sequences.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::config::CFG;
use ndarray::Array2;
use rand::{Rng, rngs::ThreadRng};

/// Data loader for batch generation.
///
/// # Details
/// Manages tokenized data and provides random batches.
/// Used for training loop iteration.
///
/// # Fields
/// * `data` - Reference to tokenized training data
pub struct DataLoader<'a> {
    pub(crate) data: &'a [usize],
}

impl<'a> DataLoader<'a> {
    /// Creates new data loader.
    ///
    /// # Details
    /// Initializes loader with reference to training data.
    ///
    /// # Arguments
    /// * `data` - Tokenized training data
    ///
    /// # Returns
    /// * `Self` - DataLoader instance
    pub fn new(data: &'a [usize]) -> Self {
        Self { data }
    }

    /// Generates random training batch.
    ///
    /// # Details
    /// Samples random sequences from training data.
    /// Returns input and target arrays offset by one position.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// * `(Array2<usize>, Array2<usize>)` - Input and target batches
    pub fn get_batch(&self, rng: &mut ThreadRng) -> (Array2<usize>, Array2<usize>) {
        let (mut xb, mut yb) = (
            Array2::zeros((CFG.batch_size, CFG.block_size)),
            Array2::zeros((CFG.batch_size, CFG.block_size)),
        );
        for i in 0..CFG.batch_size {
            let s = rng.random_range(0..self.data.len() - CFG.block_size - 1);
            for j in 0..CFG.block_size {
                xb[[i, j]] = self.data[s + j];
                yb[[i, j]] = self.data[s + j + 1];
            }
        }
        (xb, yb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    /// Tests data loader creation.
    #[test]
    fn test_data_loader_new() {
        let data = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ];
        let loader = DataLoader::new(&data);
        assert_eq!(loader.data.len(), 20);
    }

    /// Tests get_batch output shapes.
    #[test]
    fn test_data_loader_batch_shape() {
        let data: Vec<usize> = (0..100).collect();
        let loader = DataLoader::new(&data);
        let mut rng = rng();
        let (xb, yb) = loader.get_batch(&mut rng);
        assert_eq!(xb.shape(), &[CFG.batch_size, CFG.block_size]);
        assert_eq!(yb.shape(), &[CFG.batch_size, CFG.block_size]);
    }

    /// Tests target is offset by 1.
    #[test]
    fn test_data_loader_offset() {
        let data: Vec<usize> = (0..100).collect();
        let loader = DataLoader::new(&data);
        let mut rng = rng();
        let (xb, yb) = loader.get_batch(&mut rng);
        // At least one pair should show x + 1 = y pattern
        // (since data is sequential)
        let mut found_offset = false;
        for i in 0..CFG.batch_size {
            for j in 0..CFG.block_size {
                if xb[[i, j]] + 1 == yb[[i, j]] {
                    found_offset = true;
                    break;
                }
            }
        }
        assert!(found_offset);
    }

    /// Tests batch contains valid indices.
    #[test]
    fn test_data_loader_valid_indices() {
        let data: Vec<usize> = (0..50).collect();
        let loader = DataLoader::new(&data);
        let mut rng = rng();
        let (xb, yb) = loader.get_batch(&mut rng);
        for val in xb.iter() {
            assert!(*val < 50);
        }
        for val in yb.iter() {
            assert!(*val < 50);
        }
    }

    /// Tests multiple batches are different.
    #[test]
    fn test_data_loader_random() {
        let data: Vec<usize> = (0..1000).collect();
        let loader = DataLoader::new(&data);
        let mut rng = rng();
        let (xb1, _) = loader.get_batch(&mut rng);
        let (xb2, _) = loader.get_batch(&mut rng);
        // Batches should likely be different
        let same = xb1.iter().zip(xb2.iter()).filter(|(a, b)| a == b).count();
        assert!(same < xb1.len());
    }
}
