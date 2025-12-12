/*
 * @file config.rs
 * @brief Application configuration constants
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

//! FILE: config.rs
//!
//! DESCRIPTION:
//! TinyGPT Configuration Constants.
//!
//! BRIEF:
//! Defines configuration structure for model hyperparameters.
//! Loads configuration from external JSON file.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use once_cell::sync::Lazy;
use serde::Deserialize;

/// Configuration structure for TinyGPT hyperparameters.
///
/// # Details
/// Contains all model and training configuration parameters.
/// Loaded from config.json at runtime.
///
/// # Fields
/// * `block_size` - Maximum context length for attention
/// * `embed_dim` - Embedding dimension size
/// * `n_heads` - Number of attention heads
/// * `n_layers` - Number of transformer blocks
/// * `lr` - Learning rate for gradient descent
/// * `epochs` - Number of training iterations
/// * `batch_size` - Training batch size
#[derive(Deserialize)]
pub struct Config {
    pub block_size: usize,
    pub embed_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub lr: f32,
    pub epochs: usize,
    pub batch_size: usize,
}

/// Global configuration instance.
///
/// # Details
/// Lazily loaded from config.json file.
/// Provides thread-safe access to configuration values.
pub static CFG: Lazy<Config> =
    Lazy::new(|| serde_json::from_str(include_str!("../config.json")).unwrap());

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests config loads successfully.
    #[test]
    fn test_config_loads() {
        let _ = &CFG.block_size;
    }

    /// Tests block_size is positive.
    #[test]
    fn test_config_block_size() {
        assert!(CFG.block_size > 0);
    }

    /// Tests embed_dim is positive.
    #[test]
    fn test_config_embed_dim() {
        assert!(CFG.embed_dim > 0);
    }

    /// Tests n_heads is positive.
    #[test]
    fn test_config_n_heads() {
        assert!(CFG.n_heads > 0);
    }

    /// Tests n_layers is positive.
    #[test]
    fn test_config_n_layers() {
        assert!(CFG.n_layers > 0);
    }

    /// Tests learning rate is positive.
    #[test]
    fn test_config_lr() {
        assert!(CFG.lr > 0.0);
    }

    /// Tests epochs is positive.
    #[test]
    fn test_config_epochs() {
        assert!(CFG.epochs > 0);
    }

    /// Tests batch_size is positive.
    #[test]
    fn test_config_batch_size() {
        assert!(CFG.batch_size > 0);
    }

    /// Tests embed_dim is divisible by n_heads.
    #[test]
    fn test_config_divisible() {
        assert_eq!(CFG.embed_dim % CFG.n_heads, 0);
    }
}
