/*
 * @file main.rs
 * @brief TinyGPT - Pure Rust GPT Implementation
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

//! FILE: main.rs
//!
//! DESCRIPTION:
//! TinyGPT - Pure Rust GPT Implementation.
//!
//! BRIEF:
//! Main application entry point for TinyGPT language model.
//! Implements training loop and text generation using transformer architecture.
//! Pure Rust implementation with no PyTorch dependencies.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

mod attention;
mod block;
mod config;
mod data;
mod embedding;
mod feed_forward;
mod head;
mod layer_norm;
mod linear;
mod math;
mod sampling;
mod tiny_gpt;
mod trainer;
mod vocab;

use rand::rng;
use trainer::Trainer;
use vocab::Vocab;

/// Main application entry point.
///
/// # Details
/// Initializes the TinyGPT model and runs the training loop.
/// Loads corpus from JSON and generates text after training.
///
/// # Returns
/// * `()` - Returns nothing.
fn main() {
    println!("TinyGPT");
    let mut rng = rng();
    let corpus: Vec<String> = serde_json::from_str(include_str!("../corpus.json")).unwrap();
    let vocab = Vocab::from_corpus(&corpus);
    let mut trainer = Trainer::new(vocab.size, &vocab.data);
    trainer.train(&mut rng);
    let out = trainer.generate(vocab.encode("the"), 15, &mut rng);
    println!("\ngenerated text:\n{}", vocab.decode(&out));
}
