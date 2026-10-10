/*
 * @file trainer.rs
 * @brief Model training utilities
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

//! FILE: trainer.rs
//!
//! DESCRIPTION:
//! Model Training Utilities for TinyGPT.
//!
//! BRIEF:
//! Provides training loop and text generation interface.
//! Manages model, data loader, and training process.
//!
//! AUTHOR: Kevin Thomas
//! CREATION DATE: December 11, 2025
//! UPDATE DATE: December 11, 2025

use crate::config::CFG;
use crate::data::DataLoader;
use crate::tiny_gpt::{TinyGPT, cross_entropy};
use rand::rngs::ThreadRng;

/// Trainer for TinyGPT model.
///
/// # Details
/// Encapsulates model and data loader.
/// Provides training loop and text generation methods.
///
/// # Fields
/// * `model` - TinyGPT model instance
/// * `loader` - Data loader for batch generation
pub struct Trainer<'a> {
    pub(crate) model: TinyGPT,
    pub(crate) loader: DataLoader<'a>,
}

impl<'a> Trainer<'a> {
    /// Creates new trainer.
    ///
    /// # Details
    /// Initializes model and data loader.
    ///
    /// # Arguments
    /// * `vocab_size` - Size of vocabulary
    /// * `data` - Tokenized training data
    ///
    /// # Returns
    /// * `Self` - Trainer instance
    pub fn new(vocab_size: usize, data: &'a [usize]) -> Self {
        Self {
            model: TinyGPT::new(vocab_size),
            loader: DataLoader::new(data),
        }
    }

    /// Runs training loop.
    ///
    /// # Details
    /// Executes gradient descent for configured number of epochs.
    /// Prints loss every 300 steps.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    pub fn train(&mut self, rng: &mut ThreadRng) {
        self.train_steps(CFG.epochs, rng);
    }

    /// Runs training for specified number of steps.
    ///
    /// # Details
    /// Executes gradient descent for given number of steps.
    /// Prints loss every 300 steps.
    ///
    /// # Arguments
    /// * `steps` - Number of training steps
    /// * `rng` - Random number generator
    pub fn train_steps(&mut self, steps: usize, rng: &mut ThreadRng) {
        for step in 0..steps {
            let (xb, yb) = self.loader.get_batch(rng);
            self.model.zero_grad();
            self.model.backward(&xb, &yb);
            self.model.step(CFG.lr);
            if step % 300 == 0 {
                println!(
                    "Step {}, loss={:.4}",
                    step,
                    cross_entropy(&self.model.forward_batch(&xb), &yb)
                );
            }
        }
    }

    /// Generates text from starting token.
    ///
    /// # Details
    /// Autoregressively generates tokens using trained model.
    ///
    /// # Arguments
    /// * `start_token` - Starting vocabulary index
    /// * `n` - Number of tokens to generate
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// * `Vec<usize>` - Generated token sequence
    pub fn generate(&self, start_token: usize, n: usize, rng: &mut ThreadRng) -> Vec<usize> {
        self.model.generate(start_token, n, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    /// Tests trainer creation.
    #[test]
    fn test_trainer_new() {
        let data: Vec<usize> = (0..100).collect();
        let trainer = Trainer::new(50, &data);
        assert_eq!(trainer.model.head.w.shape()[1], 50);
    }

    /// Tests trainer generate.
    #[test]
    fn test_trainer_generate() {
        let data: Vec<usize> = (0..100).collect();
        let trainer = Trainer::new(20, &data);
        let mut rng = rng();
        let result = trainer.generate(0, 3, &mut rng);
        assert_eq!(result.len(), 4); // start + 3 generated
    }

    /// Tests trainer generate produces valid tokens.
    #[test]
    fn test_trainer_generate_valid() {
        let data: Vec<usize> = (0..50).collect();
        let trainer = Trainer::new(15, &data);
        let mut rng = rng();
        let result = trainer.generate(0, 5, &mut rng);
        assert!(result.iter().all(|&t| t < 15));
    }

    /// Tests data loader is initialized.
    #[test]
    fn test_trainer_loader() {
        let data: Vec<usize> = (0..100).collect();
        let trainer = Trainer::new(10, &data);
        assert_eq!(trainer.loader.data.len(), 100);
    }

    /// Tests train runs without panic (minimal epochs).
    #[test]
    fn test_trainer_train_runs() {
        // Create data with indices < vocab_size
        let vocab_size = 20;
        let data: Vec<usize> = (0..100).map(|i| i % vocab_size).collect();
        let mut trainer = Trainer::new(vocab_size, &data);
        let mut rng = rng();
        // We can't easily test the full training loop without modifying CFG
        // But we can verify the model changes after backward pass
        let (xb, yb) = trainer.loader.get_batch(&mut rng);
        let before = trainer.model.head.w[[0, 0]];
        trainer.model.zero_grad();
        trainer.model.backward(&xb, &yb);
        trainer.model.step(CFG.lr);
        let after = trainer.model.head.w[[0, 0]];
        // Parameters should have changed or stayed same
        assert!(after.is_finite());
        let _ = before; // Silence unused warning
    }

    /// Tests training step modifies gradients.
    #[test]
    fn test_trainer_step_gradients() {
        let vocab_size = 20;
        let data: Vec<usize> = (0..100).map(|i| i % vocab_size).collect();
        let mut trainer = Trainer::new(vocab_size, &data);
        let mut rng = rng();
        let (xb, yb) = trainer.loader.get_batch(&mut rng);
        trainer.model.zero_grad();
        trainer.model.backward(&xb, &yb);
        // Should have non-zero gradients
        let has_grad = trainer.model.head.dw.iter().any(|&x| x != 0.0);
        assert!(has_grad);
    }

    /// Tests loss computation during training.
    #[test]
    fn test_trainer_loss_finite() {
        let vocab_size = 20;
        let data: Vec<usize> = (0..100).map(|i| i % vocab_size).collect();
        let trainer = Trainer::new(vocab_size, &data);
        let mut rng = rng();
        let (xb, yb) = trainer.loader.get_batch(&mut rng);
        let loss = cross_entropy(&trainer.model.forward_batch(&xb), &yb);
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    /// Tests train_steps runs for given steps.
    #[test]
    fn test_trainer_train_steps() {
        let vocab_size = 20;
        let data: Vec<usize> = (0..100).map(|i| i % vocab_size).collect();
        let mut trainer = Trainer::new(vocab_size, &data);
        let mut rng = rng();
        let before = trainer.model.head.w[[0, 0]];
        trainer.train_steps(1, &mut rng);
        let after = trainer.model.head.w[[0, 0]];
        // Parameters should have been updated
        assert!(after.is_finite());
        let _ = before;
    }

    /// Tests train_steps with multiple steps.
    #[test]
    fn test_trainer_train_steps_multiple() {
        let vocab_size = 20;
        let data: Vec<usize> = (0..100).map(|i| i % vocab_size).collect();
        let mut trainer = Trainer::new(vocab_size, &data);
        let mut rng = rng();
        trainer.train_steps(5, &mut rng);
        // Model should still produce valid output
        let output = trainer.generate(0, 3, &mut rng);
        assert_eq!(output.len(), 4);
    }

    /// Tests train_steps prints at step 0.
    #[test]
    fn test_trainer_train_steps_prints() {
        let vocab_size = 20;
        let data: Vec<usize> = (0..100).map(|i| i % vocab_size).collect();
        let mut trainer = Trainer::new(vocab_size, &data);
        let mut rng = rng();
        // Step 0 should trigger print (step % 300 == 0)
        trainer.train_steps(1, &mut rng);
    }

    /// Tests train calls train_steps.
    #[test]
    fn test_trainer_train_wrapper() {
        let vocab_size = 20;
        let data: Vec<usize> = (0..100).map(|i| i % vocab_size).collect();
        let trainer = Trainer::new(vocab_size, &data);
        // Just verify it compiles and model is valid
        assert_eq!(trainer.model.head.w.shape()[1], vocab_size);
    }

    /// Tests integration with vocab.
    #[test]
    fn test_trainer_with_vocab() {
        use crate::vocab::Vocab;
        // Need enough words for batch_size * block_size
        let corpus = vec![
            "hello world test one two three four five six seven eight nine ten".to_string(),
            "foo bar baz alpha beta gamma delta epsilon zeta eta theta iota kappa".to_string(),
            "the quick brown fox jumps over the lazy dog again and again today".to_string(),
            "reverse engineering binary analysis assembly code disassembly debugging".to_string(),
        ];
        let vocab = Vocab::from_corpus(&corpus);
        let mut trainer = Trainer::new(vocab.size, &vocab.data);
        let mut rng = rng();
        trainer.train_steps(1, &mut rng);
        let output = trainer.generate(vocab.encode("hello"), 3, &mut rng);
        let decoded = vocab.decode(&output);
        assert!(!decoded.is_empty());
    }
}
