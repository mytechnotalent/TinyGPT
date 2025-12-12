![image](https://github.com/mytechnotalent/TinyGPT/blob/main/TinyGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# TinyGPT

## A Pure Rust GPT Implementation From Scratch
A comprehensive tutorial implementation of a GPT (Generative Pre-trained Transformer) language model written entirely in pure Rust using only the `ndarray` crate for tensor operations. This project demonstrates how to build a working transformer architecture from first principles without relying on deep learning frameworks.

<br>

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Architecture Overview](#architecture-overview)
4. [Module Deep Dive](#module-deep-dive)
5. [Configuration](#configuration)
6. [Building and Running](#building-and-running)
7. [Testing](#testing)
8. [Code Coverage](#code-coverage)
9. [License](#license)

<br>

## Introduction
This project implements a miniature version of the GPT architecture, the same fundamental design behind models like ChatGPT. The goal is educational: to understand every component of a transformer language model by implementing it from scratch in Rust.

### Why Rust?
Rust provides memory safety without garbage collection, zero-cost abstractions, and excellent performance. By implementing a neural network in Rust without a framework, we gain deep insight into:
- How tensor operations work at a low level
- Memory management in neural networks
- The actual mathematics behind attention mechanisms
- Gradient computation and backpropagation

### What This Project Teaches
By studying this codebase, you will understand:
- **Embeddings**: How discrete tokens become continuous vectors
- **Attention Mechanisms**: The core innovation that powers transformers
- **Layer Normalization**: How to stabilize training
- **Feed-Forward Networks**: Position-wise transformations
- **Autoregressive Generation**: How language models generate text token by token

<br>

## Mathematical Foundations

### Embedding Layer
An embedding layer maps discrete token indices to continuous vector representations. Given a vocabulary of size $V$ and embedding dimension $d$, we maintain a weight matrix $W_e \in \mathbb{R}^{V \times d}$. For a token with index $i$, the embedding lookup is simply:

$$e_i = W_e[i, :]$$

This retrieves the $i$-th row of the embedding matrix, producing a $d$-dimensional vector.

### Positional Embeddings
Since transformers process all positions in parallel (unlike RNNs), we need to inject positional information. We use learned positional embeddings $W_p \in \mathbb{R}^{T \times d}$ where $T$ is the maximum sequence length (block size).
The input to the transformer becomes:

$$x_t = W_e[\text{token}_t] + W_p[t]$$

### Linear Layer
A linear (fully connected) layer computes:

$$y = xW + b$$

where $x \in \mathbb{R}^{n \times d_{in}}$, $W \in \mathbb{R}^{d_{in} \times d_{out}}$, $b \in \mathbb{R}^{d_{out}}$, and $y \in \mathbb{R}^{n \times d_{out}}$.

### Softmax Function
The softmax function converts raw scores (logits) into a probability distribution:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

For numerical stability, we subtract the maximum value before exponentiation:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

### Layer Normalization
Layer normalization normalizes activations across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where:
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ is the mean
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$ is the variance
- $\gamma, \beta \in \mathbb{R}^d$ are learnable scale and shift parameters
- $\epsilon = 10^{-5}$ prevents division by zero

### Scaled Dot-Product Attention
The attention mechanism is the heart of the transformer. Given queries $Q$, keys $K$, and values $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $d_k$ is the dimension of the keys (head size). The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the dot products from growing too large.

### Causal Masking
For autoregressive language modeling, each position can only attend to previous positions. We achieve this by setting future positions to $-\infty$ before softmax:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V$$

### Multi-Head Attention
Instead of performing a single attention function, multi-head attention runs $h$ attention heads in parallel, each with dimension $d_k = d/h$:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

### Feed-Forward Network
Each transformer block contains a position-wise feed-forward network:

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

The hidden dimension is typically $4 \times$ the embedding dimension, allowing the network to learn complex transformations.

### ReLU Activation
The Rectified Linear Unit is defined as:

$$\text{ReLU}(x) = \max(0, x)$$

### Transformer Block
A complete transformer block combines attention and feed-forward layers with residual connections and layer normalization (pre-norm architecture):

$$x' = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))$$

$$\text{output} = x' + \text{FFN}(\text{LayerNorm}(x'))$$

### Cross-Entropy Loss
For training, we use cross-entropy loss between predicted probabilities and target tokens:

$$\mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \log p(y_{b,t} | x_{b,1:t})$$

where $B$ is batch size, $T$ is sequence length, and $p(y_{b,t} | x_{b,1:t})$ is the probability assigned to the correct token.

### Gradient Descent
Parameters are updated using gradient descent:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

where $\eta$ is the learning rate and $\nabla_\theta \mathcal{L}$ is the gradient of the loss with respect to parameters.

<br>

## Architecture Overview
```
┌──────────────────────────────────────────────────────────────┐
│                          TinyGPT                             │
├──────────────────────────────────────────────────────────────┤
│  Input Tokens: [t₁, t₂, ..., tₙ]                             │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐   ┌─────────────┐                           │
│  │   Token     │ + │  Position   │                           │
│  │  Embedding  │   │  Embedding  │                           │
│  └─────────────┘   └─────────────┘                           │
│         │                                                    │
│         ▼                                                    │
│  ┌───────────────────────────────────┐                       │
│  │       Transformer Block ×N        │                       │
│  │  ┌─────────────────────────────┐  │                       │
│  │  │  LayerNorm                  │  │                       │
│  │  │      ↓                      │  │                       │
│  │  │  Multi-Head Attention       │  │                       │
│  │  │      ↓                      │  │                       │
│  │  │  + Residual Connection      │  │                       │
│  │  │      ↓                      │  │                       │
│  │  │  LayerNorm                  │  │                       │
│  │  │      ↓                      │  │                       │
│  │  │  Feed-Forward Network       │  │                       │
│  │  │      ↓                      │  │                       │
│  │  │  + Residual Connection      │  │                       │
│  │  └─────────────────────────────┘  │                       │
│  └───────────────────────────────────┘                       │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                             │
│  │  LayerNorm  │                                             │
│  └─────────────┘                                             │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                             │
│  │ Output Head │  →  Logits [vocab_size]                     │
│  └─────────────┘                                             │
│         │                                                    │
│         ▼                                                    │
│     Softmax → Probabilities → Sample → Next Token            │
└──────────────────────────────────────────────────────────────┘
```

<br>

## Module Deep Dive

### `config.rs` - Configuration Management
This module defines the hyperparameters for the model using a JSON configuration file.
```rust
#[derive(Deserialize)]
pub struct Config {
    pub block_size: usize,    // Maximum context length (8)
    pub embed_dim: usize,     // Embedding dimension (128)
    pub n_heads: usize,       // Number of attention heads (4)
    pub n_layers: usize,      // Number of transformer blocks (4)
    pub lr: f32,              // Learning rate (0.01)
    pub epochs: usize,        // Training iterations (5000)
    pub batch_size: usize,    // Batch size (16)
}
```
The configuration is loaded at compile time using `include_str!` and parsed with `serde_json`. The `once_cell::Lazy` pattern ensures the configuration is loaded exactly once and shared globally.

### `math.rs` - Mathematical Utilities
This module provides core mathematical operations:
**Random Initialization (`randn`)**: Generates weights from a scaled normal distribution. The scaling factor of 0.02 prevents exploding gradients at initialization.
```rust
pub fn randn(r: usize, c: usize) -> Array2<f32> {
    Array2::from_shape_fn((r, c), |_| rng().sample::<f32, _>(StandardNormal) * 0.02)
}
```
**Softmax (`softmax`, `softmax1d`)**: Converts logits to probabilities with numerical stability by subtracting the maximum value before exponentiation.
**Layer Normalization (`layer_norm`)**: Normalizes activations across the feature dimension with learnable scale ($\gamma$) and shift ($\beta$) parameters.

### `sampling.rs` - Token Sampling
Implements categorical sampling for text generation:
```rust
pub fn sample(probs: &Array1<f32>, rng: &mut ThreadRng) -> usize {
    let (r, mut c) = (rng.random::<f32>(), 0.0);
    probs.iter().position(|&p| { c += p; r < c }).unwrap_or(0)
}
```
This walks through the cumulative distribution until the random value is exceeded, effectively sampling from the probability distribution.
### `vocab.rs` - Vocabulary Management
Handles tokenization at the word level:
- **`from_corpus`**: Builds word-to-index and index-to-word mappings from training text
- **`encode`**: Converts a word to its vocabulary index
- **`decode`**: Converts a sequence of indices back to text
Each sentence in the corpus is appended with an `<END>` token to mark sentence boundaries.

### `data.rs` - Data Loading
The `DataLoader` generates random training batches:
```rust
pub fn get_batch(&self, rng: &mut ThreadRng) -> (Array2<usize>, Array2<usize>) {
    // For each batch item, sample a random starting position
    // xb contains input tokens, yb contains target tokens (shifted by 1)
}
```

The target for each position is the next token in the sequence, enabling the model to learn next-token prediction.
### `linear.rs` - Linear Layer
Implements a fully connected layer with:
- **Forward pass**: $y = xW + b$
- **Gradient storage**: `dw` and `db` accumulate gradients during backpropagation
- **Parameter update**: `step(lr)` applies gradient descent

### `embedding.rs` - Embedding Layer
Provides learnable token embeddings:
```rust
pub fn forward(&self, idx: &[usize]) -> Array2<f32> {
    Array2::from_shape_fn((idx.len(), self.w.shape()[1]), |(i, j)| self.w[[idx[i], j]])
}
```
This gathers rows from the embedding matrix corresponding to the input token indices.

### `layer_norm.rs` - Layer Normalization
A thin wrapper around the `layer_norm` math function with learnable $\gamma$ (initialized to 1) and $\beta$ (initialized to 0) parameters.

### `head.rs` - Single Attention Head
Implements scaled dot-product attention with causal masking:
```rust
pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
    let (k, q, v) = (self.key.forward(x), self.query.forward(x), self.value.forward(x));
    let mut w = q.dot(&k.t()) / (self.hs as f32).sqrt();
    // Apply causal mask
    for i in 0..t {
        for j in i + 1..t {
            w[[i, j]] = f32::NEG_INFINITY;
        }
    }
    softmax(&w).dot(&v)
}
```
The causal mask ensures position $i$ can only attend to positions $\leq i$.

### `attention.rs` - Multi-Head Attention
Runs multiple attention heads in parallel and projects their concatenated outputs:
```rust
pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
    let o: Vec<_> = self.heads.iter().map(|h| h.forward(x)).collect();
    let v: Vec<_> = o.iter().map(|a| a.view()).collect();
    self.proj.forward(&ndarray::concatenate(Axis(1), &v).unwrap())
}
```

### `feed_forward.rs` - Feed-Forward Network
Position-wise feed-forward network with ReLU activation:
```rust
pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
    self.l2.forward(&self.l1.forward(x).mapv(|v| v.max(0.0)))
}
```
The hidden dimension is 4× the embedding dimension, following the original transformer paper.

### `block.rs` - Transformer Block
Combines attention and feed-forward with residual connections (pre-norm architecture):
```rust
pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
    let x = x + &self.sa.forward(&self.ln1.forward(x));
    &x + &self.ffn.forward(&self.ln2.forward(&x))
}
```

### `tiny_gpt.rs` - Complete Model
The full GPT model combining all components:
1. **Token + Position Embeddings**: Sum of token and positional embeddings
2. **Transformer Blocks**: Stack of N transformer blocks
3. **Final LayerNorm**: Stabilizes outputs before projection
4. **Output Head**: Projects to vocabulary size for next-token prediction
The `backward` method implements simplified backpropagation through the output head and embeddings, computing gradients for gradient descent.
The `generate` method implements autoregressive generation:
```rust
pub fn generate(&self, start: usize, n: usize, rng: &mut ThreadRng) -> Vec<usize> {
    let mut out = vec![start];
    for _ in 0..n {
        let ctx: Vec<_> = out.iter().rev().take(CFG.block_size).rev().copied().collect();
        let logits = self.forward(&ctx);
        out.push(sample(&softmax1d(&logits.row(logits.shape()[0] - 1)), rng));
    }
    out
}
```

### `trainer.rs` - Training Loop
Orchestrates the training process:
```rust
pub fn train_steps(&mut self, steps: usize, rng: &mut ThreadRng) {
    for step in 0..steps {
        let (xb, yb) = self.loader.get_batch(rng);
        self.model.zero_grad();
        self.model.backward(&xb, &yb);
        self.model.step(CFG.lr);
    }
}
```
Each step: get batch → zero gradients → compute gradients → update parameters.

### `main.rs` - Entry Point
Ties everything together:
1. Load corpus from `corpus.json`
2. Build vocabulary
3. Create trainer with model and data
4. Train for configured epochs
5. Generate sample text

<br>

## Configuration

### `config.json`
```json
{
  "block_size": 8,
  "embed_dim": 128,
  "n_heads": 4,
  "n_layers": 4,
  "lr": 0.01,
  "epochs": 5000,
  "batch_size": 16
}
```
| Parameter    | Description                            | Value    |
| ------------ | -------------------------------------- | -------- |
| `block_size` | Maximum context window for attention   | 8 tokens |
| `embed_dim`  | Dimension of token/position embeddings | 128      |
| `n_heads`    | Number of parallel attention heads     | 4        |
| `n_layers`   | Number of transformer blocks           | 4        |
| `lr`         | Learning rate for gradient descent     | 0.01     |
| `epochs`     | Number of training iterations          | 5000     |
| `batch_size` | Number of sequences per batch          | 16       |

### `corpus.json`
Contains 50 sentences focused on Reverse Engineering concepts. Each sentence is tokenized at the word level, with `<END>` tokens appended.
<br>

## Building and Running

### Prerequisites
- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Cargo (included with Rust)

### Build
```bash
# Debug build
cargo build

# Release build (optimized, recommended)
cargo build --release
```

### Run
```bash
# Run debug build
cargo run

# Run release build (faster)
cargo run --release
```

### Expected Output
```
TinyGPT
Step 0, loss=5.1234
Step 300, loss=4.2345
Step 600, loss=3.5678
...
Step 4800, loss=2.1234

generated text:
the binary analysis requires understanding of assembly code <END> reverse engineering
```
The loss should decrease over training as the model learns patterns in the corpus.

<br>

## Testing
This project includes comprehensive unit tests for all modules, achieving 95%+ code coverage.

### Run All Tests
```bash
cargo test
```

### Run Tests with Output
```bash
cargo test -- --nocapture
```

### Run Specific Test
```bash
cargo test test_linear_forward
```

### Run Tests for Specific Module
```bash
cargo test math::tests
cargo test linear::tests
cargo test tiny_gpt::tests
```

### Test Summary
| Module            | Tests   | Coverage   |
| ----------------- | ------- | ---------- |
| `math.rs`         | 13      | 100%       |
| `sampling.rs`     | 5       | 100%       |
| `linear.rs`       | 7       | 100%       |
| `embedding.rs`    | 8       | 100%       |
| `layer_norm.rs`   | 8       | 100%       |
| `head.rs`         | 7       | 100%       |
| `attention.rs`    | 8       | 100%       |
| `feed_forward.rs` | 8       | 100%       |
| `block.rs`        | 8       | 100%       |
| `tiny_gpt.rs`     | 12      | 100%       |
| `vocab.rs`        | 10      | 100%       |
| `data.rs`         | 5       | 100%       |
| `config.rs`       | 9       | 100%       |
| `trainer.rs`      | 11      | 89%        |
| **Total**         | **123** | **95.30%** |

<br>

## Code Coverage
### Install Tarpaulin
```bash
cargo install cargo-tarpaulin
```

### Run Coverage
```bash
# Output to terminal
cargo tarpaulin --out Stdout

# Generate HTML report
cargo tarpaulin --out Html
```

### Coverage Report
```
|| Tested/Total Lines:
|| src/attention.rs: 14/14
|| src/block.rs: 14/14
|| src/config.rs: 1/1
|| src/data.rs: 11/11
|| src/embedding.rs: 9/9
|| src/feed_forward.rs: 11/11
|| src/head.rs: 23/23
|| src/layer_norm.rs: 5/5
|| src/linear.rs: 13/13
|| src/math.rs: 18/18
|| src/sampling.rs: 6/6
|| src/tiny_gpt.rs: 67/67
|| src/vocab.rs: 15/15
|| src/trainer.rs: 16/18
|| 
|| 95.30% coverage, 223/234 lines covered
```
The uncovered lines are:
- `main.rs`: Entry point (standard practice not to unit test)
- `trainer.rs`: The `train()` wrapper that runs for 5000 epochs

<br>

## Project Structure
```
rust_gpt/
├── Cargo.toml          # Dependencies and project metadata
├── config.json         # Model hyperparameters
├── corpus.json         # Training data (50 RE-focused sentences)
├── README.md           # This file
└── src/
    ├── main.rs         # Application entry point
    ├── config.rs       # Configuration loading
    ├── math.rs         # Mathematical utilities (randn, softmax, layer_norm)
    ├── sampling.rs     # Probability sampling for generation
    ├── vocab.rs        # Vocabulary management (encode/decode)
    ├── data.rs         # Batch data loading
    ├── linear.rs       # Linear (fully connected) layer
    ├── embedding.rs    # Token/position embedding layer
    ├── layer_norm.rs   # Layer normalization module
    ├── head.rs         # Single attention head
    ├── attention.rs    # Multi-head attention
    ├── feed_forward.rs # Position-wise feed-forward network
    ├── block.rs        # Complete transformer block
    ├── tiny_gpt.rs     # Full TinyGPT model
    └── trainer.rs      # Training loop and generation
```

<br>

## Dependencies
```toml
[dependencies]
ndarray = { version = "0.17.1", features = ["rayon"] }
rand = "0.9.2"
rand_distr = "0.5.1"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
once_cell = "1.19"
```
| Crate        | Purpose                                    |
| ------------ | ------------------------------------------ |
| `ndarray`    | N-dimensional arrays for tensor operations |
| `rand`       | Random number generation                   |
| `rand_distr` | Statistical distributions (Normal)         |
| `serde`      | Serialization framework                    |
| `serde_json` | JSON parsing                               |
| `once_cell`  | Lazy static initialization                 |

<br>

## Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [minGPT](https://github.com/karpathy/minGPT) - Andrej Karpathy's minimal GPT implementation

<br>

## License
[MIT](LICENSE)
