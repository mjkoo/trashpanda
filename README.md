# TrashPanda 🦝

[![Crates.io](https://img.shields.io/crates/v/trashpanda)](https://crates.io/crates/trashpanda)
[![docs.rs](https://img.shields.io/docsrs/trashpanda)](https://docs.rs/trashpanda)
[![CI](https://github.com/mjkoo/trashpanda/actions/workflows/test.yml/badge.svg)](https://github.com/mjkoo/trashpanda/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/mjkoo/trashpanda/branch/main/graph/badge.svg)](https://codecov.io/gh/mjkoo/trashpanda)
[![GitHub License](https://img.shields.io/github/license/mjkoo/trashpanda)](LICENSE-MIT)


A high-performance Rust implementation of contextual multi-armed bandits, inspired by Python's [MABWiser](https://github.com/fidelity/mabwiser) library.

TrashPanda provides a type-safe, memory-efficient framework for implementing multi-armed bandit algorithms with excellent performance characteristics. It offers both contextual and non-contextual bandit algorithms with a clean, idiomatic Rust API.

## Algorithms

### Non-Contextual Bandits
- **Random**: Uniform random selection (baseline)
- **Epsilon-Greedy**: Classic exploration-exploitation trade-off
- **UCB (Upper Confidence Bound)**: Optimism in the face of uncertainty
- **Thompson Sampling**: Bayesian approach with Beta distributions

### Contextual Bandits
- **LinUCB**: Linear Upper Confidence Bound for contextual problems
- **LinGreedy**: Contextual epsilon-greedy with linear reward modeling
- **LinTS**: Linear Thompson Sampling with posterior sampling

## Installation

Add TrashPanda to your `Cargo.toml`:

```toml
[dependencies]
trashpanda = "0.1.0"
```

## Quick Start

### Non-Contextual Bandit

```rust
use trashpanda::{Bandit, policies::EpsilonGreedy};
use rand::thread_rng;

// Create a bandit with three arms
let mut bandit = Bandit::new(
    vec!["red", "green", "blue"],
    EpsilonGreedy::new(0.1)
).unwrap();

// Train with historical data
let decisions = vec!["red", "blue", "red"];
let rewards = vec![1.0, 0.5, 0.8];
bandit.fit_simple(&decisions, &rewards).unwrap();

// Make a prediction
let mut rng = thread_rng();
let choice = bandit.predict_simple(&mut rng).unwrap();
println!("Recommended arm: {}", choice);

// Get expected rewards for all arms
let expectations = bandit.predict_expectations_simple(&mut rng);
for (arm, reward) in expectations {
    println!("{}: {:.3}", arm, reward);
}
```

### Contextual Bandit

```rust
use trashpanda::{Bandit, policies::LinUcb};
use rand::thread_rng;

// Create a contextual bandit
let mut bandit = Bandit::new(
    vec!["ad_1", "ad_2", "ad_3"],
    LinUcb::new(1.0, 1.0, 2)  // alpha, l2_reg, context_dim
).unwrap();

// Train with contexts and rewards
let contexts = vec![
    vec![1.0, 0.0],
    vec![0.0, 1.0],
    vec![0.5, 0.5],
];
let decisions = vec!["ad_1", "ad_2", "ad_1"];
let rewards = vec![1.0, 0.8, 0.6];

for ((context, decision), reward) in contexts.iter()
    .zip(decisions.iter())
    .zip(rewards.iter()) 
{
    bandit.partial_fit(decision, context, *reward).unwrap();
}

// Make predictions with new context
let mut rng = thread_rng();
let new_context = vec![0.7, 0.3];
let choice = bandit.predict(&new_context, &mut rng).unwrap();
println!("Recommended ad: {}", choice);
```

### Batch Operations

```rust
use trashpanda::Bandit;
use trashpanda::policies::EpsilonGreedy;
use rand::thread_rng;

let mut bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.1).unwrap();

// Batch training
bandit.fit_simple(&[1, 2, 1, 3], &[1.0, 0.5, 0.8, 0.9]).unwrap();

// Batch predictions
let mut rng = thread_rng();
let predictions = bandit.predict_batch_simple(10, &mut rng).unwrap();
println!("Batch predictions: {:?}", predictions);
```

## Examples

Explore the examples directory for more detailed usage:

```bash
# Compare different algorithms
cargo run --example algorithm_comparison

# Demonstrate batch operations
cargo run --example batch_operations

# Contextual bandits with LinUCB
cargo run --example contextual_bandits
```
## License

This project is dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- Inspired by [MABWiser](https://github.com/fidelity/mabwiser) from Fidelity Investments