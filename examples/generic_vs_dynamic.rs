/// Example demonstrating the difference between generic and dynamic dispatch approaches
use std::time::Instant;

// Mock imports - replace with actual imports when integrated
mod mock {
    use std::collections::HashMap;
    use std::hash::Hash;

    pub trait Policy<A> {
        fn update(&mut self, decisions: &[A], rewards: &[f64]);
        fn select(&self, arms: &[A], rng: &mut dyn rand::RngCore) -> Option<A>;
        fn expectations(&self, arms: &[A]) -> HashMap<A, f64>;
        fn reset_arm(&mut self, arm: &A);
        fn reset(&mut self);
    }

    #[derive(Clone)]
    pub struct EpsilonGreedy<A> {
        epsilon: f64,
        _phantom: std::marker::PhantomData<A>,
    }

    impl<A> EpsilonGreedy<A> {
        pub fn new(epsilon: f64) -> Self {
            Self {
                epsilon,
                _phantom: std::marker::PhantomData,
            }
        }
    }

    impl<A: Clone + Eq + Hash> Policy<A> for EpsilonGreedy<A> {
        fn update(&mut self, _decisions: &[A], _rewards: &[f64]) {}
        fn select(&self, arms: &[A], rng: &mut dyn rand::RngCore) -> Option<A> {
            use rand::prelude::*;
            arms.choose(rng).cloned()
        }
        fn expectations(&self, arms: &[A]) -> HashMap<A, f64> {
            let prob = 1.0 / arms.len() as f64;
            arms.iter().map(|a| (a.clone(), prob)).collect()
        }
        fn reset_arm(&mut self, _arm: &A) {}
        fn reset(&mut self) {}
    }

    pub struct GenericBandit<A, P> {
        arms: Vec<A>,
        policy: P,
    }

    impl<A: Clone, P: Policy<A>> GenericBandit<A, P> {
        pub fn new(arms: Vec<A>, policy: P) -> Self {
            Self { arms, policy }
        }

        pub fn predict(&self) -> Option<A> {
            self.policy.select(&self.arms, &mut rand::rng())
        }

        pub fn update(&mut self, decisions: &[A], rewards: &[f64]) {
            self.policy.update(decisions, rewards);
        }
    }

    pub struct DynamicBandit<A> {
        arms: Vec<A>,
        policy: Box<dyn Policy<A>>,
    }

    impl<A: Clone> DynamicBandit<A> {
        pub fn new(arms: Vec<A>, policy: Box<dyn Policy<A>>) -> Self {
            Self { arms, policy }
        }

        pub fn predict(&self) -> Option<A> {
            self.policy.select(&self.arms, &mut rand::rng())
        }

        pub fn update(&mut self, decisions: &[A], rewards: &[f64]) {
            self.policy.update(decisions, rewards);
        }
    }
}

use mock::*;

fn main() {
    println!("=== Generic vs Dynamic Dispatch Comparison ===\n");

    // 1. Generic approach - zero cost abstraction
    println!("1. Generic Bandit (compile-time dispatch):");
    let mut generic_bandit = GenericBandit::new(vec![1, 2, 3], EpsilonGreedy::new(0.1));

    // The compiler knows the exact type at compile time
    // This enables inlining and optimization
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = generic_bandit.predict();
    }
    let generic_time = start.elapsed();
    println!("   - 100k predictions: {:?}", generic_time);
    println!("   - Type known at compile time");
    println!("   - Can be inlined by compiler");
    println!("   - No heap allocation for policy");

    // 2. Dynamic dispatch approach - runtime flexibility
    println!("\n2. Dynamic Bandit (runtime dispatch):");
    let mut dynamic_bandit = DynamicBandit::new(vec![1, 2, 3], Box::new(EpsilonGreedy::new(0.1)));

    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = dynamic_bandit.predict();
    }
    let dynamic_time = start.elapsed();
    println!("   - 100k predictions: {:?}", dynamic_time);
    println!("   - Type determined at runtime");
    println!("   - Virtual function call overhead");
    println!("   - Heap allocation required");

    // 3. When to use each approach
    println!("\n3. When to use each approach:");
    println!("\n   Generic Bandit:");
    println!("   ✓ Known policy type at compile time");
    println!("   ✓ Maximum performance needed");
    println!("   ✓ No runtime policy switching");
    println!("   ✓ Embedded systems or performance-critical code");

    println!("\n   Dynamic Bandit:");
    println!("   ✓ Policy type determined at runtime");
    println!("   ✓ Need to store different policies in same collection");
    println!("   ✓ Plugin systems or configuration-driven selection");
    println!("   ✓ Runtime policy switching required");

    // 4. Code size comparison
    println!("\n4. Additional considerations:");
    println!("   - Generic: Larger binary (code generated per type)");
    println!("   - Dynamic: Smaller binary (single implementation)");
    println!("   - Generic: Better error messages");
    println!("   - Dynamic: More flexible API");

    // 5. Example of flexibility with dynamic dispatch
    println!("\n5. Dynamic dispatch use case:");
    let policies: Vec<Box<dyn Policy<i32>>> = vec![
        Box::new(EpsilonGreedy::new(0.1)),
        Box::new(EpsilonGreedy::new(0.2)),
        Box::new(EpsilonGreedy::new(0.3)),
    ];
    println!("   - Can store different policies in a collection");
    println!("   - {} policies stored", policies.len());

    // Performance comparison
    if generic_time < dynamic_time {
        let speedup = dynamic_time.as_secs_f64() / generic_time.as_secs_f64();
        println!(
            "\n📊 Generic is {:.2}x faster than dynamic dispatch",
            speedup
        );
    }
}
