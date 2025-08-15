//! Example demonstrating the unified bandit API that handles both
//! context-free and contextual bandits seamlessly.

use rand::{SeedableRng, rngs::StdRng};
use trashpanda::policies::{EpsilonGreedy, LinUcb};
use trashpanda::{Bandit, ContextFreeAdapter, ContextualAdapter};

fn main() {
    println!("=== Unified Bandit API Example ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Example 1: Context-free bandit using the unified API
    println!("--- Context-Free Bandit (Epsilon-Greedy) ---");
    demo_context_free(&mut rng);

    println!("\n--- Contextual Bandit (LinUCB) ---");
    demo_contextual(&mut rng);

    println!("\n--- Flexibility: Context-Free Using Context API ---");
    demo_flexibility(&mut rng);

    println!("\n--- Runtime Detection ---");
    demo_runtime_detection(&mut rng);
}

fn demo_context_free(rng: &mut StdRng) {
    // Create a context-free bandit using the unified API
    let policy = EpsilonGreedy::new(0.1);
    let adapter = ContextFreeAdapter::new(policy);
    let mut bandit = Bandit::new(vec!["red", "green", "blue"], adapter).unwrap();

    println!("Requires context: {}", bandit.requires_context());
    println!("Number of features: {}", bandit.num_features());

    // Train without context
    bandit
        .fit(&["red", "red", "green"], &[1.0, 0.8, 0.3])
        .unwrap();

    // Predict without context
    let choice = bandit.predict(rng).unwrap();
    println!("Prediction: {}", choice);

    // Get expectations
    let expectations = bandit.predict_expectations();
    println!("Expectations: {:?}", expectations);
}

fn demo_contextual(rng: &mut StdRng) {
    // Create a contextual bandit using the unified API
    let policy = LinUcb::new(1.0, 1.0, 3);
    let adapter = ContextualAdapter::new(policy, 3);
    let mut bandit = Bandit::new(vec![1, 2, 3], adapter).unwrap();

    println!("Requires context: {}", bandit.requires_context());
    println!("Number of features: {}", bandit.num_features());

    // Train with context
    let decisions = vec![1, 2, 3];
    let contexts = vec![
        vec![1.0, 0.0, 0.5],
        vec![0.0, 1.0, 0.3],
        vec![0.5, 0.5, 1.0],
    ];
    let rewards = vec![0.8, 0.3, 0.9];

    bandit
        .fit_with_context(&decisions, &contexts, &rewards)
        .unwrap();

    // Predict with context
    let test_context = vec![0.7, 0.2, 0.8];
    let choice = bandit.predict_with_context(&test_context, rng).unwrap();
    println!("Prediction for context {:?}: {}", test_context, choice);

    // Get expectations for the context
    let expectations = bandit
        .predict_expectations_with_context(&test_context)
        .unwrap();
    println!("Expectations: {:?}", expectations);
}

fn demo_flexibility(rng: &mut StdRng) {
    // Context-free bandits can gracefully handle context API calls
    let policy = EpsilonGreedy::<&str>::new(0.1);
    let adapter = ContextFreeAdapter::new(policy);
    let mut bandit = Bandit::new(vec!["A", "B", "C"], adapter).unwrap();

    // Even though it's context-free, we can use the context API
    // The context will simply be ignored
    let contexts = vec![
        vec![1.0, 0.0], // This will be ignored
        vec![0.5, 0.5], // This will be ignored too
    ];

    bandit
        .fit_with_context(&["A", "B"], &contexts, &[1.0, 0.5])
        .unwrap();

    // Can predict with or without context
    let choice1 = bandit.predict(rng).unwrap();
    let choice2 = bandit.predict_with_context(&[0.3, 0.7], rng).unwrap();

    println!("Context-free policy works with both APIs:");
    println!("  predict(): {}", choice1);
    println!("  predict_with_context(): {}", choice2);
}

fn demo_runtime_detection(rng: &mut StdRng) {
    // You can write generic code that works with both types

    // Create two bandits with different requirements
    let cf_bandit = Bandit::epsilon_greedy(vec!["a", "b"], 0.1).unwrap();
    let ctx_bandit = Bandit::linucb(vec!["x", "y"], 1.0, 1.0, 2).unwrap();

    // Function that handles any unified bandit
    fn describe_bandit<A, P>(bandit: &Bandit<A, P>)
    where
        A: Clone + Eq + std::hash::Hash + std::fmt::Display,
        P: trashpanda::UnifiedPolicy<A>,
    {
        if bandit.requires_context() {
            println!(
                "This is a contextual bandit expecting {} features",
                bandit.num_features()
            );
        } else {
            println!("This is a context-free bandit");
        }

        println!("Available arms: {} arms", bandit.arms().len());
    }

    println!("Bandit 1:");
    describe_bandit(&cf_bandit);

    println!("\nBandit 2:");
    describe_bandit(&ctx_bandit);

    // You could even write adaptive code
    fn make_prediction<A, P>(
        bandit: &Bandit<A, P>,
        context: Option<&[f64]>,
        rng: &mut StdRng,
    ) -> Result<A, trashpanda::BanditError>
    where
        A: Clone + Eq + std::hash::Hash,
        P: trashpanda::UnifiedPolicy<A>,
    {
        if bandit.requires_context() {
            let ctx = context.expect("Context required but not provided");
            bandit.predict_with_context(ctx, rng)
        } else {
            bandit.predict(rng)
        }
    }

    // Use the adaptive function
    let _choice1 = make_prediction(&cf_bandit, None, rng).unwrap();
    let _choice2 = make_prediction(&ctx_bandit, Some(&[0.5, 0.5]), rng).unwrap();

    println!("\nAdaptive prediction function works with both types!");
}
