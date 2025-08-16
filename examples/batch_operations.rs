use rand::SeedableRng;
use trashpanda::{Bandit, simple::epsilon_greedy::EpsilonGreedy};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== TrashPanda Batch Operations Demo ===\n");

    // Create a non-contextual epsilon-greedy bandit
    let mut bandit = Bandit::new(
        vec!["Product A", "Product B", "Product C"],
        EpsilonGreedy::new(0.1),
    )?;

    // Train the bandit with some historical data
    let decisions = vec![
        "Product A",
        "Product A",
        "Product B",
        "Product C",
        "Product A",
    ];
    let rewards = vec![0.8, 0.9, 0.6, 0.7, 0.85];

    bandit.fit_simple(&decisions, &rewards)?;
    println!("✅ Trained bandit with {} observations", decisions.len());

    // Create RNG for predictions
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Single prediction vs batch prediction comparison
    println!("\n--- Single vs Batch Predictions ---");

    // Single predictions (traditional approach)
    println!("Single predictions:");
    for i in 0..3 {
        let prediction = bandit.predict_simple(&mut rng)?;
        println!("  Prediction {}: {}", i + 1, prediction);
    }

    // Batch predictions (new approach)
    println!("\nBatch predictions:");
    let batch_predictions = bandit.predict_batch_simple(3, &mut rng)?;
    for (i, prediction) in batch_predictions.iter().enumerate() {
        println!("  Prediction {}: {}", i + 1, prediction);
    }

    // Batch expectations
    println!("\n--- Batch Expectations ---");
    let batch_expectations = bandit.predict_expectations_batch_simple(2, &mut rng);
    for (i, expectations) in batch_expectations.iter().enumerate() {
        println!("Expectation set {}:", i + 1);
        for (arm, reward) in expectations {
            println!("  {}: {:.3}", arm, reward);
        }
    }

    // Contextual bandit batch operations
    println!("\n--- Contextual Bandit Batch Operations ---");

    let mut contextual_bandit = Bandit::linucb(
        vec!["Campaign A", "Campaign B"],
        1.0, // alpha
        1.0, // l2_lambda
        2,   // num_features: [user_age, user_income_level]
    )?;

    // Train with batch contexts (each decision has different context)
    let decisions = vec!["Campaign A", "Campaign B", "Campaign A"];
    let contexts = vec![
        vec![25.0, 1.0], // Young, low income
        vec![45.0, 3.0], // Middle-aged, high income
        vec![30.0, 2.0], // Young adult, medium income
    ];
    let rewards = vec![0.7, 0.9, 0.8];

    contextual_bandit.fit_batch(&decisions, &contexts, &rewards)?;
    println!(
        "✅ Trained contextual bandit with {} contextual observations",
        decisions.len()
    );

    // Batch predictions with different contexts
    let test_contexts = vec![
        vec![22.0, 1.0], // Very young, low income
        vec![50.0, 3.0], // Older, high income
        vec![35.0, 2.0], // Mid-career, medium income
    ];

    let contextual_predictions =
        contextual_bandit.predict_batch_contexts(&test_contexts, &mut rng)?;
    println!("\nContextual batch predictions:");
    for (i, (context, prediction)) in test_contexts
        .iter()
        .zip(contextual_predictions.iter())
        .enumerate()
    {
        println!(
            "  Context {} [age: {}, income: {}] → {}",
            i + 1,
            context[0],
            context[1],
            prediction
        );
    }

    // Batch expectations for contextual bandit
    let contextual_expectations =
        contextual_bandit.predict_expectations_batch_contexts(&test_contexts, &mut rng);
    println!("\nContextual batch expectations:");
    for (i, (context, expectations)) in test_contexts
        .iter()
        .zip(contextual_expectations.iter())
        .enumerate()
    {
        println!(
            "  Context {} [age: {}, income: {}]:",
            i + 1,
            context[0],
            context[1]
        );
        for (arm, reward) in expectations {
            println!("    {}: {:.3}", arm, reward);
        }
    }

    println!("\n=== Batch Operations Demo Complete ===");
    Ok(())
}
