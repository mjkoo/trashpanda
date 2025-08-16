//! Example demonstrating the LinUCB contextual bandit algorithm
//!
//! This example shows how to use LinUCB for contextual decision-making,
//! where the optimal arm depends on context features (e.g., user characteristics).

use rand::rng;
use trashpanda::Bandit;

fn main() {
    println!("=== LinUCB Contextual Bandit Example ===\n");

    // Simulate a recommendation system with 3 content types
    let content_types = vec!["sports", "tech", "politics"];

    // Create LinUCB bandit with:
    // - alpha=1.0 (exploration parameter)
    // - l2_lambda=1.0 (regularization)
    // - 3 context features (age_group, interest_level, time_of_day)
    let mut bandit = Bandit::linucb(content_types.clone(), 1.0, 1.0, 3).unwrap();

    println!("Content types: {:?}", content_types);
    println!("Context features: [age_group, interest_level, time_of_day]\n");

    // Training phase: Learn from historical data
    println!("--- Training Phase ---");

    // Simulate training data where:
    // - Young users (age_group=0.0) prefer sports
    // - Middle-aged users (age_group=0.5) prefer tech
    // - Older users (age_group=1.0) prefer politics

    let training_data = vec![
        // Young users engaging with sports content
        ("sports", vec![0.0, 0.8, 0.5], 0.9),
        ("sports", vec![0.1, 0.7, 0.3], 0.8),
        ("sports", vec![0.0, 0.9, 0.7], 0.95),
        ("tech", vec![0.0, 0.5, 0.5], 0.3),
        ("politics", vec![0.1, 0.4, 0.5], 0.2),
        // Middle-aged users engaging with tech content
        ("tech", vec![0.5, 0.9, 0.4], 0.85),
        ("tech", vec![0.4, 0.8, 0.6], 0.9),
        ("tech", vec![0.6, 0.7, 0.5], 0.8),
        ("sports", vec![0.5, 0.6, 0.5], 0.4),
        ("politics", vec![0.5, 0.5, 0.3], 0.5),
        // Older users engaging with politics content
        ("politics", vec![0.9, 0.8, 0.4], 0.9),
        ("politics", vec![1.0, 0.7, 0.6], 0.85),
        ("politics", vec![0.8, 0.9, 0.5], 0.88),
        ("sports", vec![0.9, 0.4, 0.5], 0.3),
        ("tech", vec![1.0, 0.3, 0.4], 0.25),
    ];

    // Convert to the format needed for fit_batch
    let decisions: Vec<&str> = training_data.iter().map(|(d, _, _)| *d).collect();
    let contexts: Vec<Vec<f64>> = training_data.iter().map(|(_, c, _)| c.clone()).collect();
    let rewards: Vec<f64> = training_data.iter().map(|(_, _, r)| *r).collect();

    bandit.fit_batch(&decisions, &contexts, &rewards).unwrap();

    println!("Trained on {} samples\n", training_data.len());

    // Testing phase: Make predictions for new users
    println!("--- Testing Phase ---");

    let mut rng = rng();

    // Test contexts representing different user types
    let test_contexts = vec![
        ("Young sports fan", vec![0.05, 0.85, 0.5]),
        ("Tech professional", vec![0.45, 0.9, 0.4]),
        ("Retired politics enthusiast", vec![0.9, 0.8, 0.6]),
        ("Uncertain user", vec![0.5, 0.5, 0.5]),
    ];

    for (user_type, context) in test_contexts {
        println!("\n{} (context: {:?})", user_type, context);

        // Get expected rewards for each content type
        let expectations = bandit.predict_expectations(&context);

        println!("Expected rewards:");
        let mut sorted_expectations: Vec<_> = expectations.iter().collect();
        sorted_expectations.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (content, reward) in sorted_expectations {
            println!("  {}: {:.3}", content, reward);
        }

        // Make a prediction (includes exploration via UCB)
        let recommendation = bandit.predict(&context, &mut rng).unwrap();
        println!("Recommendation: {}", recommendation);
    }

    println!("\n--- Exploration vs Exploitation ---");

    // Demonstrate how LinUCB balances exploration and exploitation
    let uncertain_context = vec![0.3, 0.3, 0.3]; // A context we haven't seen much

    println!("\nFor an uncertain context {:?}:", uncertain_context);
    println!("Making 20 predictions to show exploration behavior:");

    let mut selections = std::collections::HashMap::new();
    for _ in 0..20 {
        let choice = bandit.predict(&uncertain_context, &mut rng).unwrap();
        *selections.entry(choice).or_insert(0) += 1;
    }

    for content in &content_types {
        let count = selections.get(content).unwrap_or(&0);
        println!("  {}: {} times ({}%)", content, count, count * 5);
    }

    println!("\nNote: LinUCB explores uncertain arms while exploiting known good ones.");
    println!("The Upper Confidence Bound ensures we don't miss potentially better options.");
}
