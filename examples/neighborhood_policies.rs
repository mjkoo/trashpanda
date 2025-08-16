//! Example demonstrating neighborhood-based bandit policies
//!
//! This example shows how to use KNearest, Radius, and Clusters neighborhood policies
//! with contextual bandits.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use trashpanda::{
    Bandit, BanditError,
    contextual::lingreedy::LinGreedy,
    neighborhood::{
        clusters::Clusters,
        distance::{Cosine, Euclidean, Manhattan},
        knearest::KNearest,
        radius::Radius,
    },
};

fn main() -> Result<(), BanditError> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

    println!("=== Neighborhood-Based Bandit Policies Example ===\n");

    // Define arms (e.g., different product recommendations)
    let arms = vec!["Product A", "Product B", "Product C"];

    // Example 1: K-Nearest Neighbors
    println!("1. K-Nearest Neighbors Policy");
    println!("-------------------------------");
    demo_knearest(&arms, &mut rng)?;

    // Example 2: Radius-based Neighbors
    println!("\n2. Radius-based Neighbors Policy");
    println!("---------------------------------");
    demo_radius(&arms, &mut rng)?;

    // Example 3: Clustering Policy
    println!("\n3. Clustering Policy");
    println!("--------------------");
    demo_clusters(&arms, &mut rng)?;

    Ok(())
}

fn demo_knearest(arms: &[&str], rng: &mut Xoshiro256PlusPlus) -> Result<(), BanditError> {
    // Create a KNearest policy with LinGreedy as underlying policy
    let underlying = LinGreedy::new(0.1, 1.0, 2); // epsilon=0.1, l2_reg=1.0, 2 features
    let policy = KNearest::new(underlying, 3, Euclidean); // k=3 nearest neighbors
    let mut bandit = Bandit::new(arms.to_vec(), policy)?;

    // Train with some synthetic data
    // Context: [user_age_normalized, time_of_day_normalized]
    let training_data = vec![
        ([0.2, 0.1], "Product A", 1.0), // Young user, morning -> Product A worked
        ([0.2, 0.2], "Product A", 0.9),
        ([0.8, 0.9], "Product C", 0.8), // Older user, evening -> Product C worked
        ([0.7, 0.8], "Product C", 0.7),
        ([0.5, 0.5], "Product B", 0.6), // Middle-aged, afternoon -> Product B worked
        ([0.4, 0.6], "Product B", 0.5),
    ];

    // Train the bandit with each data point
    for (context, arm, reward) in &training_data {
        bandit.partial_fit(&[*arm], &context[..], &[*reward])?;
    }

    // Make predictions for new contexts
    let test_contexts = vec![
        ([0.25, 0.15], "young user in morning"),
        ([0.75, 0.85], "older user in evening"),
        ([0.45, 0.55], "middle-aged user in afternoon"),
    ];

    for (context, description) in test_contexts {
        let prediction = bandit.predict(&context, rng)?;
        let expectations = bandit.predict_expectations(&context, rng);

        println!("  Context: {} -> Predicted: {}", description, prediction);
        println!("  Expected rewards: {:?}", expectations);
    }

    Ok(())
}

fn demo_radius(arms: &[&str], rng: &mut Xoshiro256PlusPlus) -> Result<(), BanditError> {
    // Create a Radius policy with LinGreedy as underlying policy
    let underlying = LinGreedy::new(0.1, 1.0, 2);
    let policy = Radius::new(underlying, 0.3, Manhattan) // radius=0.3 with Manhattan distance
        .with_min_neighbors(2); // require at least 2 neighbors
    let mut bandit = Bandit::new(arms.to_vec(), policy)?;

    // Train with clustered data
    let training_data = vec![
        // Cluster 1: young users prefer Product A
        ([0.1, 0.1], "Product A", 0.9),
        ([0.15, 0.12], "Product A", 0.85),
        ([0.2, 0.15], "Product A", 0.8),
        // Cluster 2: older users prefer Product C
        ([0.8, 0.85], "Product C", 0.9),
        ([0.85, 0.8], "Product C", 0.85),
        ([0.9, 0.9], "Product C", 0.8),
        // Some scattered data
        ([0.5, 0.5], "Product B", 0.6),
    ];

    // Train the bandit with each data point
    for (context, arm, reward) in &training_data {
        bandit.partial_fit(&[*arm], &context[..], &[*reward])?;
    }

    // Test predictions
    let test_contexts = vec![
        ([0.12, 0.13], "near cluster 1"),
        ([0.82, 0.87], "near cluster 2"),
        ([0.5, 0.1], "between clusters"),
    ];

    for (context, description) in test_contexts {
        let prediction = bandit.predict(&context, rng)?;
        println!("  Context: {} -> Predicted: {}", description, prediction);
    }

    Ok(())
}

fn demo_clusters(arms: &[&str], rng: &mut Xoshiro256PlusPlus) -> Result<(), BanditError> {
    // Create a Clusters policy with LinGreedy as underlying policy
    let underlying = LinGreedy::new(0.1, 1.0, 2);
    let policy = Clusters::new(underlying, 2, Cosine) // 2 clusters with Cosine distance
        .with_max_iter(50); // max 50 iterations for k-means
    let mut bandit = Bandit::new(arms.to_vec(), policy)?;

    // Generate training data with two clear patterns
    let training_data = vec![
        // Pattern 1: Low values prefer Product A
        ([0.1, 0.05], "Product A", 0.95),
        ([0.05, 0.1], "Product A", 0.9),
        ([0.15, 0.1], "Product A", 0.85),
        ([0.1, 0.15], "Product A", 0.8),
        ([0.2, 0.1], "Product B", 0.3),
        // Pattern 2: High values prefer Product C
        ([0.9, 0.95], "Product C", 0.95),
        ([0.95, 0.9], "Product C", 0.9),
        ([0.85, 0.9], "Product C", 0.85),
        ([0.9, 0.85], "Product C", 0.8),
        ([0.8, 0.9], "Product B", 0.3),
    ];

    // Train the model (clustering happens automatically)
    // Train the bandit with each data point
    for (context, arm, reward) in &training_data {
        bandit.partial_fit(&[*arm], &context[..], &[*reward])?;
    }

    // Test on new contexts
    let test_contexts = vec![
        ([0.08, 0.12], "should be cluster 1 (Product A)"),
        ([0.92, 0.88], "should be cluster 2 (Product C)"),
        ([0.5, 0.5], "between clusters"),
    ];

    for (context, description) in test_contexts {
        let prediction = bandit.predict(&context, rng)?;
        let expectations = bandit.predict_expectations(&context, rng);

        println!("  Context: {} -> Predicted: {}", description, prediction);
        println!("  Expected rewards: {:?}", expectations);
    }

    Ok(())
}
