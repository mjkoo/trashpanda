use rand::SeedableRng;
use std::collections::HashMap;
use trashpanda::{Bandit, LearningPolicy};

#[test]
fn test_random_policy_basic() {
    let bandit = Bandit::builder()
        .arms(vec!["a", "b", "c"])
        .policy(LearningPolicy::Random)
        .build()
        .unwrap();

    // Should be able to predict
    let choice = bandit.predict().unwrap();
    assert!(["a", "b", "c"].contains(&choice.as_ref()));
}

#[test]
fn test_random_policy_expectations() {
    let bandit = Bandit::builder()
        .arms(vec![1, 2, 3, 4])
        .policy(LearningPolicy::Random)
        .build()
        .unwrap();

    let expectations = bandit.predict_expectations().unwrap();

    assert_eq!(expectations.len(), 4);
    for i in 1..=4 {
        let prob = expectations.get(&i).unwrap();
        assert!((prob - 0.25).abs() < 1e-10);
    }
}

#[test]
fn test_random_policy_distribution() {
    let bandit = Bandit::builder()
        .arms(vec!["red", "green", "blue"])
        .policy(LearningPolicy::Random)
        .build()
        .unwrap();

    // Make many predictions and check distribution
    let mut counts = HashMap::new();
    let n_samples = 3000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    for _ in 0..n_samples {
        let choice = bandit.predict_with_rng(&mut rng).unwrap();
        *counts.entry(choice).or_insert(0) += 1;
    }

    // Each arm should be selected roughly 1/3 of the time
    for count in counts.values() {
        let proportion = *count as f64 / n_samples as f64;
        assert!((proportion - 1.0 / 3.0).abs() < 0.05); // Allow 5% deviation
    }
}

#[test]
fn test_random_policy_with_training() {
    let mut bandit = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::Random)
        .build()
        .unwrap();

    // Train with some data
    let decisions = vec![1, 2, 1, 3, 2];
    let rewards = vec![1.0, 0.5, 0.8, 0.2, 0.9];
    bandit.fit(&decisions, &rewards).unwrap();

    // Random policy should still give equal probabilities
    let expectations = bandit.predict_expectations().unwrap();
    for prob in expectations.values() {
        assert!((prob - 1.0 / 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_random_policy_partial_fit() {
    let mut bandit = Bandit::builder()
        .arms(vec!["x", "y", "z"])
        .policy(LearningPolicy::Random)
        .build()
        .unwrap();

    // Partial fit with some data
    bandit.partial_fit(&["x"], &[1.0]).unwrap();
    bandit.partial_fit(&["y", "z"], &[0.5, 0.8]).unwrap();

    // Should still work for predictions
    let choice = bandit.predict().unwrap();
    assert!(["x", "y", "z"].contains(&choice.as_ref()));
}

#[test]
fn test_random_policy_dynamic_arms() {
    let mut bandit = Bandit::builder()
        .arms(vec![1, 2])
        .policy(LearningPolicy::Random)
        .build()
        .unwrap();

    // Initial expectations
    let expectations = bandit.predict_expectations().unwrap();
    assert_eq!(expectations.len(), 2);
    assert!((expectations[&1] - 0.5).abs() < 1e-10);

    // Add an arm
    bandit.add_arm(3).unwrap();
    let expectations = bandit.predict_expectations().unwrap();
    assert_eq!(expectations.len(), 3);
    for prob in expectations.values() {
        assert!((prob - 1.0 / 3.0).abs() < 1e-10);
    }

    // Remove an arm
    bandit.remove_arm(&2).unwrap();
    let expectations = bandit.predict_expectations().unwrap();
    assert_eq!(expectations.len(), 2);
    assert!((expectations[&1] - 0.5).abs() < 1e-10);
    assert!((expectations[&3] - 0.5).abs() < 1e-10);
}
