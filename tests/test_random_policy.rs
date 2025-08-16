use rand::SeedableRng;
use std::collections::HashMap;
use trashpanda::Bandit;
use trashpanda::policies::Random;

#[test]
fn test_random_policy_basic() {
    let bandit = Bandit::new(vec!["a", "b", "c"], Random::default()).unwrap();

    // Should be able to predict
    let mut rng = rand::rng();
    let choice = bandit.predict_simple(&mut rng).unwrap();
    assert!(["a", "b", "c"].contains(&choice.as_ref()));
}

#[test]
fn test_random_policy_expectations() {
    let bandit = Bandit::new(vec![1, 2, 3, 4], Random::default()).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let expectations = bandit.predict_expectations_simple(&mut rng);

    assert_eq!(expectations.len(), 4);
    // With no training data, all expectations should be 0.0
    for i in 1..=4 {
        let expected_reward = expectations.get(&i).unwrap();
        assert_eq!(*expected_reward, 0.0);
    }
}

#[test]
fn test_random_policy_distribution() {
    let bandit = Bandit::new(vec!["red", "green", "blue"], Random::default()).unwrap();

    // Make many predictions and check distribution
    let mut counts = HashMap::new();
    let n_samples = 3000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    for _ in 0..n_samples {
        let choice = bandit.predict_simple(&mut rng).unwrap();
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
    let mut bandit = Bandit::new(vec![1, 2, 3], Random::default()).unwrap();

    // Train with some data
    let decisions = vec![1, 2, 1, 3, 2];
    let rewards = vec![1.0, 0.5, 0.8, 0.2, 0.9];
    bandit.fit_simple(&decisions, &rewards).unwrap();

    // Random policy should now reflect actual average rewards from training
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let expectations = bandit.predict_expectations_simple(&mut rng);
    // Expected: arm 1 = (1.0 + 0.8) / 2 = 0.9, arm 2 = (0.5 + 0.9) / 2 = 0.7, arm 3 = 0.2 / 1 = 0.2
    assert!((expectations[&1] - 0.9).abs() < 1e-10);
    assert!((expectations[&2] - 0.7).abs() < 1e-10);
    assert!((expectations[&3] - 0.2).abs() < 1e-10);
}

#[test]
fn test_random_policy_partial_fit() {
    let mut bandit = Bandit::new(vec!["x", "y", "z"], Random::default()).unwrap();

    // Partial fit with some data
    bandit.fit_simple(&["x"], &[1.0]).unwrap();
    bandit.fit_simple(&["y", "z"], &[0.5, 0.8]).unwrap();

    // Should still work for predictions
    let mut rng = rand::rng();
    let choice = bandit.predict_simple(&mut rng).unwrap();
    assert!(["x", "y", "z"].contains(&choice.as_ref()));
}

#[test]
fn test_random_policy_dynamic_arms() {
    let mut bandit = Bandit::new(vec![1, 2], Random::default()).unwrap();

    // Initial expectations - no training data, so all 0.0
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let expectations = bandit.predict_expectations_simple(&mut rng);
    assert_eq!(expectations.len(), 2);
    assert_eq!(expectations[&1], 0.0);
    assert_eq!(expectations[&2], 0.0);

    // Add an arm
    bandit.add_arm(3).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let expectations = bandit.predict_expectations_simple(&mut rng);
    assert_eq!(expectations.len(), 3);
    // All arms still have no training data
    for expected_reward in expectations.values() {
        assert_eq!(*expected_reward, 0.0);
    }

    // Remove an arm
    bandit.remove_arm(&2).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let expectations = bandit.predict_expectations_simple(&mut rng);
    assert_eq!(expectations.len(), 2);
    assert_eq!(expectations[&1], 0.0);
    assert_eq!(expectations[&3], 0.0);
}
