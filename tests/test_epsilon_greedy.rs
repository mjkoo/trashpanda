use rand::SeedableRng;
use std::collections::HashMap;
use trashpanda::Bandit;
use trashpanda::policies::EpsilonGreedy;

#[test]
fn test_epsilon_greedy_pure_exploitation() {
    // With epsilon = 0, should always choose the best arm
    let mut bandit = Bandit::new(vec!["a", "b", "c"], EpsilonGreedy::new(0.0)).unwrap();

    // Train with data where "b" is clearly the best
    bandit
        .fit(&["a", "b", "c", "b", "a"], &[0.1, 1.0, 0.2, 0.9, 0.3])
        .unwrap();

    // Should always predict "b"
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..20 {
        let choice = bandit.predict_with_rng(&mut rng).unwrap();
        assert_eq!(choice, "b");
    }
}

#[test]
fn test_epsilon_greedy_pure_exploration() {
    // With epsilon = 1.0, should behave like random policy
    let mut bandit = Bandit::new(vec![1, 2, 3], EpsilonGreedy::new(1.0)).unwrap();

    // Train with data
    bandit.fit(&[1, 2, 3, 2], &[0.1, 1.0, 0.2, 0.9]).unwrap();

    // Check that expectations are uniform despite training
    let expectations = bandit.predict_expectations().unwrap();
    for (_arm, prob) in expectations {
        assert!((prob - 1.0 / 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_epsilon_greedy_mixed_strategy() {
    // With epsilon = 0.2, should explore 20% and exploit 80%
    let mut bandit = Bandit::new(vec!["red", "green", "blue"], EpsilonGreedy::new(0.2)).unwrap();

    // Train with data where "green" is best
    bandit
        .fit(
            &["red", "green", "blue", "green", "red"],
            &[0.3, 0.9, 0.1, 0.8, 0.2],
        )
        .unwrap();

    // Check expectations
    let expectations = bandit.predict_expectations().unwrap();

    // "green" should get 80% + 6.67% ≈ 86.67%
    assert!((expectations[&"green"] - (0.8 + 0.2 / 3.0)).abs() < 1e-10);
    // Others should get 6.67% each
    assert!((expectations[&"red"] - 0.2 / 3.0).abs() < 1e-10);
    assert!((expectations[&"blue"] - 0.2 / 3.0).abs() < 1e-10);
}

#[test]
fn test_epsilon_greedy_learning() {
    // Test that the policy learns and adapts
    let mut bandit = Bandit::new(vec![1, 2, 3], EpsilonGreedy::new(0.1)).unwrap();

    // Initially all arms are equal
    let initial_exp = bandit.predict_expectations().unwrap();
    // With no data, expectations might be arbitrary but valid
    assert_eq!(initial_exp.len(), 3);

    // Train with data where arm 3 is best
    bandit
        .fit(&[1, 2, 3, 3, 3], &[0.2, 0.3, 0.9, 0.8, 0.95])
        .unwrap();

    // Now arm 3 should have highest probability
    let trained_exp = bandit.predict_expectations().unwrap();
    assert!(trained_exp[&3] > trained_exp[&1]);
    assert!(trained_exp[&3] > trained_exp[&2]);
}

#[test]
fn test_epsilon_greedy_incremental_learning() {
    // Test partial_fit
    let mut bandit = Bandit::new(vec!["x", "y", "z"], EpsilonGreedy::new(0.0)).unwrap();

    // Incrementally train
    bandit.partial_fit(&["x"], &[0.5]).unwrap();
    bandit.partial_fit(&["y"], &[0.7]).unwrap();
    bandit.partial_fit(&["z"], &[0.3]).unwrap();
    bandit.partial_fit(&["y"], &[0.8]).unwrap();

    // "y" should be best with average 0.75
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let choice = bandit.predict_with_rng(&mut rng).unwrap();
    assert_eq!(choice, "y");
}

#[test]
fn test_epsilon_greedy_dynamic_arms() {
    let mut bandit = Bandit::new(vec![1, 2], EpsilonGreedy::new(0.3)).unwrap();

    // Train initial arms
    bandit.fit(&[1, 2, 1], &[0.6, 0.4, 0.7]).unwrap();

    // Add a new arm
    bandit.add_arm(3).unwrap();

    // New arm starts with no data (average = 0)
    // Arm 1 has average 0.65, so it should be preferred
    let expectations = bandit.predict_expectations().unwrap();
    assert!(expectations[&1] > expectations[&3]);

    // Train the new arm to be best
    bandit.partial_fit(&[3, 3], &[0.9, 0.95]).unwrap();

    // Now arm 3 should be preferred
    let expectations = bandit.predict_expectations().unwrap();
    assert!(expectations[&3] > expectations[&1]);
    assert!(expectations[&3] > expectations[&2]);
}

#[test]
fn test_epsilon_greedy_distribution() {
    // Statistical test of epsilon-greedy behavior
    let mut bandit = Bandit::new(vec!["a", "b", "c"], EpsilonGreedy::new(0.3)).unwrap();

    // Train so "b" is clearly best
    bandit.fit(&["a", "b", "c"], &[0.2, 0.9, 0.1]).unwrap();
    bandit.fit(&["a", "b", "c"], &[0.3, 0.8, 0.2]).unwrap();

    // Sample many times
    let mut counts = HashMap::new();
    let n_samples = 10000;
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    for _ in 0..n_samples {
        let choice = bandit.predict_with_rng(&mut rng).unwrap();
        *counts.entry(choice).or_insert(0) += 1;
    }

    // "b" should be selected approximately 70% + 10% = 80% of the time
    let b_proportion = counts[&"b"] as f64 / n_samples as f64;
    assert!((b_proportion - 0.8).abs() < 0.02); // Allow 2% deviation

    // "a" and "c" should each be selected approximately 10% of the time
    let a_proportion = counts[&"a"] as f64 / n_samples as f64;
    let c_proportion = counts[&"c"] as f64 / n_samples as f64;
    assert!((a_proportion - 0.1).abs() < 0.02);
    assert!((c_proportion - 0.1).abs() < 0.02);
}
