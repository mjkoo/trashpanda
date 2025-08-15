//! Integration tests for core infrastructure.

use trashpanda::policies::{EpsilonGreedy, Random};
use trashpanda::prelude::*;

#[test]
fn test_basic_bandit_creation() {
    let bandit = Bandit::builder()
        .arms(vec!["option_a", "option_b", "option_c"])
        .policy(LearningPolicy::Random)
        .build::<Random>()
        .unwrap();

    assert_eq!(bandit.n_arms(), 3);
    assert!(bandit.has_arm(&"option_a"));
    assert!(bandit.has_arm(&"option_b"));
    assert!(bandit.has_arm(&"option_c"));
}

#[test]
fn test_integer_arms() {
    let bandit = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.15 })
        .build::<EpsilonGreedy<_>>()
        .unwrap();

    assert_eq!(bandit.n_arms(), 3);
    assert!(bandit.has_arm(&1));
    assert!(bandit.has_arm(&2));
    assert!(bandit.has_arm(&3));
}

#[test]
fn test_string_arms() {
    let bandit = Bandit::builder()
        .arms(vec!["red", "green", "blue"])
        .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.15 })
        .build::<EpsilonGreedy<_>>()
        .unwrap();

    assert_eq!(bandit.n_arms(), 3);
    assert!(bandit.has_arm(&"red"));
    assert!(bandit.has_arm(&"green"));
    assert!(bandit.has_arm(&"blue"));
}

#[test]
fn test_dynamic_arm_management() {
    let mut bandit = Bandit::builder()
        .arms(vec![1, 2])
        .policy(LearningPolicy::Random)
        .build::<Random>()
        .unwrap();

    // Start with 2 arms
    assert_eq!(bandit.n_arms(), 2);

    // Add a new arm
    bandit.add_arm(3).unwrap();
    assert_eq!(bandit.n_arms(), 3);
    assert!(bandit.has_arm(&3));

    // Remove an arm
    bandit.remove_arm(&2).unwrap();
    assert_eq!(bandit.n_arms(), 2);
    assert!(!bandit.has_arm(&2));

    // Verify remaining arms
    assert!(bandit.has_arm(&1));
    assert!(bandit.has_arm(&3));
}

#[test]
fn test_policy_validation() {
    // Valid epsilon
    let result = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.5 })
        .build::<EpsilonGreedy<_>>();
    assert!(result.is_ok());

    // Invalid epsilon (too high)
    let result = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::EpsilonGreedy { epsilon: 1.5 })
        .build::<EpsilonGreedy<_>>();
    assert!(matches!(result, Err(BanditError::InvalidParameter { .. })));

    // Invalid epsilon (negative)
    let result = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::EpsilonGreedy { epsilon: -0.1 })
        .build::<EpsilonGreedy<_>>();
    assert!(matches!(result, Err(BanditError::InvalidParameter { .. })));
}

#[test]
fn test_direct_construction() {
    // Test with convenience constructors
    let bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.1).unwrap();
    assert_eq!(bandit.n_arms(), 3);

    let bandit = Bandit::random(vec!["a", "b", "c"]).unwrap();
    assert_eq!(bandit.n_arms(), 3);

    // Test direct new
    let bandit = Bandit::new(vec![1, 2, 3], Random).unwrap();
    assert_eq!(bandit.n_arms(), 3);

    let bandit = Bandit::new(vec!["x", "y"], EpsilonGreedy::new(0.2)).unwrap();
    assert_eq!(bandit.n_arms(), 2);
}
