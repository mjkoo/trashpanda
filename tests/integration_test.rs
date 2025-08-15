//! Integration tests for core infrastructure.

use trashpanda::prelude::*;

#[test]
fn test_basic_bandit_creation() {
    let bandit = Bandit::builder()
        .arms(vec!["option_a", "option_b", "option_c"])
        .policy(LearningPolicy::Random)
        .build()
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
        .build()
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
        .build()
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
        .build()
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
        .build();
    assert!(result.is_ok());

    // Invalid epsilon (too high)
    let result = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::EpsilonGreedy { epsilon: 1.5 })
        .build();
    assert!(matches!(result, Err(BanditError::InvalidParameter { .. })));

    // Invalid epsilon (negative)
    let result = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::EpsilonGreedy { epsilon: -0.1 })
        .build();
    assert!(matches!(result, Err(BanditError::InvalidParameter { .. })));

    // TODO: Enable UCB tests when UCB is implemented
    // // Valid UCB alpha
    // let result = Bandit::builder()
    //     .arms(vec![1, 2, 3])
    //     .policy(LearningPolicy::Ucb { alpha: 2.0 })
    //     .build();
    // assert!(result.is_ok());

    // // Invalid UCB alpha
    // let result = Bandit::builder()
    //     .arms(vec![1, 2, 3])
    //     .policy(LearningPolicy::Ucb { alpha: 0.0 })
    //     .build();
    // assert!(matches!(result, Err(BanditError::InvalidParameter { .. })));
}
