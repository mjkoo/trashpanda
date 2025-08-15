//! Integration tests for core infrastructure.

use trashpanda::prelude::*;

#[test]
fn test_basic_bandit_creation() {
    let bandit = Bandit::builder()
        .arms(vec!["option_a", "option_b", "option_c"])
        .policy(LearningPolicy::Random)
        .seed(42)
        .build()
        .unwrap();

    assert_eq!(bandit.n_arms(), 3);
    assert!(bandit.has_arm(&Arm::from("option_a")));
    assert!(bandit.has_arm(&Arm::from("option_b")));
    assert!(bandit.has_arm(&Arm::from("option_c")));
}

#[test]
fn test_mixed_arm_types() {
    let bandit = Bandit::builder()
        .arms(vec![Arm::from(1), Arm::from(2.5), Arm::from("three")])
        .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.15 })
        .build()
        .unwrap();

    assert_eq!(bandit.n_arms(), 3);
    assert!(bandit.has_arm(&Arm::from(1)));
    assert!(bandit.has_arm(&Arm::from(2.5)));
    assert!(bandit.has_arm(&Arm::from("three")));
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
    bandit.add_arm(Arm::from(3)).unwrap();
    assert_eq!(bandit.n_arms(), 3);
    assert!(bandit.has_arm(&Arm::from(3)));

    // Remove an arm
    bandit.remove_arm(&Arm::from(2)).unwrap();
    assert_eq!(bandit.n_arms(), 2);
    assert!(!bandit.has_arm(&Arm::from(2)));

    // Verify remaining arms
    assert!(bandit.has_arm(&Arm::from(1)));
    assert!(bandit.has_arm(&Arm::from(3)));
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

    // Valid UCB alpha
    let result = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::Ucb { alpha: 2.0 })
        .build();
    assert!(result.is_ok());

    // Invalid UCB alpha
    let result = Bandit::builder()
        .arms(vec![1, 2, 3])
        .policy(LearningPolicy::Ucb { alpha: 0.0 })
        .build();
    assert!(matches!(result, Err(BanditError::InvalidParameter { .. })));
}
