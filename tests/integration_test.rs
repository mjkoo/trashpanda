//! Integration tests for core infrastructure.

use trashpanda::policies::{EpsilonGreedy, Random};
use trashpanda::prelude::*;

#[test]
fn test_basic_bandit_creation() {
    let bandit = Bandit::new(vec!["option_a", "option_b", "option_c"], Random).unwrap();

    assert_eq!(bandit.arms().len(), 3);
    assert!(bandit.has_arm(&"option_a"));
    assert!(bandit.has_arm(&"option_b"));
    assert!(bandit.has_arm(&"option_c"));
}

#[test]
fn test_integer_arms() {
    let bandit = Bandit::new(vec![1, 2, 3], EpsilonGreedy::new(0.15)).unwrap();

    assert_eq!(bandit.arms().len(), 3);
    assert!(bandit.has_arm(&1));
    assert!(bandit.has_arm(&2));
    assert!(bandit.has_arm(&3));
}

#[test]
fn test_string_arms() {
    let bandit = Bandit::new(vec!["red", "green", "blue"], EpsilonGreedy::new(0.15)).unwrap();

    assert_eq!(bandit.arms().len(), 3);
    assert!(bandit.has_arm(&"red"));
    assert!(bandit.has_arm(&"green"));
    assert!(bandit.has_arm(&"blue"));
}

#[test]
fn test_dynamic_arm_management() {
    let mut bandit = Bandit::new(vec![1, 2], Random).unwrap();

    // Start with 2 arms
    assert_eq!(bandit.arms().len(), 2);

    // Add a new arm
    bandit.add_arm(3).unwrap();
    assert_eq!(bandit.arms().len(), 3);
    assert!(bandit.has_arm(&3));

    // Remove an arm
    bandit.remove_arm(&2).unwrap();
    assert_eq!(bandit.arms().len(), 2);
    assert!(!bandit.has_arm(&2));

    // Verify remaining arms
    assert!(bandit.has_arm(&1));
    assert!(bandit.has_arm(&3));
}

#[test]
fn test_policy_validation() {
    // Valid epsilon
    let result = Bandit::new(vec![1, 2, 3], EpsilonGreedy::new(0.5));
    assert!(result.is_ok());

    // Note: Invalid epsilon values are now caught at policy construction time
    // by the assert! in EpsilonGreedy::new, not at bandit build time
}

#[test]
fn test_direct_construction() {
    // Test direct construction with policies
    let bandit = Bandit::new(vec![1, 2, 3], EpsilonGreedy::new(0.1)).unwrap();
    assert_eq!(bandit.arms().len(), 3);

    let bandit = Bandit::new(vec!["a", "b", "c"], Random).unwrap();
    assert_eq!(bandit.arms().len(), 3);

    // Test with direct construction
    let bandit2 = Bandit::new(vec!["x", "y"], EpsilonGreedy::new(0.2)).unwrap();
    assert_eq!(bandit2.arms().len(), 2);
}
