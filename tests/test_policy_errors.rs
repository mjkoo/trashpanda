//! Tests for policy error handling

use rand::SeedableRng;
use trashpanda::Bandit;

#[test]
fn test_bandit_api_prevents_misuse() {
    // The Bandit API should prevent using the wrong methods
    let mut context_free = Bandit::epsilon_greedy(vec!["a", "b"], 0.1).unwrap();
    let mut contextual = Bandit::linucb(vec![1, 2], 1.0, 1.0, 2).unwrap();

    // Context-free bandit accepts fit() normally
    assert!(!context_free.requires_context());
    assert!(context_free.fit(&["a"], &[1.0]).is_ok());

    // Context-free bandit can also use context API (context is ignored)
    let contexts = vec![vec![1.0, 0.0]];
    assert!(
        context_free
            .fit_with_context(&["a"], &contexts, &[1.0])
            .is_ok()
    );

    // Contextual bandit rejects fit() without context
    assert!(contextual.requires_context());
    let result = contextual.fit(&[1], &[1.0]);
    assert!(result.is_err());

    // Check the error message
    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("requires context"));
    }

    // But contextual API works
    let contexts = vec![vec![1.0, 0.0]];
    assert!(contextual.fit_with_context(&[1], &contexts, &[1.0]).is_ok());
}

#[test]
fn test_predict_context_requirements() {
    let context_free = Bandit::epsilon_greedy(vec!["a", "b"], 0.1).unwrap();
    let contextual = Bandit::linucb(vec![1, 2], 1.0, 1.0, 2).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Context-free bandit can predict without context
    assert!(context_free.predict(&mut rng).is_ok());

    // Context-free can also use context API (context ignored)
    assert!(
        context_free
            .predict_with_context(&[1.0, 0.0], &mut rng)
            .is_ok()
    );

    // Contextual bandit requires context
    let result = contextual.predict(&mut rng);
    assert!(result.is_err());

    // Check the error message
    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("requires context"));
    }

    // But works with context
    assert!(
        contextual
            .predict_with_context(&[1.0, 0.0], &mut rng)
            .is_ok()
    );
}

#[test]
fn test_expectations_context_requirements() {
    let context_free = Bandit::epsilon_greedy(vec!["a", "b"], 0.1).unwrap();
    let contextual = Bandit::linucb(vec![1, 2], 1.0, 1.0, 2).unwrap();

    // Context-free bandit can get expectations without context
    let exp = context_free.predict_expectations();
    assert_eq!(exp.len(), 2);

    // Context-free can also use context API (context ignored)
    let exp_with = context_free
        .predict_expectations_with_context(&[1.0, 0.0])
        .unwrap();
    assert_eq!(exp_with.len(), 2);

    // Contextual bandit requires context
    let exp_contextual = contextual
        .predict_expectations_with_context(&[1.0, 0.0])
        .unwrap();
    assert_eq!(exp_contextual.len(), 2);
}
