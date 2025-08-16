use rand::SeedableRng;
use rand::rngs::StdRng;
use trashpanda::Bandit;
use trashpanda::policies::{KNearest, LinGreedy, Policy};

#[test]
fn test_knearest_basic() {
    // Create KNearest with LinGreedy as underlying policy (contextual)
    let underlying = LinGreedy::new(0.1, 1.0, 2); // epsilon=0.1, l2_lambda=1.0, 2 features
    let policy = KNearest::new(underlying, 2, "euclidean");

    let mut bandit = Bandit::new(vec!["A", "B", "C"], policy).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    // Train with some contextual data
    // We need to train one observation at a time since we have different contexts

    // Context [1, 0] leads to arm A being good
    let ctx1 = [1.0, 0.0];
    bandit.partial_fit(&["A"], &ctx1[..], &[1.0]).unwrap();

    let ctx2 = [0.9, 0.1];
    bandit.partial_fit(&["A"], &ctx2[..], &[0.9]).unwrap();

    let ctx3 = [1.1, 0.0];
    bandit.partial_fit(&["A"], &ctx3[..], &[0.8]).unwrap();

    // Context [0, 1] leads to arm B being good
    let ctx4 = [0.0, 1.0];
    bandit.partial_fit(&["B"], &ctx4[..], &[1.0]).unwrap();

    let ctx5 = [0.1, 0.9];
    bandit.partial_fit(&["B"], &ctx5[..], &[0.9]).unwrap();

    let ctx6 = [0.0, 1.1];
    bandit.partial_fit(&["B"], &ctx6[..], &[0.85]).unwrap();

    // Context [0.5, 0.5] leads to arm C being good
    let ctx7 = [0.5, 0.5];
    bandit.partial_fit(&["C"], &ctx7[..], &[1.0]).unwrap();

    let ctx8 = [0.4, 0.6];
    bandit.partial_fit(&["C"], &ctx8[..], &[0.95]).unwrap();

    // Test predictions - should favor A when context is near [1, 0]
    let ctx_test = [1.0, 0.0];
    let _pred = bandit.predict(&ctx_test[..], &mut rng).unwrap();
    // With k=2, should find the two closest contexts which were for arm A

    // Test expectations
    let ctx_test2 = [0.0, 1.0];
    let expectations = bandit.predict_expectations(&ctx_test2[..], &mut rng);
    assert!(expectations.contains_key(&"B"));
}

#[test]
fn test_knearest_empty_history() {
    let underlying = LinGreedy::new(0.1, 1.0, 2);
    let policy = KNearest::new(underlying, 3, "manhattan");

    let bandit = Bandit::new(vec![1, 2, 3], policy).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    // Should work even with no historical data (random selection)
    let ctx = [1.0, 2.0];
    let pred = bandit.predict(&ctx[..], &mut rng);
    assert!(pred.is_ok());

    let expectations = bandit.predict_expectations(&ctx[..], &mut rng);
    assert_eq!(expectations.len(), 3);
}

#[test]
fn test_knearest_cosine_distance() {
    let underlying = LinGreedy::new(0.0, 1.0, 2); // Greedy for deterministic test
    let policy = KNearest::new(underlying, 1, "cosine");

    let mut bandit = Bandit::new(vec!["X", "Y"], policy).unwrap();
    let mut rng = StdRng::seed_from_u64(42);

    // Train with orthogonal vectors
    let ctx1 = [1.0, 0.0];
    bandit.partial_fit(&["X"], &ctx1[..], &[1.0]).unwrap();

    let ctx2 = [0.0, 1.0];
    bandit.partial_fit(&["Y"], &ctx2[..], &[1.0]).unwrap();

    // Predict with vector closer to [1, 0] in cosine similarity
    let ctx_test1 = [2.0, 0.1];
    let _pred = bandit.predict(&ctx_test1[..], &mut rng).unwrap();
    // Should select X since [2, 0.1] is more similar to [1, 0] than [0, 1]

    // Predict with vector closer to [0, 1] in cosine similarity
    let ctx_test2 = [0.1, 2.0];
    let _pred2 = bandit.predict(&ctx_test2[..], &mut rng).unwrap();
    // Should select Y since [0.1, 2] is more similar to [0, 1] than [1, 0]
}

#[test]
fn test_knearest_reset() {
    let underlying = LinGreedy::new(0.1, 1.0, 2);
    let mut policy = KNearest::new(underlying, 2, "euclidean");

    // Add some data
    let ctx1 = [1.0, 0.0];
    policy.update(&"A", &ctx1[..], 1.0);

    let ctx2 = [0.0, 1.0];
    policy.update(&"B", &ctx2[..], 0.5);

    // Reset specific arm
    policy.reset_arm(&"A");

    // Full reset
    policy.reset();
}
