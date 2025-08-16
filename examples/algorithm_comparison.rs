use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use trashpanda::{
    Bandit,
    simple::{epsilon_greedy::EpsilonGreedy, random::Random, thompson::ThompsonSampling, ucb::Ucb},
};

fn main() {
    println!("TrashPanda: Multi-Armed Bandit Algorithm Comparison\n");
    println!("{}", "=".repeat(60));

    // Define the true reward probabilities for each arm
    let true_rewards = HashMap::from([
        ("Arm A", 0.3),
        ("Arm B", 0.5),
        ("Arm C", 0.8), // Best arm
        ("Arm D", 0.4),
    ]);

    println!("True reward probabilities:");
    for (arm, prob) in &true_rewards {
        println!("  {}: {:.2}", arm, prob);
    }
    println!("\nBest arm: Arm C (0.80)\n");
    println!("{}", "=".repeat(60));

    // Run simulation for each algorithm
    let arms = vec!["Arm A", "Arm B", "Arm C", "Arm D"];

    // Test Random algorithm
    {
        println!("\nRandom");
        println!("{}", "-".repeat(6));

        let mut bandit = Bandit::new(arms.clone(), Random::default()).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut total_reward = 0.0;
        let mut arm_selections = HashMap::new();

        // Simulate 1000 rounds
        for _ in 0..1000 {
            // Select an arm
            let selected_arm = bandit.predict_simple(&mut rng).unwrap();

            // Track selection
            *arm_selections.entry(selected_arm).or_insert(0) += 1;

            // Generate reward based on true probability
            let true_prob = true_rewards[selected_arm];
            let reward = if rng.random::<f64>() < true_prob {
                1.0
            } else {
                0.0
            };

            total_reward += reward;

            // Update the bandit
            bandit.fit_simple(&[selected_arm], &[reward]).unwrap();
        }

        // Report results
        println!("  Total reward: {:.1}/1000", total_reward);
        println!("  Average reward: {:.3}", total_reward / 1000.0);
        println!("  Arm selection counts:");

        let mut selections: Vec<_> = arm_selections.iter().collect();
        selections.sort_by_key(|&(arm, _)| *arm);

        for (arm, count) in selections {
            println!("    {}: {} ({:.1}%)", arm, count, (*count as f64 / 10.0));
        }

        // Show final expectations
        let mut rng_exp = rand::rngs::StdRng::seed_from_u64(123);
        let expectations = bandit.predict_expectations_simple(&mut rng_exp);
        println!("  Final arm probabilities:");

        let mut exp_vec: Vec<_> = expectations.iter().collect();
        exp_vec.sort_by_key(|&(arm, _)| *arm);

        for (arm, prob) in exp_vec {
            println!("    {}: {:.3}", arm, prob);
        }
    }

    // Test Epsilon-Greedy algorithm
    {
        println!("\nEpsilon-Greedy (ε=0.1)");
        println!("{}", "-".repeat(23));

        let mut bandit = Bandit::new(arms.clone(), EpsilonGreedy::new(0.1)).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut total_reward = 0.0;
        let mut arm_selections = HashMap::new();

        for _ in 0..1000 {
            let selected_arm = bandit.predict_simple(&mut rng).unwrap();
            *arm_selections.entry(selected_arm).or_insert(0) += 1;
            let true_prob = true_rewards[selected_arm];
            let reward = if rng.random::<f64>() < true_prob {
                1.0
            } else {
                0.0
            };
            total_reward += reward;
            bandit.fit_simple(&[selected_arm], &[reward]).unwrap();
        }

        println!("  Total reward: {:.1}/1000", total_reward);
        println!("  Average reward: {:.3}", total_reward / 1000.0);
        println!("  Arm selection counts:");
        let mut selections: Vec<_> = arm_selections.iter().collect();
        selections.sort_by_key(|&(arm, _)| *arm);
        for (arm, count) in selections {
            println!("    {}: {} ({:.1}%)", arm, count, (*count as f64 / 10.0));
        }
        let mut rng_exp = rand::rngs::StdRng::seed_from_u64(123);
        let expectations = bandit.predict_expectations_simple(&mut rng_exp);
        println!("  Final arm probabilities:");
        let mut exp_vec: Vec<_> = expectations.iter().collect();
        exp_vec.sort_by_key(|&(arm, _)| *arm);
        for (arm, prob) in exp_vec {
            println!("    {}: {:.3}", arm, prob);
        }
    }

    // Test UCB1 algorithm
    {
        println!("\nUCB1 (α=1.414)");
        println!("{}", "-".repeat(14));

        let mut bandit = Bandit::new(arms.clone(), Ucb::new(1.414)).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut total_reward = 0.0;
        let mut arm_selections = HashMap::new();

        for _ in 0..1000 {
            let selected_arm = bandit.predict_simple(&mut rng).unwrap();
            *arm_selections.entry(selected_arm).or_insert(0) += 1;
            let true_prob = true_rewards[selected_arm];
            let reward = if rng.random::<f64>() < true_prob {
                1.0
            } else {
                0.0
            };
            total_reward += reward;
            bandit.fit_simple(&[selected_arm], &[reward]).unwrap();
        }

        println!("  Total reward: {:.1}/1000", total_reward);
        println!("  Average reward: {:.3}", total_reward / 1000.0);
        println!("  Arm selection counts:");
        let mut selections: Vec<_> = arm_selections.iter().collect();
        selections.sort_by_key(|&(arm, _)| *arm);
        for (arm, count) in selections {
            println!("    {}: {} ({:.1}%)", arm, count, (*count as f64 / 10.0));
        }
        let mut rng_exp = rand::rngs::StdRng::seed_from_u64(123);
        let expectations = bandit.predict_expectations_simple(&mut rng_exp);
        println!("  Final arm probabilities:");
        let mut exp_vec: Vec<_> = expectations.iter().collect();
        exp_vec.sort_by_key(|&(arm, _)| *arm);
        for (arm, prob) in exp_vec {
            println!("    {}: {:.3}", arm, prob);
        }
    }

    // Test Thompson Sampling algorithm
    {
        println!("\nThompson Sampling");
        println!("{}", "-".repeat(17));

        let mut bandit = Bandit::new(arms, ThompsonSampling::new()).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut total_reward = 0.0;
        let mut arm_selections = HashMap::new();

        for _ in 0..1000 {
            let selected_arm = bandit.predict_simple(&mut rng).unwrap();
            *arm_selections.entry(selected_arm).or_insert(0) += 1;
            let true_prob = true_rewards[selected_arm];
            let reward = if rng.random::<f64>() < true_prob {
                1.0
            } else {
                0.0
            };
            total_reward += reward;
            bandit.fit_simple(&[selected_arm], &[reward]).unwrap();
        }

        println!("  Total reward: {:.1}/1000", total_reward);
        println!("  Average reward: {:.3}", total_reward / 1000.0);
        println!("  Arm selection counts:");
        let mut selections: Vec<_> = arm_selections.iter().collect();
        selections.sort_by_key(|&(arm, _)| *arm);
        for (arm, count) in selections {
            println!("    {}: {} ({:.1}%)", arm, count, (*count as f64 / 10.0));
        }
        let mut rng_exp = rand::rngs::StdRng::seed_from_u64(123);
        let expectations = bandit.predict_expectations_simple(&mut rng_exp);
        println!("  Final arm probabilities:");
        let mut exp_vec: Vec<_> = expectations.iter().collect();
        exp_vec.sort_by_key(|&(arm, _)| *arm);
        for (arm, prob) in exp_vec {
            println!("    {}: {:.3}", arm, prob);
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("\nAnalysis:");
    println!("- Random: Selects uniformly, achieves ~0.5 average (theoretical: 0.5)");
    println!("- Epsilon-Greedy: Quickly identifies best arm, balances exploration");
    println!("- UCB1: Systematically explores all arms initially, then exploits");
    println!(
        "- Thompson Sampling: Probabilistic approach, naturally balances exploration/exploitation"
    );
    println!("\nAll algorithms except Random converge to preferring Arm C (the optimal choice).");
}
