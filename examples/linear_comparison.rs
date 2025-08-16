use rand::prelude::*;
use rand::rngs::StdRng;
use trashpanda::prelude::*;

/// Example demonstrating all three linear contextual bandit algorithms
fn main() {
    println!("Linear Contextual Bandit Algorithm Comparison");
    println!("==============================================");
    println!("Testing with 3 arms and 3-dimensional contexts");
    println!("True optimal arm depends on context\n");

    // Set up context-dependent rewards for 3 arms
    // Arm 0: Best for context feature 0
    // Arm 1: Best for context feature 1
    // Arm 2: Best for context feature 2
    let true_weights = vec![
        vec![0.8, 0.1, 0.1], // Arm 0 weights
        vec![0.1, 0.8, 0.1], // Arm 1 weights
        vec![0.1, 0.1, 0.8], // Arm 2 weights
    ];

    // Test LinUCB
    println!("\nLinUCB Results:");
    run_experiment("LinUCB", LinUcb::new(1.0, 1.0, 3), &true_weights);

    // Test LinTS
    println!("\nLinTS Results:");
    run_experiment("LinTS", LinTs::new(1.0, 1.0, 3), &true_weights);

    // Test LinGreedy
    println!("\nLinGreedy Results:");
    run_experiment("LinGreedy", LinGreedy::new(0.1, 1.0, 3), &true_weights);

    println!("\n==============================================");
    println!("Comparison Summary:");
    println!("- LinUCB: Uses upper confidence bounds for exploration");
    println!("- LinTS: Uses Thompson sampling from posterior distribution");
    println!("- LinGreedy: Uses epsilon-greedy with learned linear model");
    println!("\nAll algorithms should converge to similar performance,");
    println!("but with different exploration strategies.");
}

fn run_experiment<P>(name: &str, policy: P, true_weights: &[Vec<f64>])
where
    P: for<'a> Policy<i32, &'a [f64]>,
{
    let mut rng = StdRng::seed_from_u64(42);
    let mut bandit = Bandit::new(vec![0, 1, 2], policy).unwrap();
    let mut total_reward = 0.0;
    let mut optimal_selections = 0;
    let n_rounds = 1000;

    for round in 0..n_rounds {
        // Generate random context (one feature will be dominant)
        let mut context = vec![rng.random::<f64>() * 0.2; 3];
        let dominant_feature = rng.random_range(0..3);
        context[dominant_feature] = 0.8 + rng.random::<f64>() * 0.2;

        // Normalize context
        let sum: f64 = context.iter().sum();
        for c in &mut context {
            *c /= sum;
        }

        // Select arm
        let selected = bandit.predict(&context[..], &mut rng).unwrap();

        // Calculate reward (with noise)
        let mut reward = 0.0;
        for (i, &c) in context.iter().enumerate() {
            reward += true_weights[selected as usize][i] * c;
        }
        reward += rng.random::<f64>() * 0.1 - 0.05; // Add noise
        reward = reward.clamp(0.0, 1.0);

        // Update bandit using policy_mut()
        bandit.policy_mut().update(&selected, &context[..], reward);

        total_reward += reward;

        // Check if optimal arm was selected
        let optimal_arm = dominant_feature as i32;
        if selected == optimal_arm {
            optimal_selections += 1;
        }

        // Print progress at key rounds
        if name == "LinUCB" && (round == 10 || round == 100 || round == 500) {
            println!("  Round {}: Cumulative reward = {:.2}", round, total_reward);
        }
    }

    let avg_reward = total_reward / n_rounds as f64;
    let optimal_rate = optimal_selections as f64 / n_rounds as f64;

    println!("  Average reward: {:.4}", avg_reward);
    println!("  Optimal selection rate: {:.2}%", optimal_rate * 100.0);
    println!("  Total reward: {:.2}", total_reward);
}
