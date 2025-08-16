use rand::SeedableRng;
use rand::rngs::StdRng;
use trashpanda::{Bandit, contextual::lingreedy::LinGreedy, neighborhood::knearest::KNearest};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("K-Nearest Neighbors Contextual Bandit Demo");
    println!("==========================================\n");

    // Create a KNearest policy with LinGreedy as the underlying algorithm
    let underlying = LinGreedy::new(0.0, 1.0, 2); // epsilon=0 (pure greedy), 2 features
    let policy = KNearest::new(underlying, 3, "euclidean"); // k=3 nearest neighbors

    // Initialize bandit with three products
    let mut bandit = Bandit::new(vec!["Product A", "Product B", "Product C"], policy)?;
    let mut rng = StdRng::seed_from_u64(42);

    println!("Training Phase:");
    println!("--------------");

    // Training data: Different user segments prefer different products
    // Create training contexts outside loops to avoid lifetime issues
    let young_contexts: Vec<Vec<f64>> = (0..5).map(|i| vec![0.9 + i as f64 * 0.02, 0.1]).collect();

    let middle_contexts: Vec<Vec<f64>> = (0..5).map(|i| vec![0.5, 0.5 + i as f64 * 0.02]).collect();

    let senior_contexts: Vec<Vec<f64>> = (0..5).map(|i| vec![0.1, 0.9 + i as f64 * 0.02]).collect();

    // Young users (high feature 1) prefer Product A
    println!("Young users prefer Product A:");
    for ctx in &young_contexts {
        println!("  Context: {:?} -> Product A (reward: 0.9)", ctx);
        bandit.partial_fit(&["Product A"], ctx.as_slice(), &[0.9])?;
    }

    // Middle-aged users (balanced features) prefer Product B
    println!("\nMiddle-aged users prefer Product B:");
    for ctx in &middle_contexts {
        println!("  Context: {:?} -> Product B (reward: 0.85)", ctx);
        bandit.partial_fit(&["Product B"], ctx.as_slice(), &[0.85])?;
    }

    // Senior users (high feature 2) prefer Product C
    println!("\nSenior users prefer Product C:");
    for ctx in &senior_contexts {
        println!("  Context: {:?} -> Product C (reward: 0.95)", ctx);
        bandit.partial_fit(&["Product C"], ctx.as_slice(), &[0.95])?;
    }

    println!("\n\nPrediction Phase:");
    println!("----------------");

    // Test with new user contexts
    let test_contexts = vec![
        (vec![0.95, 0.05], "Young user"),
        (vec![0.1, 0.95], "Senior user"),
        (vec![0.5, 0.5], "Middle-aged user"),
        (vec![0.7, 0.3], "Young-ish user"),
        (vec![0.3, 0.7], "Senior-ish user"),
    ];

    for (context, description) in test_contexts {
        let prediction = bandit.predict(context.as_slice(), &mut rng)?;
        let expectations = bandit.predict_expectations(context.as_slice(), &mut rng);

        println!("\n{} (context: {:?}):", description, context);
        println!("  Recommended: {}", prediction);
        println!("  Expected rewards:");
        for (product, reward) in expectations.iter() {
            println!("    {}: {:.3}", product, reward);
        }
    }

    println!("\n\nHow it works:");
    println!("-------------");
    println!("1. KNearest finds the k=3 most similar historical contexts");
    println!("2. It trains a temporary LinGreedy policy on just those observations");
    println!("3. The temporary policy makes the prediction for the new context");
    println!("4. This allows local learning - each prediction uses only relevant data");

    Ok(())
}
