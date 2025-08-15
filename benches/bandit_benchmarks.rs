use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use trashpanda::{
    Bandit,
    policies::{EpsilonGreedy, Random},
};

fn bench_arm_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("arm_operations");

    // Benchmark arm lookup with different numbers of arms
    for n_arms in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("has_arm", n_arms), n_arms, |b, &n| {
            let arms: Vec<i32> = (0..n).collect();
            let bandit = Bandit::new(arms.clone(), Random).unwrap();
            let test_arm = n / 2; // Middle arm

            b.iter(|| black_box(bandit.has_arm(&test_arm)));
        });

        group.bench_with_input(BenchmarkId::new("add_arm", n_arms), n_arms, |b, &n| {
            b.iter_batched(
                || {
                    let arms: Vec<i32> = (0..n).collect();
                    Bandit::new(arms, Random).unwrap()
                },
                |mut bandit| black_box(bandit.add_arm(n + 1)),
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("remove_arm", n_arms), n_arms, |b, &n| {
            b.iter_batched(
                || {
                    let arms: Vec<i32> = (0..n).collect();
                    Bandit::new(arms, Random).unwrap()
                },
                |mut bandit| black_box(bandit.remove_arm(&(n / 2))),
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");

    for n_arms in [10, 100, 1000].iter() {
        // Random policy prediction
        group.bench_with_input(
            BenchmarkId::new("random_predict", n_arms),
            n_arms,
            |b, &n| {
                let arms: Vec<i32> = (0..n).collect();
                let bandit = Bandit::new(arms, Random).unwrap();
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);

                b.iter(|| black_box(bandit.predict_with_rng(&mut rng).unwrap()));
            },
        );

        // Epsilon-greedy prediction
        group.bench_with_input(
            BenchmarkId::new("epsilon_greedy_predict", n_arms),
            n_arms,
            |b, &n| {
                let arms: Vec<i32> = (0..n).collect();
                let mut bandit = Bandit::new(arms.clone(), EpsilonGreedy::new(0.1)).unwrap();

                // Train with some data
                let decisions: Vec<i32> = (0..100).map(|i| i % n).collect();
                let rewards: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
                bandit.fit(&decisions, &rewards).unwrap();

                let mut rng = rand::rngs::StdRng::seed_from_u64(42);

                b.iter(|| black_box(bandit.predict_with_rng(&mut rng).unwrap()));
            },
        );

        // Expectations calculation
        group.bench_with_input(
            BenchmarkId::new("predict_expectations", n_arms),
            n_arms,
            |b, &n| {
                let arms: Vec<i32> = (0..n).collect();
                let mut bandit = Bandit::new(arms.clone(), EpsilonGreedy::new(0.1)).unwrap();

                // Train with some data
                let decisions: Vec<i32> = (0..100).map(|i| i % n).collect();
                let rewards: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
                bandit.fit(&decisions, &rewards).unwrap();

                b.iter(|| black_box(bandit.predict_expectations().unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");

    for n_samples in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("fit", n_samples), n_samples, |b, &n| {
            let arms = vec![1, 2, 3, 4, 5];
            let decisions: Vec<i32> = (0..n).map(|i| arms[i % arms.len()]).collect();
            let rewards: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();

            b.iter_batched(
                || Bandit::new(arms.clone(), EpsilonGreedy::new(0.1)).unwrap(),
                |mut bandit| black_box(bandit.fit(&decisions, &rewards).unwrap()),
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(
            BenchmarkId::new("partial_fit", n_samples),
            n_samples,
            |b, &n| {
                let arms = vec![1, 2, 3, 4, 5];
                let batch_size = 100;
                let n_batches = n / batch_size;

                b.iter_batched(
                    || {
                        let bandit = Bandit::new(arms.clone(), EpsilonGreedy::new(0.1)).unwrap();
                        let decisions: Vec<i32> = (0..n).map(|i| arms[i % arms.len()]).collect();
                        let rewards: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
                        (bandit, decisions, rewards)
                    },
                    |(mut bandit, decisions, rewards)| {
                        for i in 0..n_batches {
                            let start = i * batch_size;
                            let end = ((i + 1) * batch_size).min(n);
                            black_box(
                                bandit
                                    .partial_fit(&decisions[start..end], &rewards[start..end])
                                    .unwrap(),
                            );
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_builder(c: &mut Criterion) {
    c.bench_function("builder_construction", |b| {
        let arms = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        b.iter(|| {
            black_box(
                Bandit::builder()
                    .arms(arms.clone())
                    .policy(EpsilonGreedy::new(0.1))
                    .build()
                    .unwrap(),
            )
        });
    });

    c.bench_function("direct_construction", |b| {
        let arms = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        b.iter(|| black_box(Bandit::new(arms.clone(), EpsilonGreedy::new(0.1)).unwrap()));
    });
}

criterion_group!(
    benches,
    bench_arm_operations,
    bench_prediction,
    bench_training,
    bench_builder
);
criterion_main!(benches);
