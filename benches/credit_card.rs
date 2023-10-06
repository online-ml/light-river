use criterion::{criterion_group, criterion_main, Criterion};

use light_river::anomaly::half_space_tree::HalfSpaceTree;
use light_river::datasets::credit_card::CreditCard;

fn run_credit_card() {
    // PARAMETERS

    let window_size: u32 = 1000;
    let n_trees: u32 = 50;
    let height: u32 = 6;
    let features: Vec<String> = vec![
        String::from("V1"),
        String::from("V2"),
        String::from("V3"),
        String::from("V4"),
        String::from("V5"),
        String::from("V6"),
        String::from("V7"),
        String::from("V8"),
        String::from("V9"),
        String::from("V10"),
        String::from("V11"),
        String::from("V12"),
        String::from("V13"),
        String::from("V14"),
        String::from("V15"),
        String::from("V16"),
        String::from("V17"),
        String::from("V18"),
        String::from("V19"),
        String::from("V20"),
        String::from("V21"),
        String::from("V22"),
        String::from("V23"),
        String::from("V24"),
        String::from("V25"),
        String::from("V26"),
        String::from("V27"),
        String::from("V28"),
        String::from("Amount"),
        String::from("Time"),
    ];
    // let mut rng = rand::thread_rng();

    // INITIALIZATION

    let mut hst = HalfSpaceTree::new(window_size, n_trees, height, Some(features));
    // let n_nodes = u32::pow(2, height) - 1;
    // let n_branches = u32::pow(2, height - 1) - 1;

    // LOOP
    // We collect the dataset in a vector of DataStream we won't to take the reading csv time into account
    let transactions = CreditCard::load_credit_card_transactions().unwrap();

    for transaction in transactions {
        // let observation: Observation<f32> = transaction.unwrap().get_observation();
        let observation = transaction.unwrap().get_observation();
        let _ = hst.update(&observation, true, true);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("credit_card", |b| b.iter(|| run_credit_card()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
