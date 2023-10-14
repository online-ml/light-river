use light_river::anomaly::half_space_tree::HalfSpaceTree;
use light_river::datasets::credit_card::CreditCard;
use light_river::stream::iter_csv::IterCsv;
use std::fs::File;
use std::time::Instant;

fn main() {
    let now = Instant::now();

    // PARAMETERS
    let window_size: u32 = 1000;
    let n_trees: u32 = 50;
    let height: u32 = 6;

    // INITIALIZATION
    let mut hst: HalfSpaceTree<f32> = HalfSpaceTree::new(window_size, n_trees, height, None);

    // LOOP
    let transactions: IterCsv<f32, File> = CreditCard::load_credit_card_transactions().unwrap();
    for transaction in transactions {
        let observation = transaction.unwrap().get_observation();
        let _ = hst.update(&observation, true, true);
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());
}
