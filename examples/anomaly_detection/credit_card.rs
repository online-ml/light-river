use light_river::anomaly::half_space_tree::HalfSpaceTree;
use light_river::common::ClassifierOutput;
use light_river::common::ClassifierTarget;
use light_river::datasets::credit_card::CreditCard;
use light_river::metrics::rocauc::ROCAUC;
use light_river::metrics::traits::ClassificationMetric;
use light_river::stream::data_stream::DataStream;
use light_river::stream::iter_csv::IterCsv;
use std::fs::File;
use std::time::Instant;

fn main() {
    let now = Instant::now();

    // PARAMETERS
    let window_size: u32 = 1000;
    let n_trees: u32 = 50;
    let height: u32 = 6;
    let pos_val_metric = ClassifierTarget::from("1".to_string());
    let pos_val_tree = pos_val_metric.clone();
    let mut roc_auc: ROCAUC<f32> = ROCAUC::new(Some(10), pos_val_metric.clone());
    // INITIALIZATION
    let mut hst: HalfSpaceTree<f32> =
        HalfSpaceTree::new(window_size, n_trees, height, None, Some(pos_val_tree));

    // LOOP
    let transactions: IterCsv<f32, File> = CreditCard::load_credit_card_transactions().unwrap();
    for transaction in transactions {
        let data = transaction.unwrap();
        let observation = data.get_observation();
        let label = data.to_classifier_target("Class").unwrap();
        let score = hst.update(&observation, true, true).unwrap();
        // println!("Label: {:?}", label);
        // println!("Score: {:?}", score);
        // roc_auc.update(&score, &label, Some(1.));
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());
    println!("ROCAUC: {:.2}%", roc_auc.get() * (100.0 as f32));
}
