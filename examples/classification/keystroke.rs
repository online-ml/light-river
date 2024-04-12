use light_river::classification::mondrian_tree::MondrianTree;
use light_river::common::ClassifierOutput;
use light_river::common::ClassifierTarget;
use light_river::datasets::keystroke::Keystroke;
use light_river::metrics::rocauc::ROCAUC;
use light_river::metrics::traits::ClassificationMetric;
use light_river::stream::data_stream::DataStream;
use light_river::stream::iter_csv::IterCsv;
use std::fs::File;
use std::time::Instant;

fn main() {
    let now = Instant::now();

    let window_size: usize = 1000;
    let n_trees: usize = 1;
    let height: usize = 4;

    // TODO: Check if still need for classification or it was useful only in anomany detection
    let pos_val_metric = ClassifierTarget::from("1".to_string());
    let pos_val_tree = pos_val_metric.clone();

    let features = vec!["F1".to_string(), "F2".to_string(), "F3".to_string()];

    // INITIALIZATION
    let mut mt: MondrianTree<f32> =
        MondrianTree::new(window_size, n_trees, height, features, pos_val_tree);

    // DEBUG: remove it
    let mut counter = 0;

    // LOOP
    let transactions: IterCsv<f32, File> = Keystroke::load_data().unwrap();
    for transaction in transactions {
        let data = transaction.unwrap();
        println!("Data: {data}");
        let observation = data.get_observation();
        // println!("Observation: {:?}", observation);
        let label = data.to_classifier_target("subject").unwrap();
        // let score = mt.update(&observation, true, true).unwrap();

        // Label: No idea why we it
        // println!("Label: {:?}", label);
        // println!("Score: {:?}", score);
        // println!("");

        counter += 1;
        if counter > 10 {
            break;
        }
        // roc_auc.update(&score, &label, Some(1.));
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());
    // println!("ROCAUC: {:.2}%", roc_auc.get() * (100.0 as f32));
}
