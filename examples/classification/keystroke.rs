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

    // TODO: change it, probably they can be taken from "data.get_x" or somewhere there
    let features = vec!["F1".to_string(), "F2".to_string(), "F3".to_string()];

    let mut mt: MondrianTree<f32> = MondrianTree::new(window_size, n_trees, height, features);

    // DEBUG: remove it
    let mut counter = 0;

    // LOOP
    let transactions = Keystroke::load_data().unwrap();
    for transaction in transactions {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let y = data.to_classifier_target("subject").unwrap();

        mt.partial_fit(&x, &y);

        // let score = mt.update(&x, &y, true, false, &target_label).unwrap();

        // println!("=== Score: {:?}", score);
        println!("");

        counter += 1;
        if counter >= 3 {
            break;
        }
        // roc_auc.update(&score, &label, Some(1.));
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());
    // println!("ROCAUC: {:.2}%", roc_auc.get() * (100.0 as f32));
}
