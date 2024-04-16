use light_river::classification::mondrian_tree::MondrianTree;
use light_river::common::ClassifierOutput;
use light_river::common::ClassifierTarget;
use light_river::datasets::keystroke::Keystroke;
use light_river::metrics::rocauc::ROCAUC;
use light_river::metrics::traits::ClassificationMetric;
use light_river::stream::data_stream::DataStream;
use light_river::stream::iter_csv::IterCsv;
use ndarray::Array1;
use std::borrow::Borrow;
use std::fs::File;
use std::time::Instant;

/// Get list of features of the dataset.
///
/// e.g. features: ["H.e", "UD.t.i", "H.i", ...]
fn get_features(transactions: IterCsv<f32, File>) -> Vec<String> {
    // TODO: pass transaction file by reference, in main use only one "Keystroke::load_data().unwrap()"
    let sample = transactions.into_iter().next();
    let observation = sample.unwrap().unwrap().get_observation();
    observation.iter().map(|(k, _)| k.clone()).collect()
}

fn main() {
    // DEBUG: remove it
    let mut counter = 0;
    let now = Instant::now();
    let window_size: usize = 1000;
    let n_trees: usize = 1;
    let height: usize = 10;

    let transactions_f = Keystroke::load_data().unwrap();
    let features = get_features(transactions_f);

    let mut mt: MondrianTree<f32> = MondrianTree::new(window_size, n_trees, height, &features);

    let transactions = Keystroke::load_data().unwrap();
    for transaction in transactions {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let y = data.to_classifier_target("subject").unwrap();

        mt.partial_fit(&x, &y);

        let x_ord_vec: Vec<f32> = features.iter().map(|k| x[k]).collect();
        let x_ord_arr = Array1::<f32>::from_vec(x_ord_vec);
        let score = mt.predict_proba(&x_ord_arr, &y);

        // println!("=== Score: {:?}", score);
        // println!("");

        // counter += 1;
        // if counter >= 3 {
        //     break;
        // }
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());
    // println!("ROCAUC: {:.2}%", roc_auc.get() * (100.0 as f32));
}
