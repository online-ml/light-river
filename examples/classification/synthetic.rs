use light_river::classification::alias::FType;
use light_river::classification::mondrian_forest::MondrianForest;
use light_river::classification::mondrian_tree::MondrianTree;
use light_river::common::ClassifierOutput;
use light_river::common::ClassifierTarget;
use light_river::datasets::synthetic::Synthetic;
use light_river::metrics::rocauc::ROCAUC;
use light_river::metrics::traits::ClassificationMetric;
use light_river::stream::data_stream::DataStream;
use light_river::stream::iter_csv::IterCsv;
use ndarray::{s, Array1};
use std::borrow::Borrow;
use std::fs::File;
use std::time::Instant;

/// Get list of features of the dataset.
///
/// e.g. features: ["H.e", "UD.t.i", "H.i", ...]
fn get_features(transactions: IterCsv<f32, File>) -> Vec<String> {
    let sample = transactions.into_iter().next();
    let observation = sample.unwrap().unwrap().get_observation();
    let mut out: Vec<String> = observation.iter().map(|(k, _)| k.clone()).collect();
    out.sort();
    out
}

fn get_labels(transactions: IterCsv<f32, File>) -> Vec<String> {
    let mut labels = vec![];
    for t in transactions {
        let data = t.unwrap();
        // TODO: use instead 'to_classifier_target' and a vector of 'ClassifierTarget'
        let target = data.get_y().unwrap()["label"].to_string();
        if !labels.contains(&target) {
            labels.push(target);
        }
    }
    labels
}

fn main() {
    let now = Instant::now();
    let window_size: usize = 1000;
    let n_trees: usize = 1;

    let transactions_f = Synthetic::load_data().unwrap();
    let features = get_features(transactions_f);

    let transactions_c = Synthetic::load_data().unwrap();
    let labels = get_labels(transactions_c);
    println!("labels: {labels:?}, features: {features:?}");
    let mut mf: MondrianForest<f32> = MondrianForest::new(window_size, n_trees, &features, &labels);

    let transactions = Synthetic::load_data().unwrap();
    for transaction in transactions {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let y = data.to_classifier_target("label").unwrap();
        // TODO: generalize to non-classification only by implementing 'ClassifierTarget'
        // instead of taking directly the string.
        let y = match y {
            ClassifierTarget::String(y) => y,
            _ => unimplemented!(),
        };
        let x_ord = Array1::<f32>::from_vec(features.iter().map(|k| x[k]).collect());

        println!("=M=1 partial_fit {x_ord}");
        mf.partial_fit(&x_ord, &y);

        println!("=M=2 predict_proba");
        let probs = mf.predict_proba(&x_ord);

        println!("=M=3 probs: {:?}", probs.to_vec());

        let score = mf.score(&x_ord, &y);
        println!("=M=4 score: {:?}", score);

        println!("");
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());
    // println!("ROCAUC: {:.2}%", roc_auc.get() * (100.0 as f32));
}
