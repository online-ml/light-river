use light_river::common::ClassifierTarget;
use light_river::datasets::keystroke::Keystroke;
use light_river::mondrian_forest::mondrian_forest::{MondrianForest, MondrianForestClassifier};

use light_river::stream::iter_csv::IterCsv;
use ndarray::Array1;

use std::fs::File;
use std::time::Instant;

/// Get list of features of the dataset.
///
/// e.g. features: ["H.e", "UD.t.i", "H.i", ...]
fn get_features(transactions: IterCsv<f32, File>) -> Vec<String> {
    // TODO: pass transaction file by reference, in main use only one "Keystroke::load_data().unwrap()"
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
        let target = data.get_y().unwrap()["subject"].to_string();
        if !labels.contains(&target) {
            labels.push(target);
        }
    }
    labels
}

fn main() {
    // DEBUG: remove it
    let mut counter = 0;
    let now = Instant::now();
    let window_size: usize = 1000;
    let n_trees: usize = 1;

    let transactions_f = Keystroke::load_data().unwrap();
    let features = get_features(transactions_f);
    // DEBUG: remove it
    // let features = features[0..2].to_vec();

    let transactions_c = Keystroke::load_data().unwrap();
    let labels = get_labels(transactions_c);
    // DEBUG: remove it
    // let labels = labels[0..3].to_vec();
    println!("labels: {labels:?}");
    let mut mf: MondrianForestClassifier<f32> =
        MondrianForestClassifier::new(n_trees, features.len(), labels.len());

    let transactions = Keystroke::load_data().unwrap();
    for transaction in transactions {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let y = data.to_classifier_target("subject").unwrap();
        // TODO: generalize to non-classification only by implementing 'ClassifierTarget'
        // instead of taking directly the string.
        let y = match y {
            ClassifierTarget::String(y) => y,
            _ => unimplemented!(),
        };
        let y = labels.clone().iter().position(|l| l == &y).unwrap();

        let x_ord = Array1::<f32>::from_vec(features.iter().map(|k| x[k]).collect());
        // DEBUG: remove it
        // let x_ord = x_ord.slice(s![0..2]).to_owned();

        println!("=M=1 partial_fit");
        mf.partial_fit(&x_ord, y);

        println!("=M=2 predict_proba");
        let score = mf.predict_proba(&x_ord);

        println!("=M=3 score: {:?}", score);
        println!("");

        counter += 1;
        if counter >= 3 {
            break;
        }
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());
    // println!("ROCAUC: {:.2}%", roc_auc.get() * (100.0 as f32));
}
