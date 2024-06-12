use light_river::mondrian_forest::mondrian_forest::MondrianForestClassifier;

use light_river::common::{Classifier, ClfTarget};
use light_river::datasets::synthetic::Synthetic;
use light_river::stream::iter_csv::IterCsv;
use ndarray::Array1;
use num::ToPrimitive;

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
        // TODO: use instead 'to_classifier_target' and a vector of 'ClfTarget'
        let target = data.get_y().unwrap()["label"].to_string();
        if !labels.contains(&target) {
            labels.push(target);
        }
    }
    labels
}

fn get_dataset_size(transactions: IterCsv<f32, File>) -> usize {
    let mut length = 0;
    for _ in transactions {
        length += 1;
    }
    length
}

fn main() {
    let n_trees: usize = 1;

    let transactions_f = Synthetic::load_data();
    let features = get_features(transactions_f);

    let transactions_c = Synthetic::load_data();
    let labels = get_labels(transactions_c);
    println!("labels: {labels:?}, features: {features:?}");
    let mut mf: MondrianForestClassifier<f32> =
        MondrianForestClassifier::new(n_trees, features.len(), labels.len());
    let mut score_total = 0.0;

    let transactions_l = Synthetic::load_data();
    let dataset_size = get_dataset_size(transactions_l);

    let now = Instant::now();

    let transactions = Synthetic::load_data();
    for (idx, transaction) in transactions.enumerate() {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let x = Array1::<f32>::from_vec(features.iter().map(|k| x[k]).collect());

        let y = data.to_classifier_target("label").unwrap();
        let y = match y {
            ClfTarget::String(y) => y,
            _ => unimplemented!(),
        };
        let y = labels.clone().iter().position(|l| l == &y).unwrap();
        let y = ClfTarget::from(y);
        // println!("=M=1 x:{}, idx: {}", x, idx);

        // Skip first sample since tree has still no node
        if idx != 0 {
            let score = mf.predict_one(&x, &y);
            score_total += score;
            // println!(
            //     "Accuracy: {} / {} = {}",
            //     score_total,
            //     dataset_size - 1,
            //     score_total / idx.to_f32().unwrap()
            // );
        }

        // if idx == 4 {
        //     break;
        // }

        mf.learn_one(&x, &y);
    }

    let elapsed_time = now.elapsed();
    println!("Took {}ms", elapsed_time.as_millis());

    println!(
        "Accuracy: {} / {} = {}",
        score_total,
        dataset_size - 1,
        score_total / (dataset_size - 1).to_f32().unwrap()
    );

    let forest_size = mf.get_forest_size();
    println!("Forest tree sizes: {:?}", forest_size);
}
