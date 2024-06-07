use light_river::mondrian_forest::mondrian_forest::MondrianForestRegressor;

use light_river::common::{Regressor, RegressorTarget};
use light_river::datasets::synthetic_regression::SyntheticRegression;
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

fn get_dataset_size(transactions: IterCsv<f32, File>) -> usize {
    let mut length = 0;
    for _ in transactions {
        length += 1;
    }
    length
}

fn main() {
    let n_trees: usize = 1;

    let transactions_f = SyntheticRegression::load_data();
    let features = get_features(transactions_f);

    let mut mf: MondrianForestRegressor<f32> =
        MondrianForestRegressor::new(n_trees, features.len());
    let mut score_total = 0.0;

    let transactions_l = SyntheticRegression::load_data();
    let dataset_size = get_dataset_size(transactions_l);

    let now = Instant::now();

    let transactions = SyntheticRegression::load_data();
    for (idx, transaction) in transactions.enumerate() {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let x = Array1::<f32>::from_vec(features.iter().map(|k| x[k]).collect());

        let y = data.to_regression_target("label").unwrap();

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

        // if idx == 527 {
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
