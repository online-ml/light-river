// TODO: padding
// TODO: handle non [0, 1] features
// TODO: weighted sampling of features
// TODO: early stopping in walk

use csv::WriterBuilder;
use light_river::datasets::credit_card::CreditCard;
use light_river::stream::data_stream::{Data, DataStream};
use rand::prelude::*;

#[allow(dead_code)]
#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_mut)]
#[allow(unused_assignments)]
#[allow(unused_must_use)]
#[allow(unused_parens)]
use std::convert::TryFrom;
use std::fs::File;
use std::mem;
use std::time::SystemTime;

// Return the index of a node's left child node.
fn left_child(node: u32) -> u32 {
    node * 2 + 1
}

// Return the index of a node's right child node.
fn right_child(node: u32) -> u32 {
    node * 2 + 2
}

#[derive(Clone)]
struct HST {
    feature: Vec<String>,
    threshold: Vec<f32>,
    l_mass: Vec<f32>,
    r_mass: Vec<f32>,
}

impl HST {
    fn new(n_trees: u32, height: u32, features: &Vec<String>, rng: &mut ThreadRng) -> Self {
        // #nodes = 2 ^ height - 1
        let n_nodes: usize = usize::try_from(n_trees * (u32::pow(2, height) - 1)).unwrap();
        // #branches = 2 ^ (height - 1) - 1
        let n_branches = usize::try_from(n_trees * (u32::pow(2, height - 1) - 1)).unwrap();

        // Helper function to create and populate a Vec with a given capacity
        fn init_vec<T>(capacity: usize, default_value: T) -> Vec<T>
        where
            T: Clone,
        {
            let mut vec = Vec::with_capacity(capacity);
            vec.resize(capacity, default_value);
            vec
        }

        // Allocate memory for the HST
        let mut hst = HST {
            feature: init_vec(n_branches, String::from("")),
            threshold: init_vec(n_branches, 0.0),
            l_mass: init_vec(n_nodes, 0.0),
            r_mass: init_vec(n_nodes, 0.0),
        };

        // Randomly assign features and thresholds to each branch
        for branch in 0..n_branches {
            let feature = features.choose(rng).unwrap();
            hst.feature[branch] = feature.clone();
            hst.threshold[branch] = rng.gen(); // [0, 1]
        }
        hst
    }
}

fn main() {
    // Create a CSV writer
    let file = File::create("scores.csv");
    let mut csv_writer = WriterBuilder::new()
        .has_headers(false) // If you don't want headers
        .from_writer(file.unwrap());

    // PARAMETERS

    let window_size: u32 = 1000;
    let mut counter: u32 = 0;
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
    let mut rng = rand::thread_rng();

    // INITIALIZATION

    let mut hst = HST::new(n_trees, height, &features, &mut rng);
    let n_nodes = u32::pow(2, height) - 1;
    let n_branches = u32::pow(2, height - 1) - 1;

    // LOOP
    // We collect the dataset in a vector of DataStream we won't to take the reading csv time into account
    let transactions = CreditCard::load_credit_card_transactions().unwrap();
    let mut dataset: Vec<DataStream<f32>> = Vec::new();

    for transaction in transactions {
        dataset.push(transaction.unwrap());
    }

    let start = SystemTime::now();
    for line in dataset.iter() {
        let mut score: f32 = 0.0;
        for tree in 0..n_trees {
            let mut node: u32 = 0;
            for depth in 0..height {
                // Update the score
                score += hst.r_mass[(tree * n_nodes + node) as usize] * u32::pow(2, depth) as f32;

                // Update the l_mass
                hst.l_mass[(tree * n_nodes + node) as usize] += 1.0;

                // Stop if the node is a leaf or stop early if the mass of the node is too small
                if depth == height - 1 {
                    break;
                }

                // Get the feature and threshold of the current node so that we can determine
                // whether to go left or right
                let feature = &hst.feature[(tree * n_branches + node) as usize];
                let threshold = hst.threshold[(tree * n_branches + node) as usize];

                // Get the value of the current feature
                node = match line.get_x().get(feature) {
                    Some(Data::Scalar(value)) => {
                        // Update the mass of the current node
                        if *value < threshold {
                            left_child(node)
                        } else {
                            right_child(node)
                        }
                    }
                    Some(Data::String(_)) => panic!("String feature not supported yet"),
                    None => {
                        // If the feature is missing, go down both branches and select the node with the
                        // the biggest l_mass
                        if hst.l_mass[(tree * n_nodes + left_child(node)) as usize]
                            > hst.l_mass[(tree * n_nodes + right_child(node)) as usize]
                        {
                            left_child(node)
                        } else {
                            right_child(node)
                        }
                    }
                };
            }
        }

        // Pivot if the window is full
        counter += 1;
        if counter == window_size {
            mem::swap(&mut hst.r_mass, &mut hst.l_mass);
            hst.l_mass.fill(0.0);
            counter = 0;
        }

        // Output score to CSV
        let _ = csv_writer.serialize(score);
    }

    let _ = csv_writer.flush();

    match start.elapsed() {
        Ok(elapsed) => {
            // it prints '2'
            println!("{}", elapsed.as_millis());
        }
        Err(e) => {
            // an error occurred!
            println!("Error: {e:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_left_child() {
        let node: u32 = 42;
        let left = left_child(node);
        assert_eq!(left, 85);
    }

    #[test]
    fn test_right_child() {
        let node: u32 = 42;
        let left = right_child(node);
        assert_eq!(left, 86);
    }
}
