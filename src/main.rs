#[allow(dead_code)]
#[allow(unused_imports)]
#[allow(unused_variables)]
#[allow(unused_mut)]
#[allow(unused_assignments)]
#[allow(unused_must_use)]
#[allow(unused_parens)]

use std::convert::TryFrom;
use std::collections::HashMap;
use rand::prelude::*;
use light_river::datasets::credit_card::CreditCard;


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

    fn new(height: u32, features: Vec<String>, rng: &mut ThreadRng) -> Self {

        // TODO: padding
        // TODO: handle non [0, 1] features
        // TODO: weighted sampling of features

        // #nodes = 2 ^ height - 1
        let n_nodes: usize = usize::try_from(u32::pow(2, height) - 1).unwrap();
        // #branches = 2 ^ (height - 1) - 1
        let n_branches = usize::try_from(u32::pow(2, height - 1) - 1).unwrap();
        // Allocate memory for the HST
        let mut hst = HST {
            feature: vec![String::from(""); usize::try_from(n_branches).unwrap()],
            threshold: vec![0.0; usize::try_from(n_branches).unwrap()],
            l_mass: vec![0.0; usize::try_from(n_nodes).unwrap()],
            r_mass: vec![0.0; usize::try_from(n_nodes).unwrap()],
        };
        // Randomly assign features and thresholds to each branch
        for branch in 0..n_branches {
            let feature = features.choose(rng).unwrap();
            hst.feature[branch] = feature.clone();
            hst.threshold[branch] = rng.gen();  // [0, 1]
        }
        hst
    }

    // fn get_index_of_feature(&mut self, name: String) -> u32 {
    //     // Use entry() to handle cases where the feature doesn't exist yet.
    //     let index = match self.feature_indices.entry(name.clone()) {
    //         std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
    //         std::collections::hash_map::Entry::Vacant(entry) => {
    //             let new_index = self.feature_indices.len() as u32;
    //             entry.insert(new_index);
    //             new_index
    //         }
    //     };
    //     index
    // }
}

fn main() {

    // PARAMETERS

    let window_size: u32 = 100;
    let n_trees: u32 = 100;
    let height: u32 = 3;
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
        String::from("Time")
    ];
    let mut rng = rand::thread_rng();

    // INITIALIZATION

    let mut trees: Vec<HST> = Vec::new();
    for _ in 0..n_trees {
        trees.push(HST::new(height, features.clone(), &mut rng));
    }

    // UPDATE

    let transactions = CreditCard::load_credit_card_transactions().unwrap();

    for (i, transaction) in transactions.enumerate() {
        let line = transaction.unwrap();

        println!("Data: {:?}", line.get_x());
        println!("Target: {:?}", line.get_y().unwrap());

        if i == 10 {
            break;
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
