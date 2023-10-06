// https://pastebin.com/ZLD6E5FT

use rand::prelude::*;

use std::convert::TryFrom;
use std::mem;

use crate::common::Observation;

// Return the index of a node's left child node.
fn left_child(node: u32) -> u32 {
    node * 2 + 1
}

// Return the index of a node's right child node.
fn right_child(node: u32) -> u32 {
    node * 2 + 2
}

#[derive(Clone)]
struct Trees {
    feature: Vec<String>,
    threshold: Vec<f32>,
    l_mass: Vec<f32>,
    r_mass: Vec<f32>,
}

impl Trees {
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
        let mut hst = Trees {
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
pub struct HalfSpaceTree {
    window_size: u32,
    counter: u32,
    n_trees: u32,
    height: u32,
    features: Option<Vec<String>>,
    rng: ThreadRng,
    n_branches: u32,
    n_nodes: u32,
    trees: Option<Trees>,
    first_learn: bool,
}
impl HalfSpaceTree {
    pub fn new(
        window_size: u32,
        n_trees: u32,
        height: u32,
        features: Option<Vec<String>>,
        // rng: ThreadRng,
    ) -> Self {
        // let mut rng = rand::thread_rng();
        let n_branches = u32::pow(2, height - 1) - 1;
        let n_nodes = u32::pow(2, height) - 1;

        let features_clone = features.clone();
        let mut rng = rand::thread_rng();
        let trees = if let Some(features) = features {
            Some(Trees::new(n_trees, height, &features, &mut rng))
        } else {
            None
        };
        HalfSpaceTree {
            window_size: window_size,
            counter: 0,
            n_trees: n_trees,
            height: height,
            features: features_clone,
            rng: rng,
            n_branches: n_branches,
            n_nodes: n_nodes,
            trees: trees,
            first_learn: false,
        }
    }

    pub fn update(
        &mut self,
        observation: &Observation<f32>,
        do_score: bool,
        do_update: bool,
    ) -> Option<f32> {
        // build trees during the first pass
        if (!self.first_learn) && self.features.is_none() {
            self.features = Some(observation.clone().into_keys().collect());
            self.trees = Some(Trees::new(
                self.n_trees,
                self.height,
                &self.features.as_ref().unwrap(),
                &mut self.rng,
            ));
            self.first_learn = true;
        }

        let mut score: f32 = 0.0;

        for tree in 0..self.n_trees {
            let mut node: u32 = 0;
            for depth in 0..self.height {
                // Update the score
                let hst = &mut self.trees.as_mut().unwrap();

                // Flag for scoring
                if do_score {
                    score += hst.r_mass[(tree * self.n_nodes + node) as usize]
                        * u32::pow(2, depth) as f32;
                }

                if do_update {
                    // Update the l_mass
                    hst.l_mass[(tree * self.n_nodes + node) as usize] += 1.0;
                }

                // Stop if the node is a leaf or stop early if the mass of the node is too small
                if depth == self.height - 1 {
                    break;
                }

                // Get the feature and threshold of the current node so that we can determine
                // whether to go left or right
                let feature = &hst.feature[(tree * self.n_branches + node) as usize];
                let threshold = hst.threshold[(tree * self.n_branches + node) as usize];

                // Get the value of the current feature
                // node = self.walk(observation, node, tree, threshold, feature, hst);
                node = match observation.get(feature) {
                    Some(value) => {
                        // Update the mass of the current node
                        if *value < threshold {
                            left_child(node)
                        } else {
                            right_child(node)
                        }
                    }
                    None => {
                        // If the feature is missing, go down both branches and select the node with the
                        // the biggest l_mass
                        if hst.l_mass[(tree * self.n_nodes + left_child(node)) as usize]
                            > hst.l_mass[(tree * self.n_nodes + right_child(node)) as usize]
                        {
                            left_child(node)
                        } else {
                            right_child(node)
                        }
                    }
                };
            }
        }
        if do_update {
            // Pivot if the window is full
            let hst = &mut self.trees.as_mut().unwrap();
            self.counter += 1;
            if self.counter == self.window_size {
                mem::swap(&mut hst.r_mass, &mut hst.l_mass);
                hst.l_mass.fill(0.0);
                self.counter = 0;
            }
        }
        if do_score {
            return Some(score);
        }
        return None;
    }
    pub fn learn_one(&mut self, observation: &Observation<f32>) {
        self.update(observation, false, true);
    }
    pub fn score_one(&mut self, observation: &Observation<f32>) -> Option<f32> {
        self.update(observation, true, false)
    }
}
