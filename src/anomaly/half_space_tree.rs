// https://pastebin.com/ZLD6E5FT

use rand::prelude::*;

use num::{Float, FromPrimitive};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::mem;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use crate::common::{ClassifierOutput, ClassifierTarget, Observation};

use super::AnomalyDetector;

// Return the index of a node's left child node.
#[inline]
fn left_child(node: u32) -> u32 {
    node * 2 + 1
}

// Return the index of a node's right child node.
#[inline]
fn right_child(node: u32) -> u32 {
    node * 2 + 2
}

#[derive(Clone)]
struct Trees<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    feature: Vec<String>,
    threshold: Vec<F>,
    l_mass: Vec<F>,
    r_mass: Vec<F>,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> Trees<F> {
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
            threshold: init_vec(n_branches, F::zero()),
            l_mass: init_vec(n_nodes, F::zero()),
            r_mass: init_vec(n_nodes, F::zero()),
        };

        // Randomly assign features and thresholds to each branch
        for branch in 0..n_branches {
            let feature = features.choose(rng).unwrap();
            hst.feature[branch] = feature.clone();
            let random_threshold: f64 = rng.gen();
            hst.threshold[branch] = F::from_f64(random_threshold).unwrap(); // [0, 1]
        }
        hst
    }
}
/// Half-space trees are an online variant of isolation forests.
/// They work well when anomalies are spread out.
/// However, they do not work well if anomalies are packed together in windows.
/// By default, this implementation assumes that each feature has values that are comprised
/// between 0 and 1.
/// # Parameters
///
/// - `window_size`: The number of observations to consider when computing the score.
/// - `n_trees`: The number of trees to use.
/// - `height`: The height of each tree.
/// - `features`: The list of features to use. If `None`, the features will be inferred from the first observation.
///
/// # Example
///
/// ```
///
///
///
/// ```
pub struct HalfSpaceTree<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    window_size: u32,
    counter: u32,
    n_trees: u32,
    height: u32,
    features: Option<Vec<String>>,
    rng: ThreadRng,
    n_branches: u32,
    n_nodes: u32,
    trees: Option<Trees<F>>,
    first_learn: bool,
    pos_val: Option<ClassifierTarget>,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> HalfSpaceTree<F> {
    pub fn new(
        window_size: u32,
        n_trees: u32,
        height: u32,
        features: Option<Vec<String>>,
        pos_val: Option<ClassifierTarget>,
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
            pos_val: pos_val,
        }
    }

    pub fn update(
        &mut self,
        observation: &Observation<F>,
        do_score: bool,
        do_update: bool,
    ) -> Option<ClassifierOutput<F>> {
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

        let mut score: F = F::zero();

        for tree in 0..self.n_trees {
            let mut node: u32 = 0;
            for depth in 0..self.height {
                // Update the score
                let hst = &mut self.trees.as_mut().unwrap();

                // Flag for scoring
                if do_score {
                    score += hst.r_mass[(tree * self.n_nodes + node) as usize]
                        * F::from_u32(u32::pow(2, depth)).unwrap();
                }

                if do_update {
                    // Update the l_mass
                    hst.l_mass[(tree * self.n_nodes + node) as usize] += F::one();
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
                hst.l_mass.fill(F::zero());
                self.counter = 0;
            }
        }
        if do_score {
            score = F::one() - (score / self.max_score());

            return Some(ClassifierOutput::Probabilities(HashMap::from([(
                ClassifierTarget::from(
                    self.pos_val.clone().unwrap_or(ClassifierTarget::from(true)),
                ),
                score,
            )])));
            // return Some(score);
        }
        return None;
    }
    fn max_score(&self) -> F {
        F::from(self.n_trees).unwrap()
            * F::from(self.window_size).unwrap()
            * (F::from(2.).unwrap().powi(self.height as i32 + 1) - F::one())
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> AnomalyDetector<F>
    for HalfSpaceTree<F>
{
    fn learn_one(&mut self, observation: &Observation<F>) {
        self.update(observation, false, true);
    }
    fn score_one(&mut self, observation: &Observation<F>) -> Option<ClassifierOutput<F>> {
        self.update(observation, true, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::datasets::credit_card::CreditCard;
    use crate::stream::iter_csv::IterCsv;
    use std::fs::File;
    #[test]
    fn test_hst() {
        // PARAMETERS
        let window_size: u32 = 1000;
        let n_trees: u32 = 50;
        let height: u32 = 6;

        // INITIALIZATION
        let mut hst: HalfSpaceTree<f32> = HalfSpaceTree::new(
            window_size,
            n_trees,
            height,
            None,
            Some(ClassifierTarget::from("1".to_string())),
        );

        // LOOP
        let transactions: IterCsv<f32, File> = CreditCard::load_credit_card_transactions().unwrap();
        for transaction in transactions {
            let data = transaction.unwrap();
            let observation = data.get_observation();
            let label = data.get_y().unwrap().get("Class").unwrap();
            let _ = hst.update(&observation, true, true);
        }
    }
}

mod tests {
    #[test]
    fn test_left_child() {
        let node = 42;
        let child = left_child(node);
        assert_eq!(child, 85);
    }

    #[test]
    fn test_right_child() {
        let node = 42;
        let child = right_child(node);
        assert_eq!(child, 86);
    }
}
