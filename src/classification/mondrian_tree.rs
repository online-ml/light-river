use core::iter::zip;
use ndarray::{Array1, Array2};
use num::pow::Pow;
use num::traits::float;
use rand::prelude::*;

use num::{Float, FromPrimitive};
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::env::consts;
use std::iter::FlatMap;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use std::rc::Rc;
use std::{cmp, mem, usize};

use crate::common::{ClassifierOutput, ClassifierTarget, Observation};
use crate::stream::data_stream::Data;

trait FType:
    Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign + std::fmt::Debug
{
}
impl<T> FType for T where
    T: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign + std::fmt::Debug
{
}

/// Stats assocociated to one node
///
/// In nel215 code it is "Classifier"
#[derive(Debug, Clone)]
struct Stats<F> {
    sums: Array2<F>,
    sq_sums: Array1<F>,
    counts: Array1<usize>,
}
impl<F: FType> Stats<F> {
    fn new(num_labels: usize, feature_dim: usize) -> Self {
        Stats {
            sums: Array2::zeros((num_labels, feature_dim)),
            sq_sums: Array1::zeros(num_labels),
            counts: Array1::zeros(num_labels),
        }
    }
    fn create_result(&self, x: &Array1<F>, w: F) -> ClassifierOutput<F> {
        // unimplemented!();
        let probabilities = self.predict_proba(x);
        let mut results = HashMap::new();
        for (index, &prob) in probabilities.iter().enumerate() {
            results.insert(ClassifierTarget::from(index.to_string()), prob * w);
        }
        ClassifierOutput::Probabilities(results)
    }
    fn add(&mut self, x: &Array1<F>, label: usize) {
        unimplemented!()
        // for (s, &xi) in self.sums[label].iter_mut().zip(x.iter()) {
        //     *s += xi;
        // }
        // self.sq_sums[label] += x.iter().map(|&xi| xi * xi).sum();
        // self.counts[label] += 1;
    }
    fn merge(&mut self, other: &Stats<F>) {
        unimplemented!()
        // for (i, (self_sum, self_sq_sum, self_count)) in self.sums.iter_mut().zip(self.sq_sums.iter_mut()).zip(self.counts.iter_mut()).enumerate() {
        //     for (s, &o) in self_sum.iter_mut().zip(other.sums[i].iter()) {
        //         *s += o;
        //     }
        //     *self_sq_sum += other.sq_sums[i];
        //     *self_count += other.counts[i];
        // }
    }
    fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        unimplemented!()
        // let mut probabilities = Array1::zeros(self.sums.len());
        // let mut sum_prob = F::zero();

        // for (index, ((sum, &sq_sum), &count)) in self.sums.outer_iter().zip(self.sq_sums.iter()).zip(self.counts.iter()).enumerate() {
        //     if count > 1 {
        //         // Check if Array has better functions than these horrible things
        //         let mean = sum.iter().zip(count.to_f64().unwrap()).map(|(s, c)| *s / F::from(c).unwrap()).collect::<Vec<F>>();
        //         let var = sq_sum / F::from(count).unwrap() - mean.iter().map(|m| *m * *m).sum::<F>() + F::epsilon();
        //         let sigma = count.to_f64().unwrap() * var.to_f64().unwrap() / (count.to_f64().unwrap() - F::one() + F::epsilon());
        //         let norm_factor = (F::from(2.0 * std::f64::consts::PI).unwrap() * F::from(sigma).unwrap()).sqrt();
        //         let exp_term = x.iter().zip(mean.iter()).map(|(&xi, &mi)| {
        //             let diff = xi - mi;
        //             diff * diff / F::from(2.0 * sigma).unwrap()
        //         }).sum::<F>();
        //         let prob = (-exp_term).exp() / norm_factor;

        //         probabilities[index] = prob;
        //         sum_prob += prob;
        //     }
        // }

        // for prob in probabilities.iter_mut() {
        //     *prob /= sum_prob;
        // }

        // probabilities
    }
}

/// Node struct
#[derive(Debug, Clone)]
struct Node<F> {
    parent: Option<usize>,
    tau: F, // Time parameter: updated during 'node creation' or 'node update'
    is_active: bool,
    min_list: Array1<F>, // Lists representing the minimum and maximum values of the data points contained in the current node
    max_list: Array1<F>,
    delta: usize, // Dimension in which a split occurs (?)
    xi: F,        // Split point along the dimension specified by delta
    left: Option<usize>,
    right: Option<usize>,
    stats: Stats<F>,
}
impl<F: FType> Node<F> {
    pub fn update_leaf(&self) {
        unimplemented!()
    }
    pub fn update_internal(&self) {
        unimplemented!()
    }
    pub fn get_parent_tau(&self, parent: Option<&Node<F>>) -> F {
        panic!(
            "Not implemented, adds a lot of complexity for no reason. Just extract tau directly."
        )
        // match self.parent {
        //     Some(_) => parent.tau,
        //     None => F::from_f32(0.0).unwrap(),
        // }
    }
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

struct Trees<F: FType> {
    nodes: Vec<Node<F>>,
}
impl<F: FType> Trees<F> {
    fn new(
        n_trees: usize,
        height: usize,
        features: &Vec<String>,
        rng: &mut ThreadRng,
        n_nodes: usize,
    ) -> Self {
        if n_trees != 1 {
            unimplemented!("Only implemented for 1 tree. This code has to be heavily restructured for multiple trees.");
        }

        // e.g. [0.0, 0.0, 0.0, ...]
        let min_list_vec: Vec<F> = features.iter().map(|_| F::from_f32(0.0).unwrap()).collect();
        let min_list = Array1::<F>::from_vec(min_list_vec);
        let max_list_vec: Vec<F> = features.iter().map(|_| F::from_f32(0.0).unwrap()).collect();
        let max_list = Array1::<F>::from_vec(max_list_vec);
        // TODO: get this number. 51 is the number of subjects for keystrokes.
        let num_labels = 51;
        let feature_dim = features.len();

        let node_default = Node::<F> {
            parent: None,
            tau: F::zero(),
            is_active: false,
            min_list,
            max_list,
            delta: 0,
            xi: F::zero(),
            left: None,
            right: None,
            stats: Stats::new(num_labels, feature_dim),
        };
        let mut nodes = vec![node_default; n_nodes];

        // For each node assign indicies of: parent, and left/right child
        for i in 0..n_nodes {
            let left_idx = 2 * i + 1;
            let right_idx = 2 * i + 2;

            if (left_idx < n_nodes) && (right_idx < n_nodes) {
                nodes[i].left = Some(left_idx);
                nodes[i].right = Some(right_idx);
                nodes[left_idx].parent = Some(i);
                nodes[right_idx].parent = Some(i);
            } else {
                nodes[i].is_active = true;
            }
        }

        Trees { nodes }
    }
}

pub struct MondrianTree<F: FType> {
    window_size: usize,
    n_trees: usize,
    height: usize,
    features: Vec<String>,
    rng: ThreadRng,
    n_nodes: usize,
    trees: Trees<F>,
    first_learn: bool,
    // pos_val: ClassifierTarget,
}
impl<F: FType> MondrianTree<F> {
    pub fn new(
        window_size: usize,
        n_trees: usize,
        height: usize,
        features: &Vec<String>,
        // pos_val: ClassifierTarget,
    ) -> Self {
        let features_clone = features.clone();
        let mut rng = rand::thread_rng();
        // #nodes = 2 ^ height - 1
        let n_nodes = usize::pow(2, height.try_into().unwrap()) - 1;
        // TODO: this is only 1 tree, implement later for mulpile ones
        let mut trees = Trees::new(n_trees, height, &features, &mut rng, n_nodes);
        MondrianTree::<F> {
            window_size,
            n_trees,
            height,
            features: features_clone,
            rng,
            n_nodes,
            trees,
            first_learn: false,
            // pos_val,
        }
    }

    fn create_leaf(&self) {
        unimplemented!()
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one, so it's the same as "predict()".
    pub fn predict_proba(
        &mut self,
        // x: &HashMap<String, f32>,
        x: &Array1<F>,
        y: &ClassifierTarget,
    ) -> ClassifierOutput<F> {
        self.predict(x, 0, F::one())
    }

    fn extend_mondrian_block(&self) {
        // println!("WARNING: extend_mondrian_block not implemented")
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one.
    ///
    /// Function in River/LightRiver: "learn_one()"
    pub fn partial_fit(&mut self, x: &HashMap<String, f32>, y: &ClassifierTarget) {
        // No need to check if node is root, the full tree is already built
        self.extend_mondrian_block();
    }

    fn fit(&self) {
        unimplemented!()
    }

    /// Function in River/LightRiver: "score_one()"
    ///
    /// Recursive function to predict probabilities.
    /// - `x`: Input data.
    /// - `node_idx`: Current node index in the tree.
    /// - `p_not_separated_yet`: Probability that `x` has not been separated by any split in the tree up to this node.
    fn predict(
        &mut self,
        x: &Array1<F>,
        node_idx: usize,
        p_not_separated_yet: F,
    ) -> ClassifierOutput<F> {
        let node = &self.trees.nodes[node_idx];

        // Calculate the time delta from the parent node.
        // If node is root its time is 0
        let parent_tau: F = match node.parent {
            Some(_) => self.trees.nodes[node.parent.unwrap()].tau,
            None => F::from_f32(0.0).unwrap(),
        };
        let d = node.tau - parent_tau;

        // Step 2: Compute the distance `eta` of `x` from the node's data boundaries.
        // TODO: test it, I'm 70% sure this works.
        let eta_min = (&node.max_list - x).mapv(|v| F::max(v, F::zero()));
        let eta_max = (&node.max_list - x).mapv(|v| F::max(v, F::zero()));
        let eta = eta_min.sum() + eta_max.sum();
        println!("eta_min: {:?}", eta_min);
        println!("eta_max: {:?}", eta_max);
        println!("eta: {:?}", eta);

        // Step 3: Calculate the probability `p` of not being separated by new splits.
        let p = F::one() - (-d * eta).exp();

        // Step 4: Generate a result for the current node using its statistics.
        let result = node.stats.create_result(x, p_not_separated_yet * p);

        if node.is_leaf() {
            let w = p_not_separated_yet * (F::one() - p);
            return result.merge(node.stats.create_result(x, w));
        } else {
            let child_idx = if x[node.delta] <= node.xi {
                node.left
            } else {
                node.right
            };
            let child_result =
                self.predict(x, child_idx.unwrap(), p_not_separated_yet * (F::one() - p));
            return result.merge(child_result);
        }

        ClassifierOutput::Probabilities(HashMap::from([(
            ClassifierTarget::from("target-example"),
            F::one(),
        )]))
    }

    fn get_params(&self) {
        unimplemented!()
    }
}
