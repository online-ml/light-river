use num::pow::Pow;
use rand::prelude::*;

use num::{Float, FromPrimitive};
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::env::consts;
use std::iter::FlatMap;
use std::mem;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use std::rc::Rc;

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
/// Vecotor are the labels
struct Stats {
    sum: Vec<f64>,
    sq_sum: Vec<f64>,
    count: Vec<i32>,
}
impl Stats {
    fn create_result() {
        unimplemented!()
    }
    fn add() {
        unimplemented!()
    }
    fn merge() {
        unimplemented!()
    }
    fn predict_proba() {
        unimplemented!()
    }
}

/// Node struct
#[derive(Debug, Copy, Clone)]
struct Node<F> {
    parent: Option<usize>,
    tau: F, // Time parameter (?)
    is_leaf: bool,
    min_list: [F; 2], // Lists representing the minimum and maximum values of the data points contained in the current node
    max_list: [F; 2],
    delta: F, // Dimension in which a split occurs (?)
    xi: F,    // Split point along the dimension specified by delta
    left: Option<usize>,
    right: Option<usize>,
    // stats: Stats, // Ignoring stats for now since it should be a fixed-size array, vector should not work since we are using fixed-size arrays in Trees, but try it out and see what comes out
}
impl<F: FType> Node<F> {
    pub fn update_leaf(&mut self) {
        unimplemented!()
    }
    pub fn update_internal(&mut self) {
        unimplemented!()
    }
    pub fn get_parent_tau(&self) -> f64 {
        unimplemented!()
        // match self.parent {
        //     Some(ref parent) => parent.borrow().tau,
        //     None => 0.0,
        // }
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
            unimplemented!();
        }

        let node_default = Node::<F> {
            parent: None,
            tau: F::from_f64(0.33).unwrap(),
            is_leaf: false,
            min_list: [F::from_f64(0.1).unwrap(), F::from_f64(0.2).unwrap()],
            max_list: [F::from_f64(0.3).unwrap(), F::from_f64(0.4).unwrap()],
            delta: F::from_f64(0.123).unwrap(),
            xi: F::from_f64(0.456).unwrap(),
            left: None,
            right: None,
            // stats: Stats::new,
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
                nodes[i].is_leaf = true;
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
        features: Vec<String>,
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

    fn predict_proba(&self) {
        unimplemented!()
    }

    fn extend_mondrian_block(&self) {
        unimplemented!()
    }

    /// Function "learn_one"
    pub fn partial_fit(&self, x: &HashMap<String, f32>, y: &ClassifierTarget) {
        // No need to check if node is root, the full tree is already built
        self.extend_mondrian_block()
    }

    fn fit(&self) {
        unimplemented!()
    }

    // Function "score_one"
    pub fn predict(
        &mut self,
        x: &HashMap<String, Data<f32>>,
        y: &HashMap<String, Data<f32>>,
    ) -> Option<ClassifierOutput<F>> {
        unimplemented!()
    }

    fn get_params(&self) {
        unimplemented!()
    }
}
