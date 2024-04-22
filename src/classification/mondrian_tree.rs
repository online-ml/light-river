use crate::classification::alias::FType;
use crate::classification::mondrian_node::{Node, Stats};
use crate::common::{ClassifierOutput, ClassifierTarget, Observation};
use crate::stream::data_stream::Data;
use core::iter::zip;
use ndarray::array;
use ndarray::{Array1, Array2};
use ndarray::{ArrayBase, Dim, ScalarOperand, ViewRepr};
use num::pow::Pow;
use num::traits::float;
use num::{Float, FromPrimitive};
use rand::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::env::consts;
use std::iter::FlatMap;
use std::ops::{Add, Div, Mul, Sub};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use std::rc::Rc;
use std::rc::Weak;
use std::{clone, cmp, mem, usize};

#[derive(Clone)]
struct Tree<F: FType> {
    nodes: Vec<Node<F>>,
}
impl<F: FType> Tree<F> {
    fn new(features: &Vec<String>, rng: &mut ThreadRng) -> Self {
        let min_list = Array1::zeros(features.len());
        let max_list = Array1::zeros(features.len());

        println!("min_list {min_list:?}");

        // TODO: get this number. 51 is the number of subjects for keystrokes.
        let num_labels = 51;
        let feature_dim = features.len();

        let node_default = Node::<F> {
            parent: None,
            tau: F::zero(),
            is_leaf: false,
            min_list,
            max_list,
            delta: 0,
            xi: F::zero(),
            left: None,
            right: None,
            stats: Stats::new(num_labels, feature_dim),
        };
        let mut nodes = vec![node_default];

        Tree { nodes }
    }
}

#[derive(Clone)]
pub struct MondrianTree<F: FType> {
    window_size: usize,
    features: Vec<String>,
    rng: ThreadRng,
    tree: Tree<F>,
    first_learn: bool,
}
impl<F: FType> MondrianTree<F> {
    pub fn new(window_size: usize, features: &Vec<String>) -> Self {
        let features_clone = features.clone();
        let mut rng = rand::thread_rng();
        let mut tree = Tree::new(&features, &mut rng);
        MondrianTree::<F> {
            window_size,
            features: features_clone,
            rng,
            tree,
            first_learn: false,
        }
    }

    fn create_leaf(&self) {
        unimplemented!()
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one, so it's the same as "predict()".
    pub fn predict_proba(
        &self,
        // x: &HashMap<String, f32>,
        x: &Array1<F>,
        y: &ClassifierTarget,
    ) -> ClassifierOutput<F> {
        self.predict(x, 0, F::one())
    }

    fn extend_mondrian_block(&self) {
        println!("=== extend_mondrian_block not implemented")
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
        &self,
        x: &Array1<F>,
        node_idx: usize,
        p_not_separated_yet: F,
    ) -> ClassifierOutput<F> {
        let node = &self.tree.nodes[node_idx];

        // // Calculate the time delta from the parent node.
        // // If node is root its time is 0
        // let parent_tau: F = match node.parent {
        //     Some(_) => self.tree.nodes[node.parent.unwrap()].tau,
        //     None => F::from_f32(0.0).unwrap(),
        // };
        // let d = node.tau - parent_tau;

        // // Step 2: Compute the distance `eta` of `x` from the node's data boundaries.
        // // TODO: test it, I'm 70% sure this works.
        // let eta_min = (&node.max_list - x).mapv(|v| F::max(v, F::zero()));
        // let eta_max = (&node.max_list - x).mapv(|v| F::max(v, F::zero()));
        // let eta = eta_min.sum() + eta_max.sum();
        // println!("eta_min: {:?}", eta_min);
        // println!("eta_max: {:?}", eta_max);
        // println!("eta: {:?}", eta);

        // // Step 3: Calculate the probability `p` of not being separated by new splits.
        // let p = F::one() - (-d * eta).exp();

        // // Step 4: Generate a result for the current node using its statistics.
        // let result = node.stats.create_result(x, p_not_separated_yet * p);

        // if node.is_leaf() {
        //     let w = p_not_separated_yet * (F::one() - p);
        //     return result.merge(node.stats.create_result(x, w));
        // } else {
        //     let child_idx = if x[node.delta] <= node.xi {
        //         node.left
        //     } else {
        //         node.right
        //     };
        //     let child_result =
        //         self.predict(x, child_idx.unwrap(), p_not_separated_yet * (F::one() - p));
        //     return result.merge(child_result);
        // }

        ClassifierOutput::Probabilities(HashMap::from([(
            ClassifierTarget::from("target-example"),
            F::one(),
        )]))
    }

    fn get_params(&self) {
        unimplemented!()
    }
}
