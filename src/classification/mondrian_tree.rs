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
use rand_distr::{Distribution, Exp};
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::env::consts;
use std::fmt;
use std::iter::FlatMap;
use std::ops::{Add, Div, Mul, Sub};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use std::rc::Rc;
use std::rc::Weak;
use std::{clone, cmp, mem, usize};

#[derive(Clone)]
pub struct MondrianTree<F: FType> {
    window_size: usize,
    features: Vec<String>,
    labels: Vec<String>,
    rng: ThreadRng,
    first_learn: bool,
    nodes: Vec<Node<F>>,
}
impl<F: FType + fmt::Display> fmt::Display for MondrianTree<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(f, "┌ MondrianTree")?;
        write!(f, "│ window_size: {}", self.window_size)?;
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(f)?;
            write!(f, "│ │ Node {}: left = {:?}, right = {:?}, parent = {:?}, tau = {}, is_leaf = {}, min = {:?}, max = {:?}", i, node.left, node.right, node.parent, node.tau,  node.is_leaf, node.min_list.to_vec(), node.max_list.to_vec())?;
        }
        Ok(())
    }
}
impl<F: FType> MondrianTree<F> {
    pub fn new(window_size: usize, features: &Vec<String>, labels: &Vec<String>) -> Self {
        let mut rng = rand::thread_rng();
        let nodes = vec![];
        MondrianTree::<F> {
            window_size,
            features: features.clone(),
            labels: labels.clone(),
            rng,
            first_learn: false,
            nodes,
        }
    }

    fn create_leaf(&mut self, x: &Array1<F>, label: &String, parent: Option<usize>) {
        let min_list: ArrayBase<ndarray::OwnedRepr<F>, Dim<[usize; 1]>> =
            Array1::zeros(self.features.len());
        let max_list = Array1::zeros(self.features.len());

        let num_labels = self.labels.len();
        let feature_dim = self.features.len();
        let labels = self.labels.clone();

        let mut node = Node::<F> {
            parent,
            tau: F::from(1e9).unwrap(), // Very large value for tau
            is_leaf: true,
            min_list,
            max_list,
            delta: 0,
            xi: F::zero(),
            left: None,
            right: None,
            stats: Stats::new(num_labels, feature_dim),
        };
        // TODO: check if this works:
        // labels: ["s002", "s003", "s004"]
        let label_idx = labels.iter().position(|l| l == label).unwrap();
        node.update_leaf(x, label_idx);
        self.nodes.push(node);
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one, so it's the same as "predict()".
    pub fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        let root = &self.nodes[0];
        self.predict(x, root, F::one())
    }

    fn extend_mondrian_block(&mut self, node_idx: usize, x: &Array1<F>, label: &String) {
        // TODO: Check if we access the node somewhere else by reference (&Node).
        // If so pass it by ref here instead of 'node_idx' so we don't access it twice.
        let node = &self.nodes[node_idx];
        let e_min = (&node.min_list - x).mapv(|v| v.max(F::zero()));
        let e_max = (x - &node.max_list).mapv(|v| v.max(F::zero()));
        let e_sum = &e_min + &e_max;
        let rate = e_sum.sum() + F::epsilon();
        let exp_dist = Exp::new(rate.to_f32().unwrap()).unwrap();
        let E = F::from_f32(exp_dist.sample(&mut self.rng)).unwrap();
        println!("=== rate: {}, E: {}", rate, E);
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one.
    ///
    /// Function in River/LightRiver: "learn_one()"
    pub fn partial_fit(&mut self, x: &Array1<F>, y: &String) {
        if self.nodes.len() == 0 {
            self.create_leaf(x, y, None);
        } else {
            self.extend_mondrian_block(0, x, y);
        }
    }

    fn fit(&self) {
        unimplemented!("Make the program first work with 'partial_fit', then implement this")
    }

    /// Function in River/LightRiver: "score_one()"
    ///
    /// Recursive function to predict probabilities.
    fn predict(&self, x: &Array1<F>, node: &Node<F>, p_not_separated_yet: F) -> Array1<F> {
        // println!("predict_proba() - MondrianTree: {}", self);

        // Step 1: Calculate the time delta from the parent node.
        // If node is root its time is 0
        let parent_tau: F = match node.parent {
            Some(p) => self.nodes[p].tau,
            None => F::zero(),
        };
        let d = node.tau - parent_tau;

        // Step 2: If 'x' is outside the box, calculate distance of 'x' from the box
        let dist_max = (x - &node.max_list).mapv(|v| F::max(v, F::zero()));
        let dist_min = (&node.min_list - x).mapv(|v| F::max(v, F::zero()));
        let eta = dist_min.sum() + dist_max.sum();
        // It works, but check again once 'max_list' and 'min_list' are not 0s
        // println!("x: {:?}, node.max_list {:?}, max(max_list) {:?}, node.min_list {:?}, max(min_list) {:?}",
        //  x.to_vec(), node.max_list.to_vec(), dist_max.to_vec(), node.min_list.to_vec(), dist_min.to_vec());

        // Step 3: Probability 'p' of the box not splitting.
        //     eta (box dist): larger distance, more prob of splitting
        //     d (time diff with parent): more dist with parent, more prob of splitting
        let p = F::one() - (-d * eta).exp();

        // Step 4: Generate a result for the current node using its statistics.
        let res = node.stats.create_result(x, p_not_separated_yet * p);

        // DEBUG: Shadowing with bogous values since output would be simply [1.0, 0.0, 0.0]
        // let res = Array1::from_vec(vec![
        //     F::from_f32(0.7).unwrap(),
        //     F::from_f32(0.2).unwrap(),
        //     F::from_f32(0.1).unwrap(),
        // ]);
        // let p_not_separated_yet = F::from_f32(0.8).unwrap();
        // let p = F::from_f32(0.9).unwrap();

        // println!(
        //     "predict() - res: {:?}, p_not_separated_yet: {:?}, p: {:?}",
        //     res, p_not_separated_yet, p
        // );

        if node.is_leaf {
            let w = p_not_separated_yet * (F::one() - p);
            let res2 = node.stats.create_result(x, w);
            // Check sum of probabililties is 1
            let sum_res = ((&res + &res2).sum() - F::one()).abs();
            assert!(
                sum_res < F::from_f32(0.001).unwrap(),
                "Sum of probs should be 1."
            );
            return res + res2;
        } else {
            let child_idx = if x[node.delta] <= node.xi {
                node.left
            } else {
                node.right
            };
            let node = &self.nodes[child_idx.unwrap()];
            let child_res = self.predict(x, &node, p_not_separated_yet * (F::one() - p));
            return res + child_res;
        }
    }

    fn get_params(&self) {
        unimplemented!()
    }
}
