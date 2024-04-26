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
use std::collections::HashSet;
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
    root: Option<usize>,
}

impl<F: FType + fmt::Display> fmt::Display for MondrianTree<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "┌ MondrianTree")?;
        writeln!(f, "│ window_size: {}", self.window_size)?;
        self.recursive_repr(self.root, f, "│ ")
    }
}

impl<F: FType + fmt::Display> MondrianTree<F> {
    /// Helper method to recursively format node details.
    fn recursive_repr(
        &self,
        node_idx: Option<usize>,
        f: &mut fmt::Formatter<'_>,
        prefix: &str,
    ) -> fmt::Result {
        if let Some(idx) = node_idx {
            let node = &self.nodes[idx];
            writeln!(f, "{}Node {}: left={:?}, right={:?}, parent={:?}, tau={:.3}, is_leaf={}, min={:?}, max={:?}",
                     prefix, idx, node.left, node.right, node.parent, node.tau, node.is_leaf, node.min_list.to_vec(), node.max_list.to_vec())?;

            self.recursive_repr(node.left, f, &(prefix.to_owned() + "│ "))?;
            self.recursive_repr(node.right, f, &(prefix.to_owned() + "│ "))?;
        }
        Ok(())
    }
}

impl<F: FType> MondrianTree<F> {
    pub fn new(window_size: usize, features: &Vec<String>, labels: &Vec<String>) -> Self {
        MondrianTree::<F> {
            window_size,
            features: features.clone(),
            labels: labels.clone(),
            rng: rand::thread_rng(),
            first_learn: false,
            nodes: vec![],
            root: None,
        }
    }

    fn create_leaf(&mut self, x: &Array1<F>, label: &String, parent: Option<usize>) -> usize {
        let num_labels = self.labels.len();
        let feature_dim = self.features.len();

        let mut node = Node::<F> {
            parent,
            tau: F::from(1e9).unwrap(), // Very large value
            is_leaf: true,
            min_list: x.clone(),
            max_list: x.clone(),
            delta: 0,
            xi: F::zero(),
            left: None,
            right: None,
            stats: Stats::new(num_labels, feature_dim),
        };

        let label_idx = self.labels.clone().iter().position(|l| l == label).unwrap();
        node.update_leaf(x, label_idx);
        self.nodes.push(node);
        let node_idx = self.nodes.len() - 1;
        node_idx
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one, so it's the same as "predict()".
    pub fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        let root = 0;
        self.test_tree();
        self.predict(x, root, F::one())
    }

    fn test_tree(&self) {
        for node_idx in 0..self.nodes.len() {
            // TODO: check if self.root is None, if so tree should be empty
            if node_idx == self.root.unwrap() {
                // Root node
                assert!(self.nodes[node_idx].parent.is_none(), "Root has a parent.");
            } else {
                // Non-root node
                assert!(
                    !self.nodes[node_idx].parent.is_none(),
                    "Non-root node has no parent"
                )
            }
        }

        let children_l: Vec<usize> = self.nodes.iter().filter_map(|node| node.left).collect();
        let children_r: Vec<usize> = self.nodes.iter().filter_map(|node| node.right).collect();
        let children = [children_l.clone(), children_r.clone()].concat();
        let mut seen = HashSet::new();
        let has_duplicates = children.iter().any(|item| !seen.insert(item));
        assert!(
            !has_duplicates,
            "Multiple nodes share one child. Children left: {:?}, Children right: {:?}",
            children_l, children_r
        );
    }

    fn extend_mondrian_block(&mut self, node_idx: usize, x: &Array1<F>, label: &String) -> usize {
        // Collect necessary values for computations
        let parent_tau = self.get_parent_tau(node_idx);
        let tau = self.nodes[node_idx].tau;
        // TODO: 'node_min_list' and 'node_max_list' be accessible without cloning
        let node_min_list = self.nodes[node_idx].min_list.clone();
        let node_max_list = self.nodes[node_idx].max_list.clone();

        let e_min = (&node_min_list - x).mapv(|v| F::max(v, F::zero()));
        let e_max = (x - &node_max_list).mapv(|v| F::max(v, F::zero()));
        // e_sum: size of the box [x_size, y_size]
        let e_sum = &e_min + &e_max;
        // 'rate' is lambda
        let rate = e_sum.sum() + F::epsilon();
        let exp_dist = Exp::new(rate.to_f32().unwrap()).unwrap();
        // 'exp_sample' is 'E' in nel215 code
        let exp_sample = F::from_f32(exp_dist.sample(&mut self.rng)).unwrap();
        // DEBUG: shadowing with Exp expected value
        let exp_sample = F::one() / rate;

        if parent_tau + exp_sample < tau {
            let cumsum = e_sum
                .iter()
                .scan(F::zero(), |acc, &x| {
                    *acc = *acc + x;
                    Some(*acc)
                })
                .collect::<Array1<F>>();
            // DEBUG: shadowing with expected value
            let e_sample = F::from_f32(self.rng.gen::<f32>()).unwrap() * e_sum.sum();
            let e_sample = F::from_f32(0.5).unwrap() * e_sum.sum();
            let delta = cumsum.iter().position(|&val| val > e_sample).unwrap_or(0);

            let (lower_bound, upper_bound) = if x[delta] > node_min_list[delta] {
                (
                    node_min_list[delta].to_f32().unwrap(),
                    x[delta].to_f32().unwrap(),
                )
            } else {
                (
                    x[delta].to_f32().unwrap(),
                    node_max_list[delta].to_f32().unwrap(),
                )
            };
            let xi = F::from_f32(self.rng.gen_range(lower_bound..upper_bound)).unwrap();
            // DEBUG: setting expected value
            let xi = F::from_f32((lower_bound + upper_bound) / 2.0).unwrap();

            let mut min_list = node_min_list;
            let mut max_list = node_max_list;
            min_list.zip_mut_with(x, |a, &b| *a = F::min(*a, b));
            max_list.zip_mut_with(x, |a, &b| *a = F::max(*a, b));

            // Create and push new parent node
            let parent_node = Node {
                parent: self.nodes[node_idx].parent,
                tau: parent_tau + exp_sample,
                is_leaf: false,
                min_list,
                max_list,
                delta,
                xi,
                left: None,
                right: None,
                stats: Stats::new(self.labels.len(), self.features.len()),
            };
            println!(
                "extend_mondrian_block() - mid if - grandpa: {:?}",
                self.nodes[node_idx].parent
            );

            self.nodes.push(parent_node);
            let parent_idx = self.nodes.len() - 1;
            let sibling_idx = self.create_leaf(x, label, Some(parent_idx));

            // Set the children appropriately
            if x[delta] <= xi {
                // Grandpa: self.nodes[node_idx].parent
                // (new) Parent: parent_idx
                // Child: node_idx
                // (new) Sibling: sibling_idx
                self.nodes[parent_idx].left = Some(sibling_idx);
                self.nodes[parent_idx].right = Some(node_idx);
            } else {
                self.nodes[parent_idx].left = Some(node_idx);
                self.nodes[parent_idx].right = Some(sibling_idx);
            }

            self.nodes[node_idx].parent = Some(parent_idx);

            self.update_internal(parent_idx);

            println!(
                "extend_mondrian_block() - mid if - parent: {:?}, child: {:?}",
                parent_idx, node_idx
            );

            println!("extend_modnrian_block() - post if");
            return parent_idx;
        } else {
            let node = &mut self.nodes[node_idx];
            node.min_list.zip_mut_with(x, |a, b| *a = F::min(*a, *b));
            node.max_list.zip_mut_with(x, |a, b| *a = F::max(*a, *b));
            if !node.is_leaf {
                println!(
                    "extend_mondrian_block() - mid else - is_leaf - delta: {:?}, xi: {:?}",
                    node.delta, node.xi
                );
                // TODO: understand how to make the following without making Rust angry with borrowing rules
                // node.left = Some(self.extend_mondrian_block(node.left, x, label));
                if x[node.delta] <= node.xi {
                    let node_left = node.left.unwrap();
                    let node_left_new = Some(self.extend_mondrian_block(node_left, x, label));
                    let node = &mut self.nodes[node_idx];
                    node.left = node_left_new;
                } else {
                    let node_right = node.right.unwrap();
                    let node_right_new = Some(self.extend_mondrian_block(node_right, x, label));
                    let node = &mut self.nodes[node_idx];
                    node.right = node_right_new;
                };
                self.update_internal(node_idx);
            } else {
                println!(
                    "extend_mondrian_block() - mid else - is_leaf NOT - delta: {:?}, xi: {:?}",
                    node.delta, node.xi
                );
                let label_idx = self.labels.iter().position(|l| l == label).unwrap();
                node.update_leaf(x, label_idx);
            }
            println!("extend_modnrian_block() - post else");
            return node_idx;
        }
    }

    fn update_internal(&mut self, node_idx: usize) {
        // In nel215 code update_internal is not called for the children, check if it's needed
        let node: &Node<F> = &self.nodes[node_idx];
        let left_s = node.left.map(|idx| &self.nodes[idx].stats);
        let right_s = node.right.map(|idx| &self.nodes[idx].stats);
        node.update_internal(left_s, right_s);
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one.
    ///
    /// Function in River/LightRiver: "learn_one()"
    pub fn partial_fit(&mut self, x: &Array1<F>, y: &String) {
        // TODO: remove prints, roll back to previous version
        self.root = match self.root {
            None => {
                let a = Some(self.create_leaf(x, y, None));
                println!("create_leaf() - post {self}");
                a
            }
            Some(root_idx) => {
                let a = Some(self.extend_mondrian_block(root_idx, x, y));
                println!("extend_modnrian_block() - post {self}");
                a
            }
        };
    }

    fn fit(&self) {
        unimplemented!("Make the program first work with 'partial_fit', then implement this")
    }

    /// Function in River/LightRiver: "score_one()"
    ///
    /// Recursive function to predict probabilities.
    fn predict(&self, x: &Array1<F>, node_idx: usize, p_not_separated_yet: F) -> Array1<F> {
        // println!("predict_proba() - MondrianTree: {}", self);
        let node = &self.nodes[node_idx];
        // Step 1: Calculate the time delta from the parent node.
        let d = node.tau - self.get_parent_tau(node_idx);

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
            let child_res =
                self.predict(x, child_idx.unwrap(), p_not_separated_yet * (F::one() - p));
            return res + child_res;
        }
    }

    fn get_params(&self) {
        unimplemented!()
    }

    pub fn get_parent_tau(&self, node_idx: usize) -> F {
        // If node is root, tau is 0
        match self.nodes[node_idx].parent {
            Some(parent_idx) => self.nodes[parent_idx].tau,
            None => F::from_f32(0.0).unwrap(),
        }
    }
}
