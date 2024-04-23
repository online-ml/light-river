use crate::classification::alias::FType;
use crate::common::{ClassifierOutput, ClassifierTarget, Observation};
use crate::stream::data_stream::Data;
use core::iter::zip;
use ndarray::{array, Array3};
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

/// Node struct
#[derive(Debug, Clone)]
pub struct Node<F> {
    // Change 'Rc' to 'Weak'
    pub parent: Option<usize>, // Option<Rc<RefCell<Node<F>>>>,
    pub tau: F,                // Time parameter: updated during 'node creation' or 'node update'
    pub is_leaf: bool,
    pub min_list: Array1<F>, // Lists representing the minimum and maximum values of the data points contained in the current node
    pub max_list: Array1<F>,
    pub delta: usize,         // Dimension in which a split occurs (?)
    pub xi: F,                // Split point along the dimension specified by delta
    pub left: Option<usize>,  // Option<Rc<RefCell<Node<F>>>>,
    pub right: Option<usize>, // Option<Rc<RefCell<Node<F>>>>,
    pub stats: Stats<F>,
}
impl<F: FType> Node<F> {
    pub fn update_leaf(&mut self, x: &Array1<F>, label: &ClassifierTarget) {
        self.stats.add(x, label);
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
}

/// Stats assocociated to one node
///
/// In nel215 code it is "Classifier"
#[derive(Debug, Clone)]
pub struct Stats<F> {
    sums: Array2<F>,
    sq_sums: Array1<F>,
    counts: Array1<usize>,
}
impl<F: FType> Stats<F> {
    pub fn new(num_labels: usize, feature_dim: usize) -> Self {
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
    fn add(&mut self, x: &Array1<F>, label: &ClassifierTarget) {
        println!("mondrian-node::add() - x: {:?} - {label:?}", x.to_vec());
        unimplemented!("Check nel215 add() implementation, check if these actions are supported by 'Array1' instead of this ugly code.");
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
        let mut probabilities = Array1::zeros(self.sums.len());
        let mut sum_prob = F::zero();

        // // for (index, ((sum, &sq_sum), &count)) in self.sums.outer_iter().zip(self.sq_sums.iter()).zip(self.counts.iter()).enumerate() {
        // for (index, ((sum, sq_sum), &count)) in self.sums.outer_iter().zip(self.sq_sums.iter()).zip(self.counts.iter()).enumerate() {
        //     if count > 1 {
        //         let count_f = F::from_usize(count).unwrap(); // Making sure the type is correct
        //         let mean = &sum / count_f;  // This division should work if sum is an Array1<F>
        //         let variance = (*sq_sum / count_f) - (&mean * &mean) + F::epsilon();
        //         let sigma = count_f * variance / (count_f - F::one() + F::epsilon());
        //         let norm_factor = ((2.0 * std::f64::consts::PI * sigma).sqrt());

        //         let exp_term = x.iter()
        //             .zip(mean.iter())
        //             .map(|(&xi, &mi)| {
        //                 let diff = xi - mi;
        //                 (diff * diff) / (2.0 * sigma)
        //             })
        //             .sum::<f64>();

        //         let prob = (-exp_term).exp() / norm_factor;
        //         probabilities[index] = prob;
        //         sum_prob += prob;
        //         }
        // }

        for prob in probabilities.iter_mut() {
            *prob /= sum_prob;
        }

        probabilities
    }
}
