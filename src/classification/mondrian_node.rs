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
use std::fmt;
use std::iter::FlatMap;
use std::ops::{Add, Div, Mul, Sub};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use std::rc::Rc;
use std::rc::Weak;
use std::{clone, cmp, mem, usize};

/// Node struct
#[derive(Clone)]
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
    pub fn update_leaf(&mut self, x: &Array1<F>, label_idx: usize) {
        self.stats.add(x, label_idx);
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
#[derive(Clone)]
pub struct Stats<F> {
    pub sums: Array2<F>,
    pub sq_sums: Array2<F>,
    pub counts: Array1<usize>,
    num_labels: usize,
}
impl<F: FType + fmt::Display> fmt::Display for Stats<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "┌ Stats")?;
        // sums
        write!(f, "│ sums: [")?;
        for row in self.sums.outer_iter() {
            write!(f, "{:?}, ", row.to_vec())?;
        }
        writeln!(f, "]")?;
        // sq_sums
        write!(f, "│ sq_sums: [")?;
        for row in self.sq_sums.outer_iter() {
            write!(f, "{:?}, ", row.to_vec())?;
        }
        writeln!(f, "]")?;
        // count
        write!(f, "└ counts: {}", self.counts)?;
        Ok(())
    }
}
impl<F: FType> Stats<F> {
    pub fn new(num_labels: usize, feature_dim: usize) -> Self {
        Stats {
            sums: Array2::zeros((num_labels, feature_dim)),
            sq_sums: Array2::zeros((num_labels, feature_dim)),
            counts: Array1::zeros(num_labels),
            num_labels,
        }
    }
    pub fn create_result(&self, x: &Array1<F>, w: F) -> Array1<F> {
        let probs = self.predict_proba(x);
        probs * w
    }
    pub fn add(&mut self, x: &Array1<F>, label_idx: usize) {
        // Same as: self.sums[label] += x;
        self.sums
            .row_mut(label_idx)
            .zip_mut_with(&x, |a, &b| *a += b);

        // Same as: self.sq_sums[label_idx] += x*x;
        // e.g. x: [1.059 0.580] -> x*x: [1.122  0.337]
        self.sq_sums
            .row_mut(label_idx)
            .zip_mut_with(&x, |a, &b| *a += b * b);

        self.counts[label_idx] += 1;
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
    /// Return probabilities of sample 'x' belonging to each class.
    ///
    /// e.g. probs: [0.1, 0.2, 0.7]
    ///
    /// TODO: Remove the assert that check for exact values, I was testing if unit tests make sense, but as
    /// shown below this does not show the error. The function is just too complex.
    ///
    /// # Example
    /// ```
    /// use light_river::classification::alias::FType;
    /// use light_river::classification::mondrian_node::Stats;
    /// use ndarray::{Array1, Array2};
    ///
    /// let mut stats = Stats::new(3, 2); // 3 classes and 2 features
    /// stats.sums = Array2::from_shape_vec((3,2), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    ///     .expect("Failed to create Array2");
    /// stats.sq_sums = Array2::from_shape_vec((3,2), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    ///     .expect("Failed to create Array2");;
    /// stats.counts = Array1::from_vec(vec![4, 5]);
    /// stats.add(&Array1::from_vec(vec![1.0, 2.0]), 0);
    /// stats.add(&Array1::from_vec(vec![2.0, 3.0]), 1);
    /// stats.add(&Array1::from_vec(vec![2.0, 4.0]), 1);
    ///
    /// let x = Array1::from_vec(vec![1.5, 3.0]);
    /// let probs = stats.predict_proba(&x);
    /// let expected = vec![0.998075, 0.001924008, 0.0];
    /// assert!(
    ///     (probs.clone() - Array1::from_vec(expected)).mapv(|a: f32| a.abs()).iter().all(|&x| x < 1e-4),
    ///     "Probabilities do not match expected values"
    /// );
    /// // Check all values inside [0, 1] range
    /// assert!(probs.clone().iter().all(|&x| x >= 0.0 && x <= 1.0), "Probabilities should be in [0, 1] range");
    /// // Check sum is 1
    /// assert!((probs.clone().sum() - 1.0).abs() < 1e-4, "Sum of probabilities should be 1");
    /// ```
    pub fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        let mut probs = Array1::zeros(self.num_labels);
        let mut sum_prob = F::zero();
        println!("{self}");

        for (index, ((sum, sq_sum), &count)) in self
            .sums
            .outer_iter()
            .zip(self.sq_sums.outer_iter())
            .zip(self.counts.iter())
            .enumerate()
        {
            let count_f = F::from_usize(count).unwrap();
            let avg = &sum / count_f;
            let var = (&sq_sum / count_f) - (&avg * &avg) + F::epsilon();
            let sigma = (&var * count_f) / (count_f - F::one() + F::epsilon());
            let pi = F::from_f32(std::f32::consts::PI).unwrap() * F::from_f32(2.0).unwrap();
            let z = pi.powi(x.len() as i32) * sigma.mapv(|s| s * s).sum().sqrt();
            // Same as dot product
            let dot_delta = (&(x - &avg) * &(x - &avg)).sum();
            let dot_sigma = (&sigma * &sigma).sum();
            let exponent = F::from_f32(0.5).unwrap() * dot_delta / dot_sigma;
            let mut prob = exponent.exp() / z;
            if count >= 1 {
                assert!(!prob.is_nan(), "Probabaility should never be NaN.");
            } else {
                assert!(prob.is_nan(), "Probabaility should be NaN.");
                prob = F::zero();
            }
            sum_prob += prob;
            probs[index] = prob;
        }

        for prob in probs.iter_mut() {
            *prob /= sum_prob;
        }
        probs
    }
}
