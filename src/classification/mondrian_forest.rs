use crate::classification::alias::FType;
use crate::classification::mondrian_tree::MondrianTree;
use crate::common::{ClassifierOutput, ClassifierTarget, Observation};
use crate::stream::data_stream::Data;
use core::iter::zip;
use ndarray::array;
use ndarray::{arr1, Array1, Array2};
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
use std::{cmp, mem, usize};

pub struct MondrianForest<F: FType> {
    trees: Vec<MondrianTree<F>>,
    labels: Vec<String>,
}
impl<F: FType> MondrianForest<F> {
    pub fn new(
        window_size: usize,
        n_trees: usize,
        features: &Vec<String>,
        labels: &Vec<String>,
    ) -> Self {
        let tree_default = MondrianTree::new(window_size, features, labels);
        let trees = vec![tree_default; n_trees];
        let labels = labels.clone();
        MondrianForest::<F> { trees, labels }
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one.
    ///
    /// Function in River/LightRiver: "learn_one()"
    pub fn partial_fit(&mut self, x: &Array1<F>, y: &String) {
        println!("partial_fit() - x: {:?}, y: {y:?}", x.to_vec());
        for tree in &mut self.trees {
            tree.partial_fit(x, y);
        }
    }

    pub fn fit(x: &HashMap<String, f32>, y: &String) {
        unimplemented!()
    }

    pub fn predict_proba(&self, x: &Array1<F>) -> F {
        // scores shape in nel215: (n_trees, n_samples, n_labels)
        // scores shape here: (n_trees, n_labels)
        let mut scores = Array2::<F>::zeros((self.trees.len(), self.labels.len()));
        for tree_idx in 0..self.trees.len() {
            let out = self.trees[tree_idx].predict_proba(x);
            // Sort 'Probabilities' (the HashMap) output labels getting
            let probs = out.get_probabilities();
            unimplemented!("Must first fix 'predict_proba' to return probabilities (HashMap) with -> 'class1': 0.1, ...");
            let probs_sorted: Vec<F> = self
                .labels
                .iter()
                .map(|label| probs[&ClassifierTarget::from(label)])
                .collect();

            // scores[tree_idx] = probs_sorted;
            println!("probs {probs:?}");
        }
        let mut sum = F::zero();
        println!("scores: {:?}", scores);
        // for score in &scores {
        //     sum += score;
        // }
        // sum /= scores.len() as f32;
        // ClassifierOutput::Prediction(sum)
        F::one()
    }
}
