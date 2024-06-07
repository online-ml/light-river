use crate::common::{
    Classifier, ClassifierTarget, ModelTarget, Observation, Regressor, RegressorTarget,
};
use crate::mondrian_forest::alias::FType;
use crate::mondrian_forest::mondrian_tree::MondrianTreeClassifier;
use ndarray::Array1; // Importing the required traits

use num::{Float, FromPrimitive};
use rand::prelude::*;

use std::collections::HashMap;

use std::usize;

use super::mondrian_tree::MondrianTreeRegressor;

pub struct MondrianForestClassifier<F: FType> {
    trees: Vec<MondrianTreeClassifier<F>>,
    n_labels: usize,
}

impl<F: FType> MondrianForestClassifier<F> {
    pub fn new(n_trees: usize, n_features: usize, n_labels: usize) -> Self {
        let tree_default = MondrianTreeClassifier::new(n_features, n_labels);
        let trees = vec![tree_default; n_trees];
        MondrianForestClassifier::<F> { trees, n_labels }
    }

    pub fn get_forest_size(&self) -> Vec<usize> {
        self.trees.iter().map(|t| t.get_tree_size()).collect()
    }
}

impl<F: FType> Classifier<F> for MondrianForestClassifier<F> {
    fn learn_one(&mut self, x: &Array1<F>, y: &ClassifierTarget) {
        for tree in &mut self.trees {
            tree.learn_one(x, y);
        }
    }

    // TODO: rename function to "predict_proba" both in trait and here
    fn predict_one(&mut self, x: &Array1<F>, y: &ClassifierTarget) -> F {
        let y = (*y).clone().into();
        let probs = self.predict_proba(x);
        let pred_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        // println!("probs: {}, pred_idx: {}, y (correct): {}, is_correct: {}", probs, pred_idx, y, pred_idx == y);
        if pred_idx == y {
            F::one()
        } else {
            F::zero()
        }
    }

    fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        let mut tot_probs = Array1::<F>::zeros(self.n_labels);
        for tree in &self.trees {
            let probs = tree.predict_proba(x);
            debug_assert!(
                !probs.iter().any(|&x| x.is_nan()),
                "Probability should not be NaN. Found: {:?}.",
                probs.to_vec()
            );
            tot_probs += &probs;
        }
        tot_probs /= F::from_usize(self.trees.len()).unwrap();
        tot_probs
    }
}

pub struct MondrianForestRegressor<F: FType> {
    trees: Vec<MondrianTreeRegressor<F>>,
}

impl<F: FType> MondrianForestRegressor<F> {
    pub fn new(n_trees: usize, n_features: usize) -> Self {
        let tree_default = MondrianTreeRegressor::new(n_features);
        let trees = vec![tree_default; n_trees];
        MondrianForestRegressor::<F> { trees }
    }

    pub fn get_forest_size(&self) -> Vec<usize> {
        self.trees.iter().map(|t| t.get_tree_size()).collect()
    }
}

impl<F: FType> Regressor<F> for MondrianForestRegressor<F> {
    fn learn_one(&mut self, x: &Array1<F>, y: &RegressorTarget<F>) {
        for tree in &mut self.trees {
            tree.learn_one(x, y);
        }
    }

    fn predict_one(&mut self, x: &Array1<F>, y: &RegressorTarget<F>) -> F {
        F::one()
    }
}
