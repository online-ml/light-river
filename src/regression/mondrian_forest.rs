use crate::classification::alias::FType;
use crate::classification::mondrian_tree::MondrianTreeClassifier;

use ndarray::Array1;

use num::{Float, FromPrimitive};
use rand::prelude::*;

use std::collections::HashMap;

use std::usize;

pub struct MondrianForestRegressor<F: FType> {
    trees: Vec<MondrianTreeClassifier<F>>,
}
impl<F: FType> MondrianForestRegressor<F> {
    pub fn new(n_trees: usize, n_features: usize) -> Self {
        let tree_default = MondrianTreeClassifier::new(n_features, 1);
        let trees = vec![tree_default; n_trees];
        MondrianForestRegressor::<F> { trees }
    }

    pub fn partial_fit(&mut self, x: &Array1<F>, y: F) {
        // for tree in &mut self.trees {
        //     tree.partial_fit(x, 123);
        // }
    }

    pub fn score(&mut self, x: &Array1<F>, y: F) -> F {
        F::one()
    }

    pub fn get_forest_size(&self) -> Vec<usize> {
        self.trees.iter().map(|t| t.get_tree_size()).collect()
    }
}
