use crate::classification::alias::FType;
use crate::classification::mondrian_tree::MondrianTree;
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
use std::{cmp, mem, usize};

pub struct MondrianForest<F: FType> {
    trees: Vec<MondrianTree<F>>,
}
impl<F: FType> MondrianForest<F> {
    pub fn new(window_size: usize, n_trees: usize, features: &Vec<String>) -> Self {
        let tree_default = MondrianTree::new(window_size, features);
        let trees = vec![tree_default; n_trees];
        MondrianForest::<F> { trees }
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one.
    ///
    /// Function in River/LightRiver: "learn_one()"
    pub fn partial_fit(&mut self, x: &HashMap<String, f32>, y: &ClassifierTarget) {
        for tree in &mut self.trees {
            tree.partial_fit(x, y)
        }
    }

    pub fn fit(x: &HashMap<String, f32>, y: &ClassifierTarget) {
        unimplemented!()
    }

    pub fn predict_proba(&self, x: &Array1<F>, y: &ClassifierTarget) -> F {
        let mut scores = vec![];
        for tree in &self.trees {
            let score = tree.predict_proba(x, y);
            scores.push(score);
        }
        let mut sum = F::zero();
        // for score in &scores {
        //     sum += score;
        // }
        // sum /= scores.len() as f32;
        // ClassifierOutput::Prediction(sum)
        F::one()
    }
}
