use crate::common::{ClfTarget, RegTarget};
use crate::mondrian_forest::alias::FType;

use ndarray::{Array, Array1, Array2};

use num::{Float, FromPrimitive, ToPrimitive};

use std::fmt;

use std::ops::Add;

use std::usize;

#[derive(Clone)]
pub struct NodeClassifier<F> {
    pub parent: Option<usize>,
    pub time: F, // How much I increased the area of the hyper-rectangle
    pub is_leaf: bool,
    pub range_min: Array1<F>,
    pub range_max: Array1<F>,
    pub feature: usize, // Feature number of the split
    pub threshold: F,   // Value of the split of the feature
    pub left: Option<usize>,
    pub right: Option<usize>,
    // TODO: rename 'sums' -> 'sum'
    pub sums: Array2<F>,
    // TODO: rename 'sq_sums' -> 'sq_sum'
    pub sq_sums: Array2<F>,
    // TODO: rename 'counts' -> 'count'
    pub counts: Array1<usize>,
    pub n_labels: usize,
    pub n_features: usize,
}
impl<F: FType + fmt::Display> fmt::Display for NodeClassifier<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node<time={:.3}, min={:?}, max={:?}, counts={:?}>",
            self.time,
            self.range_min.to_vec(),
            self.range_max.to_vec(),
            self.counts.to_vec(),
        )?;
        Ok(())
    }
}

impl<F: FType> NodeClassifier<F> {
    pub fn update_internal(&mut self, left: NodeClassifier<F>, right: NodeClassifier<F>) {
        self.sums = left.sums + right.sums;
        self.sq_sums = left.sq_sums + right.sq_sums;
        self.counts = left.counts + right.counts;
    }

    /// Check if all the labels are the same in the node.
    /// e.g. y=2, counts=[0, 1, 10] -> False
    /// e.g. y=2, counts=[0, 0, 10] -> True
    /// e.g. y=1, counts=[0, 0, 10] -> False
    pub fn is_dirac(&self, y: &ClfTarget) -> bool {
        let y: usize = (*y).clone().into();
        return self.counts.sum() == self.counts[y];
    }

    pub fn reset_stats(&mut self) {
        self.sums = Array2::zeros((self.n_labels, self.n_features));
        self.sq_sums = Array2::zeros((self.n_labels, self.n_features));
        self.counts = Array1::zeros(self.n_labels);
    }

    pub fn copy_stats_from(&mut self, node: &NodeClassifier<F>) {
        self.sums = node.sums.clone();
        self.sq_sums = node.sq_sums.clone();
        self.counts = node.counts.clone();
    }

    pub fn create_result(&self, x: &Array1<F>, w: F) -> Array1<F> {
        let probs = self.predict_proba(x);
        probs * w
    }

    pub fn add(&mut self, x: &Array1<F>, y: &ClfTarget) {
        // println!("add() - x={x}, y={y}, count={}, \nsums={}, \nsq_sums={}", self.counts, self.sums, self.sq_sums);

        let y: usize = (*y).clone().into();

        // Same as: self.sums[y] += x;
        self.sums.row_mut(y).zip_mut_with(&x, |a, &b| *a += b);
        // Same as: self.sq_sums[y] += x*x;
        // e.g. x: [1.059 0.580] -> x*x: [1.122  0.337]
        self.sq_sums
            .row_mut(y)
            .zip_mut_with(&x, |a, &b| *a += b * b);
        self.counts[y] += 1;

        // println!("      - y={y}, count={}, \nsums={}, \nsq_sums={}", self.counts, self.sums, self.sq_sums);
    }

    /// Return probabilities of sample 'x' belonging to each class.
    pub fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        let mut probs = Array1::zeros(self.n_labels);

        // println!("predict_proba() - start {}", self);

        // println!("var aware est   - counts: {}", self.counts);

        // Iterate over each label
        for (idx, ((sum, sq_sum), &count)) in self
            .sums
            .outer_iter()
            .zip(self.sq_sums.outer_iter())
            .zip(self.counts.iter())
            .enumerate()
        {
            // println!("                - idx: {idx}, count: {count}, sum: {sum}, sq_sum: {sq_sum}");

            let epsilon = F::epsilon();
            let count_f = F::from_usize(count).unwrap();
            let avg = &sum / count_f;
            let var = (&sq_sum / count_f) - (&avg * &avg) + epsilon;
            let sigma = (&var * count_f) / (count_f - F::one() + epsilon);
            let pi = F::from_f32(std::f32::consts::PI).unwrap() * F::from_f32(2.0).unwrap();
            let z = pi.powi(x.len() as i32) * sigma.mapv(|s| s * s).sum().sqrt();
            // Dot product
            let dot_feature = (&(x - &avg) * &(x - &avg)).sum();
            let dot_sigma = (&sigma * &sigma).sum();
            let exponent = -F::from_f32(0.5).unwrap() * dot_feature / dot_sigma;
            // epsilon added since exponent.exp() could be zero if exponent is very small
            let mut prob = (exponent.exp() + epsilon) / z;
            if count <= 0 {
                // prob is NaN
                prob = F::zero();
            }
            probs[idx] = prob;

            // DEBUG: stop using variance aware estimation
            probs[idx] = count_f;
        }

        if probs.iter().all(|&x| x == F::zero()) {
            // [0, 0, 0] -> [0.33, 0.33, 0.33]
            probs = probs
                .iter()
                .map(|_| F::one() / F::from_f32(probs.len().to_f32().unwrap()).unwrap())
                .collect();
        }
        let probs_sum = probs.sum();
        for prob in probs.iter_mut() {
            *prob /= probs_sum;
        }
        // println!("                - probs out: {}", probs);
        probs
    }
}

//#############################

#[derive(Clone)]
pub struct NodeRegressor<F> {
    pub parent: Option<usize>,
    pub time: F, // How much I increased the area of the hyper-rectangle
    pub is_leaf: bool,
    pub range_min: Array1<F>,
    pub range_max: Array1<F>,
    pub feature: usize, // Feature number of the split
    pub threshold: F,   // Value of the split of the feature
    pub left: Option<usize>,
    pub right: Option<usize>,
    // TODO: rename 'sums' -> 'sum'
    pub sums: F,
    // TODO: rename 'counts' -> 'count'
    pub counts: usize,
    pub n_features: usize,
}
impl<F: FType + fmt::Display> fmt::Display for NodeRegressor<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node<time={:.3}, min={:?}, max={:?}, counts={:?}>",
            self.time,
            self.range_min.to_vec(),
            self.range_max.to_vec(),
            self.counts,
        )?;
        Ok(())
    }
}

impl<F: FType> NodeRegressor<F> {
    pub fn update_internal(&mut self, left: NodeRegressor<F>, right: NodeRegressor<F>) {
        self.sums = left.sums + right.sums;
        self.counts = left.counts + right.counts;
    }

    /// Check if all the labels are the same in the node.
    /// e.g. y=2, counts=[0, 1, 10] -> False
    /// e.g. y=2, counts=[0, 0, 10] -> True
    /// e.g. y=1, counts=[0, 0, 10] -> False
    // TODO: remove this function, doesn't exist in Regression
    // pub fn is_dirac(&self, y: &RegTarget<F>) -> bool {
    //     let y: usize = (*y).clone().into();
    //     return self.counts.sum() == self.counts[y];
    // }

    pub fn reset_stats(&mut self) {
        self.sums = F::zero();
        self.counts = 0;
    }

    pub fn copy_stats_from(&mut self, node: &NodeRegressor<F>) {
        self.sums = node.sums.clone();
        self.counts = node.counts.clone();
    }

    pub fn create_result(&self, x: &Array1<F>, w: F) -> F {
        let probs = self.predict(x);
        probs * w
    }

    pub fn add(&mut self, x: &Array1<F>, y: &RegTarget<F>) {
        // println!("add() - x={x}, y={y}, count={}, \nsums={}, \nsq_sums={}", self.counts, self.sums, self.sq_sums);
        let y: F = (*y).into();
        self.sums += y;
        self.counts += 1;
        // println!("      - y={y}, count={}, \nsums={}, \nsq_sums={}", self.counts, self.sums, self.sq_sums);
    }

    pub fn predict(&self, x: &Array1<F>) -> F {
        if self.counts == 0 {
            println!("  predict() - is=zero");
            F::zero()
        } else {
            println!(
                "  predict() - sum={}, count={}, tot={}",
                self.sums,
                self.counts,
                self.sums / F::from_usize(self.counts).unwrap()
            );
            self.sums / F::from_usize(self.counts).unwrap()
        }
    }
}
