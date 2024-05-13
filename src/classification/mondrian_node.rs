use crate::classification::alias::FType;

use ndarray::{Array1, Array2};

use num::{Float, FromPrimitive};

use std::fmt;

use std::ops::Add;

use std::usize;

/// Node struct
#[derive(Clone)]
pub struct Node<F> {
    pub parent: Option<usize>,
    pub time: F, // Time: how much I increased the size of the box
    pub is_leaf: bool,
    pub min_list: Array1<F>, // Lists representing the minimum and maximum values of the data points contained in the current node
    pub max_list: Array1<F>,
    pub feature: usize, // Feature in which a split occurs
    pub threshold: F,   // Threshold in which the split occures
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub stats: Stats<F>,
}
impl<F: FType + fmt::Display> fmt::Display for Node<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node<time={:.3}, min={:?}, max={:?}, counts={:?}>",
            self.time,
            self.min_list.to_vec(),
            self.max_list.to_vec(),
            self.stats.counts.to_vec(),
        )?;
        Ok(())
    }
}

impl<F: FType> Node<F> {
    pub fn update_leaf(&mut self, x: &Array1<F>, y: usize) {
        self.stats.add(x, y);
    }
    pub fn update_internal(&self, left_s: &Stats<F>, right_s: &Stats<F>) -> Stats<F> {
        left_s.merge(right_s)
    }
    /// Check if all the labels are the same in the node.
    /// e.g. y=2, stats.counts=[0, 1, 10] -> False
    /// e.g. y=2, stats.counts=[0, 0, 10] -> True
    /// e.g. y=1, stats.counts=[0, 0, 10] -> False
    pub fn is_dirac(&self, y: usize) -> bool {
        return self.stats.counts.sum() == self.stats.counts[y];
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
    n_labels: usize,
}
impl<F: FType + fmt::Display> fmt::Display for Stats<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n┌ Stats")?;
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
    pub fn new(n_labels: usize, n_features: usize) -> Self {
        Stats {
            sums: Array2::zeros((n_labels, n_features)),
            sq_sums: Array2::zeros((n_labels, n_features)),
            counts: Array1::zeros(n_labels),
            n_labels,
        }
    }
    pub fn create_result(&self, x: &Array1<F>, w: F) -> Array1<F> {
        let probs = self.predict_proba(x);
        probs * w
    }
    pub fn add(&mut self, x: &Array1<F>, y: usize) {
        // Same as: self.sums[y] += x;
        self.sums.row_mut(y).zip_mut_with(&x, |a, &b| *a += b);

        // Same as: self.sq_sums[y] += x*x;
        // e.g. x: [1.059 0.580] -> x*x: [1.122  0.337]
        self.sq_sums
            .row_mut(y)
            .zip_mut_with(&x, |a, &b| *a += b * b);

        self.counts[y] += 1;
    }
    fn merge(&self, s: &Stats<F>) -> Stats<F> {
        // NOTE: nel215 returns a new Stats object, we are only changing the node values here
        Stats {
            sums: self.sums.clone() + &s.sums,
            sq_sums: self.sq_sums.clone() + &s.sq_sums,
            counts: self.counts.clone() + &s.counts,
            n_labels: self.n_labels,
        }
    }
    /// Return probabilities of sample 'x' belonging to each class.
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
    /// // Check all values inside [0, 1] range
    /// assert!(probs.clone().iter().all(|&x| x >= 0.0 && x <= 1.0), "Probabilities should be in [0, 1] range");
    /// // Check sum is 1
    /// assert!((probs.clone().sum() - 1.0f32).abs() < 1e-4, "Sum of probabilities should be 1");
    /// ```
    pub fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        let mut probs = Array1::zeros(self.n_labels);
        let mut sum_prob = F::zero();

        // println!("predict_proba() - start {}", self);

        for (index, ((sum, sq_sum), &count)) in self
            .sums
            .outer_iter()
            .zip(self.sq_sums.outer_iter())
            .zip(self.counts.iter())
            .enumerate()
        {
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
                assert!(prob.is_nan(), "Probabaility should be NaN. Found: {prob}.");
                prob = F::zero();
            }
            sum_prob += prob;
            probs[index] = prob;
        }

        // Check at least one probability is non-zero. Otherwise we have division by zero.
        assert!(
            !probs.iter().all(|&x| x == F::zero()),
            "At least one probability should not be zero. Found: {:?}.",
            probs.to_vec()
        );

        for prob in probs.iter_mut() {
            *prob /= sum_prob;
        }
        // println!("predict_proba() post - probs: {:?}", probs.to_vec());
        probs
    }
}
