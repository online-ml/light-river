use num::{Float, FromPrimitive};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use crate::common::{ClassifierOutput, Observation};

pub mod half_space_tree;

trait AnomalyDetector<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    fn learn_one(&mut self, observation: &Observation<F>);
    fn score_one(&mut self, observation: &Observation<F>) -> Option<ClassifierOutput<F>>;
}
