use std::{
    collections::HashMap,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use super::common::{
    ClassifierOutput, ClassifierTarget, ClassifierTargetProbabilities, RegressionTarget,
};
use num::{Float, FromPrimitive};

pub trait BinaryMetric<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    fn update(
        &mut self,
        y_true: &ClassifierOutput<F>,
        y_pred: &ClassifierTarget,
        sample_weight: Option<F>,
    );
    fn revert(
        &mut self,
        y_true: &ClassifierOutput<F>,
        y_pred: &ClassifierTarget,
        sample_weight: Option<F>,
    );
    fn get(&self) -> F;
}

pub trait MultiClassMetric<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    fn update(
        &mut self,
        y_true: &ClassifierOutput<F>,
        y_pred: &ClassifierTarget,
        sample_weight: Option<F>,
    );
    fn revert(
        &mut self,
        y_true: &ClassifierOutput<F>,
        y_pred: &ClassifierTarget,
        sample_weight: Option<F>,
    );
    fn get(&self) -> F;
}

pub trait RegressionMetric<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    fn update(&mut self, y_true: RegressionTarget<F>, y_pred: RegressionTarget<F>);
    fn revert(&mut self, y_true: RegressionTarget<F>, y_pred: RegressionTarget<F>);
    fn get(&self) -> F;
}

pub trait ClustringMetric<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    fn update(&mut self, y_true: i32, y_pred: i32);
    fn revert(&mut self, y_true: i32, y_pred: i32);
    fn get(&self) -> F;
}
