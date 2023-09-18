use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use crate::common::{ClassifierOutput, ClassifierTarget, RegressionTarget};
use num::{Float, FromPrimitive};

pub trait ClassificationMetric<
    F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign,
>
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
    fn is_multiclass(&self) -> bool;
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

pub enum Metric<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    Classification(Box<dyn ClassificationMetric<F>>),
    Regression(Box<dyn RegressionMetric<F>>),
    Clustring(Box<dyn ClustringMetric<F>>),
}
