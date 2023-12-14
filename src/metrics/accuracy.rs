use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use crate::common::{ClassifierOutput, ClassifierTarget};
use crate::metrics::confusion::ConfusionMatrix;
use crate::metrics::traits::ClassificationMetric;
use num::{Float, FromPrimitive};

struct Accuracy<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    cm: ConfusionMatrix<F>,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> Accuracy<F> {
    pub fn new() -> Self {
        Self {
            cm: ConfusionMatrix::new(),
        }
    }
}

// implement for trait ClassificationMetric
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
    ClassificationMetric<F> for Accuracy<F>
{
    fn update(
        &mut self,
        y_true: &ClassifierTarget,
        y_pred: &ClassifierOutput<F>,
        sample_weight: Option<F>,
    ) {
        self.cm.update(y_true, y_pred, sample_weight);
    }
    fn revert(
        &mut self,
        y_true: &ClassifierTarget,
        y_pred: &ClassifierOutput<F>,
        sample_weight: Option<F>,
    ) {
        self.cm.revert(y_true, y_pred, sample_weight);
    }
    fn get(&self) -> F {
        self.cm
            .total_true_positives()
            .div(F::from(self.cm.total_weight).unwrap())
    }

    fn is_multiclass(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_accuracy_binary_classification() {
        let mut accuracy: Accuracy<f64> = Accuracy::new();
        let y_trues = vec![ClassifierTarget::Bool(true), ClassifierTarget::Bool(false)];
        let y_preds = vec![ClassifierTarget::Bool(true), ClassifierTarget::Bool(true)];
        let real_accuracy = 0.5;
        for (y_true, y_pred) in y_trues.iter().zip(y_preds.iter()) {
            accuracy.update(&ClassifierOutput::Prediction(y_true.clone()), y_pred, None);
        }
        assert_eq!(accuracy.get(), real_accuracy);
    }

    #[test]
    fn test_accuracy_multiclassifiaction() {}
}
