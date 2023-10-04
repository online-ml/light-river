use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use crate::common::{ClassifierOutput, ClassifierTarget};
use crate::metrics::traits::ClassificationMetric;
use num::{Float, FromPrimitive};

struct Accuracy<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    n_samples: F,
    n_correct: F,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> Accuracy<F> {
    // pub fn new() -> Self {
    //     Self {
    //         n_samples: F::zero(),
    //         n_correct: F::zero(),
    //     }
    // }
}

// implement for trait ClassificationMetric
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
    ClassificationMetric<F> for Accuracy<F>
{
    fn update(
        &mut self,
        y_true: &ClassifierOutput<F>,
        y_pred: &ClassifierTarget,
        sample_weight: Option<F>,
    ) {
        let sample_weight = sample_weight.unwrap_or_else(|| F::one());
        let y_true = match y_true {
            ClassifierOutput::Prediction(y_true) => y_true,
            ClassifierOutput::Probabilities(y_true) => {
                // Find the key with the highest probabilities
                y_true
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0
            }
        };
        if y_true == y_pred {
            self.n_correct += sample_weight;
        }
        self.n_samples += sample_weight;
    }
    fn revert(
        &mut self,
        y_true: &ClassifierOutput<F>,
        y_pred: &ClassifierTarget,
        sample_weight: Option<F>,
    ) {
        let sample_weight = sample_weight.unwrap_or_else(|| F::one());
        let y_true = match y_true {
            ClassifierOutput::Prediction(y_true) => y_true,
            ClassifierOutput::Probabilities(y_true) => {
                // Find the key with the highest probabilities
                y_true
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0
            }
        };
        if y_true == y_pred {
            self.n_correct -= sample_weight;
        }
        self.n_samples -= sample_weight;

        if self.n_samples < F::zero() {
            self.n_samples = F::zero();
        }
    }
    fn get(&self) -> F {
        if self.n_samples == F::zero() {
            return F::zero();
        }
        self.n_correct / self.n_samples
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
