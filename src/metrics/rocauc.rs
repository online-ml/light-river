use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use crate::common::{ClassifierOutput, ClfTarget};
use crate::metrics::confusion::ConfusionMatrix;
use crate::metrics::traits::ClassificationMetric;
use num::{Float, FromPrimitive};

/// Receiver Operating Characteristic Area Under the Curve (ROC AUC).
///
/// This metric provides an approximation of the true ROC AUC. Computing the true ROC AUC would
/// require storing all the predictions and ground truths, which may not be efficient. The approximation
/// error is typically insignificant as long as the predicted probabilities are well calibrated. Regardless,
/// this metric can be used to reliably compare models with each other.
///
/// # Parameters
///
/// - `n_threshold`: The number of thresholds used for discretizing the ROC curve. A higher value will lead to
///   more accurate results, but will also require more computation time and memory.
/// - `pos_val`: Value to treat as "positive".
///
/// # Examples
///
/// ```rust
/// use light_river::metrics::rocauc::ROCAUC;
/// use light_river::common::{ClfTarget, ClassifierOutput};
/// use light_river::metrics::traits::ClassificationMetric;
/// use std::collections::HashMap;
///
/// let y_pred = vec![
///     ClassifierOutput::Probabilities(HashMap::from([
///         (ClfTarget::from(true), 0.1),
///         (ClfTarget::from(false), 0.9),
///     ])),
///     ClassifierOutput::Probabilities(HashMap::from([(ClfTarget::from(true), 0.4)])),
///     ClassifierOutput::Probabilities(HashMap::from([
///         (ClfTarget::from(true), 0.35),
///         (ClfTarget::from(false), 0.65),
///     ])),
///     ClassifierOutput::Probabilities(HashMap::from([
///         (ClfTarget::from(true), 0.8),
///         (ClfTarget::from(false), 0.2),
///     ])),
/// ];
/// let y_true: Vec<bool> = vec![false, false, true, true];
///
/// let mut metric = ROCAUC::new(Some(10), ClfTarget::from(true));
///
/// for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
///     metric.update(&ClfTarget::from(*yt), yp, Some(1.0));
/// }
///
/// println!("ROCAUC: {:.2}%", metric.get() * 100.0);
/// ```
///
/// # Notes
///
/// The true ROC AUC might differ from the approximation. The accuracy can be improved by increasing the number
/// of thresholds, but this comes at the cost of more computation time and memory usage.
///
pub struct ROCAUC<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    n_threshold: Option<usize>,
    pos_val: ClfTarget,
    thresholds: Vec<F>,
    cms: Vec<ConfusionMatrix<F>>,
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> ROCAUC<F> {
    pub fn new(n_threshold: Option<usize>, pos_val: ClfTarget) -> Self {
        let n_threshold = n_threshold.unwrap_or(10);

        let mut thresholds = Vec::with_capacity(n_threshold);

        for i in 0..n_threshold {
            thresholds.push(
                F::from(i).unwrap() / (F::from(n_threshold).unwrap() - F::from(1.0).unwrap()),
            );
        }
        thresholds[0] -= F::from(1e-7).unwrap();
        thresholds[n_threshold - 1] += F::from(1e-7).unwrap();

        let mut cms = Vec::with_capacity(n_threshold);
        for _ in 0..n_threshold {
            cms.push(ConfusionMatrix::new());
        }

        Self {
            n_threshold: Some(n_threshold),
            pos_val: pos_val,
            thresholds: thresholds,
            cms: cms,
        }
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
    ClassificationMetric<F> for ROCAUC<F>
{
    fn update(
        &mut self,
        y_true: &ClfTarget,
        y_pred: &ClassifierOutput<F>,
        sample_weight: Option<F>,
    ) {
        // Get the probability of the positive class
        let p_pred = y_pred.get_probabilities();
        let default_proba = F::zero();
        let p_pred_pos = p_pred.get(&self.pos_val).unwrap_or(&default_proba);

        // Convert the target to a binary target
        let y_true = ClfTarget::from(y_true.eq(&self.pos_val));

        for (threshold, cm) in self.thresholds.iter().zip(self.cms.iter_mut()) {
            let y_pred = ClassifierOutput::Prediction(ClfTarget::from(p_pred_pos.ge(threshold)));
            cm.update(&y_pred, &y_true, sample_weight);
        }
    }

    fn revert(
        &mut self,
        y_true: &ClfTarget,
        y_pred: &ClassifierOutput<F>,
        sample_weight: Option<F>,
    ) {
        let p_pred = y_pred.get_probabilities();

        let default_proba = F::zero();
        let p_pred_pos = p_pred.get(&self.pos_val).unwrap_or(&default_proba);
        let y_true = ClfTarget::from(y_true.eq(&self.pos_val));

        for (threshold, cm) in self.thresholds.iter().zip(self.cms.iter_mut()) {
            let y_pred = ClassifierOutput::Prediction(ClfTarget::from(p_pred_pos.ge(threshold)));
            cm.revert(&y_pred, &y_true, sample_weight);
        }
    }
    fn get(&self) -> F {
        let mut tprs: Vec<F> = (0..self.n_threshold.unwrap()).map(|_| F::zero()).collect();
        let mut fprs: Vec<F> = (0..self.n_threshold.unwrap()).map(|_| F::zero()).collect();

        for (i, cm) in self.cms.iter().enumerate() {
            let true_positives: F = cm.true_positives(&self.pos_val);
            let true_negatives: F = cm.true_negatives(&self.pos_val);
            let false_positives: F = cm.false_positives(&self.pos_val);
            let false_negatives: F = cm.false_negatives(&self.pos_val);
            // println!(
            //     "tp: {}, tn: {}, fp: {}, fn: {}",
            //     true_positives.to_f64().unwrap_or(0.0),
            //     true_negatives.to_f64().unwrap_or(0.0),
            //     false_positives.to_f64().unwrap_or(0.0),
            //     false_negatives.to_f64().unwrap_or(0.0)
            // );
            // Handle the ;case of zero division
            let mut tpr: Option<F> = None;
            if true_positives + false_negatives != F::zero() {
                tpr = Some(true_positives.div(true_positives + false_negatives));
            }

            tprs[i] = tpr.unwrap_or(F::zero());

            // Handle the case of zero division
            let mut fpr: Option<F> = None;
            if false_positives + true_negatives != F::zero() {
                fpr = Some(false_positives.div(false_positives + true_negatives));
            }

            fprs[i] = fpr.unwrap_or(F::zero());
        }
        // Trapezoidal integration
        let mut auc = F::zero();
        for i in 0..self.n_threshold.unwrap() - 1 {
            auc += (fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / F::from(2.0).unwrap();
        } // TODO: Turn it functional

        -auc
    }

    fn is_multiclass(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    #[test]
    fn test_rocauc() {
        println!("ROCAUC");
        // same example as in the doctest
        let y_pred = vec![
            ClassifierOutput::Prediction(ClfTarget::from("cat")),
            ClassifierOutput::Prediction(ClfTarget::from("dog")),
            ClassifierOutput::Prediction(ClfTarget::from("bird")),
            ClassifierOutput::Prediction(ClfTarget::from("cat")),
        ];
        let y_true: Vec<&str> = vec!["cat", "cat", "dog", "cat"];

        let mut metric = ROCAUC::new(Some(10), ClfTarget::from("cat"));

        for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
            metric.update(&ClfTarget::from(*yt), yp, Some(1.0));
        }
        println!("ROCAUC: {:.2}%", metric.get() * 100.0);
    }
    #[test]
    fn test_rocauc_proba() {
        let y_pred = vec![
            ClassifierOutput::Probabilities(HashMap::from([
                (ClfTarget::from(true), 0.1),
                (ClfTarget::from(false), 0.9),
            ])),
            ClassifierOutput::Probabilities(HashMap::from([(ClfTarget::from(true), 0.4)])),
            ClassifierOutput::Probabilities(HashMap::from([
                (ClfTarget::from(true), 0.35),
                (ClfTarget::from(false), 0.65),
            ])),
            ClassifierOutput::Probabilities(HashMap::from([
                (ClfTarget::from(true), 0.8),
                (ClfTarget::from(false), 0.2),
            ])),
        ];
        let y_true: Vec<bool> = vec![false, false, true, true];

        let mut metric = ROCAUC::new(Some(10), ClfTarget::from(true));

        for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
            metric.update(&ClfTarget::from(*yt), yp, Some(1.0));
        }

        assert!(metric.get() - 0.875 < 1e-7);
    }
}
