use std::fmt;
use std::{
    collections::HashMap,
    collections::HashSet,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use crate::common::{ClassifierOutput, ClfTarget};

use num::{Float, FromPrimitive};

/// Confusion Matrix for binary and multi-class classification.
///
/// # Parameters
///
/// - `classes`: The initial set of classes. This is optional and serves only for displaying purposes.
///
/// # Examples
///
/// ```
/// use light_river::metrics::confusion::ConfusionMatrix;
/// use light_river::common::{ClfTarget, ClassifierOutput};
///
/// let y_pred = vec![
///            ClassifierOutput::Prediction(ClfTarget::from("ant")),
///            ClassifierOutput::Prediction(ClfTarget::from("ant")),
///            ClassifierOutput::Prediction(ClfTarget::from("cat")),
///            ClassifierOutput::Prediction(ClfTarget::from("cat")),
///            ClassifierOutput::Prediction(ClfTarget::from("ant")),
///            ClassifierOutput::Prediction(ClfTarget::from("cat")),
/// ];
/// let y_pred_stream = y_pred.iter();
/// let y_true: Vec<String> = vec!["cat".to_string(), "ant".to_string(), "cat".to_string(), "cat".to_string(), "ant".to_string(), "bird".to_string()];
/// let y_true_stream = ClfTarget::from_iter(y_true.into_iter());
///
/// let mut cm: ConfusionMatrix<f64> = ConfusionMatrix::new();
///
/// for (yt, yp) in y_true_stream.zip(y_pred_stream) {
///     cm.update( &yp, &yt, Some(1.0)); // Assuming an update method
/// }
///
///
/// assert_eq!(*cm.get(&ClfTarget::from("bird")).get(&ClfTarget::from("cat")).unwrap_or(&0.0), 1.0);
/// ```
///
/// # Notes
///
/// This confusion matrix is a 2D matrix of shape `(n_classes, n_classes)`, corresponding
/// to a single-target (binary and multi-class) classification task.
///
/// Each row represents `true` (actual) class-labels, while each column corresponds
/// to the `predicted` class-labels. For example, an entry in position `[1, 2]` means
/// that the true class-label is 1, and the predicted class-label is 2 (incorrect prediction).
///
/// This structure is used to keep updated statistics about a single-output classifier's
/// performance and to compute multiple evaluation metrics.
///

#[derive(Clone)]
pub struct ConfusionMatrix<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    n_samples: F,
    data: HashMap<ClfTarget, HashMap<ClfTarget, F>>,
    sum_row: HashMap<ClfTarget, F>,
    sum_col: HashMap<ClfTarget, F>,
    pub total_weight: F,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> ConfusionMatrix<F> {
    pub fn new() -> Self {
        Self {
            n_samples: F::zero(),
            data: HashMap::new(),
            sum_row: HashMap::new(),
            sum_col: HashMap::new(),
            total_weight: F::zero(),
        }
    }
    pub fn get_classes(&self) -> HashSet<ClfTarget> {
        // Extracting classes from sum_row and sum_col
        let sum_row_keys = self
            .sum_row
            .keys()
            .filter(|&k| self.sum_row[k] != F::zero())
            .cloned();
        let sum_col_keys = self
            .sum_col
            .keys()
            .filter(|&k| self.sum_col[k] != F::zero())
            .cloned();

        // Combining the classes from sum_row and sum_col
        sum_row_keys.chain(sum_col_keys).collect()
    }
    fn _update(&mut self, y_pred: &ClassifierOutput<F>, y_true: &ClfTarget, sample_weight: F) {
        let label_pred = y_pred.get_predicition();
        let y = y_true.clone();
        let y_row = y.clone();
        let label_col = label_pred.clone();

        self.data
            .entry(y)
            .or_insert_with(HashMap::new)
            .entry(label_pred)
            .and_modify(|x| *x += sample_weight)
            .or_insert(sample_weight);

        self.total_weight += sample_weight;
        self.sum_row
            .entry(y_row)
            .and_modify(|x| *x += sample_weight)
            .or_insert(sample_weight);
        self.sum_col
            .entry(label_col)
            .and_modify(|x| *x += sample_weight)
            .or_insert(sample_weight);
    }
    pub fn update(
        &mut self,
        y_pred: &ClassifierOutput<F>,
        y_true: &ClfTarget,
        sample_weight: Option<F>,
    ) {
        self.n_samples += sample_weight.unwrap_or(F::one());
        self._update(y_pred, y_true, sample_weight.unwrap_or(F::one()));
    }
    pub fn revert(
        &mut self,
        y_pred: &ClassifierOutput<F>,
        y_true: &ClfTarget,
        sample_weight: Option<F>,
    ) {
        self.n_samples -= sample_weight.unwrap_or(F::one());
        self._update(y_pred, y_true, -sample_weight.unwrap_or(F::one()));
    }
    pub fn get(&self, label: &ClfTarget) -> HashMap<ClfTarget, F> {
        // return rows of the label in the confusion matrix
        self.data.get(label).unwrap_or(&HashMap::new()).clone()
    }
    pub fn support(&self, label: &ClfTarget) -> F {
        self.sum_col.get(label).unwrap_or(&F::zero()).clone()
    }
    // For the next session you will check if the implementation of the following methods is correct
    pub fn true_positives(&self, label: &ClfTarget) -> F {
        self.data
            .get(label)
            .unwrap_or(&HashMap::new())
            .get(label)
            .unwrap_or(&F::zero())
            .clone()
    }
    // pub fn true_negatives(&self, label: &ClfTarget) -> F {
    //     self.total_true_positives() - self.true_positives(label)
    // }

    pub fn total_true_positives(&self) -> F {
        self.data
            .keys()
            .fold(F::zero(), |sum, label| sum + self.true_positives(label))
    }
    pub fn false_positives(&self, label: &ClfTarget) -> F {
        *self.sum_col.get(label).unwrap_or(&F::zero()) - self.true_positives(label)
    }

    // pub fn total_true_negatives(&self) -> F {
    //     self.data
    //         .keys()
    //         .fold(F::zero(), |sum, label| sum + self.true_negatives(label))
    // }

    pub fn total_false_positives(&self) -> F {
        self.data
            .keys()
            .fold(F::zero(), |sum, label| sum + self.false_positives(label))
    }
    pub fn false_negatives(&self, label: &ClfTarget) -> F {
        *self.sum_row.get(label).unwrap_or(&F::zero()) - self.true_positives(label)
    }
    pub fn total_false_negatives(&self) -> F {
        self.data
            .keys()
            .fold(F::zero(), |sum, label| sum + self.false_negatives(label))
    }
    pub fn true_negatives(&self, label: &ClfTarget) -> F {
        self.total_weight
            - *self.sum_row.get(label).unwrap_or(&F::zero())
            - *self.sum_col.get(label).unwrap_or(&F::zero())
            + self.true_positives(label)
    }

    pub fn total_true_negatives(&self) -> F {
        self.data
            .keys()
            .fold(F::zero(), |sum, label| sum + self.true_negatives(label))
    }
}

impl<
        F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign + std::fmt::Display,
    > fmt::Debug for ConfusionMatrix<F>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Get sorted classes
        let mut classes: Vec<_> = self.get_classes().into_iter().collect();
        classes.sort();

        // Write headers
        write!(f, "{:<10}", "")?;
        for class in &classes {
            write!(f, "{:<10?}", class)?; // Use debug formatting
        }
        writeln!(f)?;
        let default_value = F::zero();
        // Write rows
        for row_class in &classes {
            write!(f, "{:<10?}", row_class)?; // Use debug formatting
            for col_class in &classes {
                let value = self
                    .data
                    .get(row_class)
                    .and_then(|inner| inner.get(col_class))
                    .unwrap_or(&default_value);
                write!(f, "{:<10.1}", *value)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> Default
    for ConfusionMatrix<F>
{
    fn default() -> Self {
        Self {
            n_samples: F::zero(),
            data: HashMap::new(),
            sum_row: HashMap::new(),
            sum_col: HashMap::new(),
            total_weight: F::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use num::Unsigned;

    use super::*;
    #[test]
    fn test_confusion_matrix() {
        let y_pred = vec![
            ClassifierOutput::Prediction(ClfTarget::from("ant")),
            ClassifierOutput::Prediction(ClfTarget::from("ant")),
            ClassifierOutput::Prediction(ClfTarget::from("cat")),
            ClassifierOutput::Prediction(ClfTarget::from("cat")),
            ClassifierOutput::Prediction(ClfTarget::from("ant")),
            ClassifierOutput::Prediction(ClfTarget::from("cat")),
        ];
        let y_pred_stream = y_pred.iter();
        let y_true: Vec<String> = vec![
            "cat".to_string(),
            "ant".to_string(),
            "cat".to_string(),
            "cat".to_string(),
            "ant".to_string(),
            "bird".to_string(),
        ];
        let y_true_stream = ClfTarget::from_iter(y_true.into_iter());

        let mut cm: ConfusionMatrix<f64> = ConfusionMatrix::new();

        for (yt, yp) in y_true_stream.zip(y_pred_stream) {
            cm.update(&yp, &yt, Some(1.0)); // Assuming an update method
        }
        println!("{:?}", cm);
        assert_eq!(
            *cm.get(&ClfTarget::from("bird"))
                .get(&ClfTarget::from("cat"))
                .unwrap_or(&0.0),
            1.0
        );
    }
    #[test]
    fn test_confusion_matrix_issue() {
        let y_pred = vec![
            ClassifierOutput::Prediction(ClfTarget::from("cat")),
            ClassifierOutput::Prediction(ClfTarget::from("dog")),
            ClassifierOutput::Prediction(ClfTarget::from("bird")),
            ClassifierOutput::Prediction(ClfTarget::from("cat")),
        ];
        let y_true: Vec<&str> = vec!["cat", "cat", "dog", "cat"];
        let y_true_stream = ClfTarget::from_iter(y_true.into_iter());
        let mut cm: ConfusionMatrix<f64> = ConfusionMatrix::new();
        let mut iteration: usize = 0;
        for (yt, yp) in y_true_stream.zip(y_pred.iter()) {
            cm.update(&yp, &yt, Some(1.0));

            let tp = cm.true_positives(&ClfTarget::from("cat"));
            let tn = cm.true_negatives(&ClfTarget::from("cat"));
            let fp = cm.false_positives(&ClfTarget::from("cat"));
            let fn_ = cm.false_negatives(&ClfTarget::from("cat"));
            println!("iteration: {}", iteration);
            println!("tp: {}, tn: {}, fp: {}, fn: {}", tp, tn, fp, fn_);
            println!("data :{:?}", cm.data);
            println!("sum col {:?}", cm.sum_col);
            println!("sum row {:?}", cm.sum_row);
            println!("total true positives: {}", cm.total_true_positives());
            println!("total true negatives: {}", cm.total_true_negatives());
            println!("total false positives: {}", cm.total_false_positives());
            println!("total false negatives: {}", cm.total_false_negatives());
            println!("-----------------");

            iteration += 1;
        }
        let tp = cm.true_positives(&ClfTarget::from("cat"));
        let tn = cm.true_negatives(&ClfTarget::from("cat"));
        let fp = cm.false_positives(&ClfTarget::from("cat"));
        let fn_ = cm.false_negatives(&ClfTarget::from("cat"));
        assert!(tp == 2.0);
        assert!(tn == 1.0);
        assert!(fp == 0.0);
        assert!(fn_ == 1.0);
    }
}
