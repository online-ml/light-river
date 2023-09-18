use std::fmt;
use std::{
    collections::HashMap,
    collections::HashSet,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use crate::common::{ClassifierOutput, ClassifierTarget};
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
/// use light_river::common::ClassifierTarget;
///
/// let y_true = vec!["cat", "ant", "cat", "cat", "ant", "bird"];
/// let y_pred = vec!["ant", "ant", "cat", "cat", "ant", "cat"];
///
/// let mut cm: ConfusionMatrix<f64> = ConfusionMatrix::new();
///
/// for (yt, yp) in y_true.iter().zip(y_pred.iter()) {
///     cm.update(yt, yp); // Assuming an update method
/// }
///
/// // Representation of the matrix. This will depend on your actual implementation
/// // Here's just a placeholder. Make sure to adjust based on your actual method and output.
/// assert_eq!(cm.to_string(), "
///     ant  bird   cat
/// ant     2     0     0
/// bird     0     0     1
/// cat     1     0     2
/// ");
///
/// assert_eq!(cm.get(ClassifierTarget::String("bird"),ClassifierTarget::String("cat")), 1.0); // Assuming a get method
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
    data: HashMap<ClassifierTarget, HashMap<ClassifierTarget, F>>,
    sum_row: HashMap<ClassifierTarget, F>,
    sum_col: HashMap<ClassifierTarget, F>,
    total_weight: F,
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
    pub fn get_classes(&self) -> HashSet<ClassifierTarget> {
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
    fn _update(
        &mut self,
        y_pred: &ClassifierOutput<F>,
        y_true: &ClassifierTarget,
        sample_weight: F,
    ) {
        let label = y_pred.get_predicition();
        let y = y_true.clone();
        let y_col = y.clone();
        let label_row = label.clone();

        self.data
            .entry(label)
            .or_insert_with(HashMap::new)
            .entry(y)
            .and_modify(|x| *x += sample_weight)
            .or_insert(sample_weight);

        self.total_weight += sample_weight;
        self.sum_row
            .entry(y_col)
            .and_modify(|x| *x += sample_weight)
            .or_insert(sample_weight);
        self.sum_col
            .entry(label_row)
            .and_modify(|x| *x += sample_weight)
            .or_insert(sample_weight);
    }
    pub fn update(
        &mut self,
        y_pred: &ClassifierOutput<F>,
        y_true: &ClassifierTarget,
        sample_weight: Option<F>,
    ) {
        self.n_samples += sample_weight.unwrap_or(F::one());
        self._update(y_pred, y_true, sample_weight.unwrap_or(F::one()));
    }
    pub fn revert(
        &mut self,
        y_pred: &ClassifierOutput<F>,
        y_true: &ClassifierTarget,
        sample_weight: F,
    ) {
        self.n_samples -= sample_weight;
        self._update(y_pred, y_true, -sample_weight);
    }
    pub fn get(&self, label: &ClassifierTarget) -> HashMap<ClassifierTarget, F> {
        self.data.get(label).unwrap_or(&HashMap::new()).clone()
    }
    pub fn support(&self, label: &ClassifierTarget) -> F {
        self.sum_col.get(label).unwrap_or(&F::zero()).clone()
    }
    // For the next session you will check if the implementation of the following methods is correct
    pub fn true_positives(&self, label: &ClassifierTarget) -> F {
        self.data
            .get(label)
            .unwrap_or(&HashMap::new())
            .get(label)
            .unwrap_or(&F::zero())
            .clone()
    }
    pub fn true_negatives(&self, label: &ClassifierTarget) -> F {
        self.total_true_positives() - self.true_positives(label)
    }

    pub fn total_true_positives(&self) -> F {
        self.data
            .keys()
            .fold(F::zero(), |sum, label| sum + self.true_positives(label))
    }
    pub fn false_positives(&self, label: &ClassifierTarget) -> F {
        *self.sum_col.get(label).unwrap_or(&F::zero()) - self.true_positives(label)
    }

    pub fn total_true_negatives(&self) -> F {
        self.data
            .keys()
            .fold(F::zero(), |sum, label| sum + self.true_negatives(label))
    }

    pub fn total_false_positives(&self) -> F {
        self.data
            .keys()
            .fold(F::zero(), |sum, label| sum + self.false_positives(label))
    }
    pub fn false_negatives(&self, label: &ClassifierTarget) -> F {
        *self.sum_row.get(label).unwrap_or(&F::zero()) - self.true_positives(label)
    }
    pub fn total_false_negatives(&self) -> F {
        self.data
            .keys()
            .fold(F::zero(), |sum, label| sum + self.false_negatives(label))
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
    use super::*;
    #[test]
    fn test_confusion_matrix() {
        let y_true = vec![
            ClassifierOutput::Prediction(ClassifierTarget::from("cat")),
            ClassifierOutput::Prediction(ClassifierTarget::from("ant")),
            ClassifierOutput::Prediction(ClassifierTarget::from("cat")),
            ClassifierOutput::Prediction(ClassifierTarget::from("cat")),
            ClassifierOutput::Prediction(ClassifierTarget::from("ant")),
            ClassifierOutput::Prediction(ClassifierTarget::from("bird")),
        ];
        let y_pred = vec![
            "ant".to_string(),
            "ant".to_string(),
            "cat".to_string(),
            "cat".to_string(),
            "ant".to_string(),
            "cat".to_string(),
        ];
        let y_pred = ClassifierTarget::from_iter(y_pred.into_iter());

        let mut cm: ConfusionMatrix<f64> = ConfusionMatrix::new();

        for (yt, yp) in y_true.iter().zip(y_pred) {
            cm.update(yt, &yp, Some(1.0)); // Assuming an update method
        }

        //         assert_eq!(
        //             cm.to_string(),
        //             "
        //  ant  bird   cat
        //  ant     2     0     0
        //  bird     0     0     1
        //  cat     1     0     2
        //  "
        //         );
        assert_eq!(
            *cm.get(&ClassifierTarget::String("bird".to_string()))
                .get(&ClassifierTarget::String("cat".to_string()))
                .unwrap_or(&0.0),
            1.0
        ); // Assuming a get method
    }
}
