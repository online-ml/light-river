use std::{
    collections::HashMap,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use ndarray::{Array, Array1};
use num::{Float, FromPrimitive};

/// Represents an observation, using a HashMap of String keys and Float values.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use num::Float;
///
/// type Observation<F: Float> = HashMap<String, F>;
///
/// let mut obs: Observation<f32> = HashMap::new();
/// obs.insert("feature1".to_string(), 1.0);
/// obs.insert("featur2".to_string(), 2.0);
/// ```
pub type Observation<F> = HashMap<String, F>;

/// Enum for classification targets, supporting boolean, integer, and string labels.
///
///
/// # Example
///
/// ```
/// use light_river::common::ClfTarget;
///
/// let target_bool = ClfTarget::Bool(true);
/// let target_int = ClfTarget::Int(1);
/// let target_string = ClfTarget::String("class".to_string());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClfTarget {
    // TODO: rename to 'ClfTarget' to follow River convention
    Bool(bool),
    Int(usize),
    String(String),
}
// impl fmt::Display for ClfTarget {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         match self {
//             ClfTarget::Bool(b) => write!(f, "{}", b),
//             ClfTarget::Int(i) => write!(f, "{}", i),
//             ClfTarget::String(s) => write!(f, "{}", s),
//         }
//     }
// }
impl ClfTarget {
    pub fn from<T: Into<ClfTarget>>(item: T) -> Self {
        item.into()
    }
    /// Converts an iterator of a specific type to an iterator of `ClfTarget`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use light_river::common::ClfTarget; // Replace `your_crate_name` with the actual crate name.
    /// let strings = vec!["hello".to_string(), "world".to_string()].into_iter();
    /// let targets: Vec<_> = ClfTarget::from_iter(strings).collect();
    /// assert_eq!(
    ///     targets,
    ///     vec![
    ///         ClfTarget::String("hello".to_string()),
    ///         ClfTarget::String("world".to_string())
    ///     ]
    /// );
    ///
    /// let bools = vec![true, false].into_iter();
    /// let targets: Vec<_> = ClfTarget::from_iter(bools).collect();
    /// assert_eq!(
    ///     targets,
    ///     vec![ClfTarget::Bool(true), ClfTarget::Bool(false)]
    /// );
    /// ```
    pub fn from_iter<I: IntoClfTargetIter>(iter: I) -> Box<dyn Iterator<Item = Self>> {
        iter.into_classifier_target_iter()
    }
}

impl From<String> for ClfTarget {
    /// Converts a String into a ClfTarget::String variant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use light_river::common::ClfTarget;
    /// let s = String::from("hello");
    /// let target = ClfTarget::from(s);
    /// assert_eq!(target, ClfTarget::String("hello".to_string()));
    /// ```
    fn from(s: String) -> Self {
        ClfTarget::String(s)
    }
}

impl From<&str> for ClfTarget {
    /// Converts a &str into a ClfTarget::String variant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use light_river::common::ClfTarget;
    /// let target = ClfTarget::from("hello");
    /// assert_eq!(target, ClfTarget::String("hello".to_string()));
    /// ```
    fn from(s: &str) -> Self {
        ClfTarget::String(s.to_string())
    }
}

impl From<usize> for ClfTarget {
    /// Converts an usize into a ClfTarget::Int variant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use light_river::common::ClfTarget;
    /// let target = ClfTarget::from(123);
    /// assert_eq!(target, ClfTarget::Int(123));
    /// ```
    fn from(i: usize) -> Self {
        ClfTarget::Int(i)
    }
}

impl From<bool> for ClfTarget {
    /// Converts a bool into a ClfTarget::Bool variant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use light_river::common::ClfTarget;
    /// let target = ClfTarget::from(true);
    /// assert_eq!(target, ClfTarget::Bool(true));
    /// ```
    fn from(b: bool) -> Self {
        ClfTarget::Bool(b)
    }
}
impl From<&bool> for ClfTarget {
    fn from(b: &bool) -> Self {
        ClfTarget::Bool(*b)
    }
}

impl From<&usize> for ClfTarget {
    fn from(i: &usize) -> Self {
        ClfTarget::Int(*i)
    }
}

impl From<&String> for ClfTarget {
    fn from(s: &String) -> Self {
        ClfTarget::String(s.clone())
    }
}

// TODO: remove this implementation.
impl From<ClfTarget> for usize {
    fn from(i: ClfTarget) -> Self {
        match i {
            ClfTarget::Int(i) => i,
            ClfTarget::Bool(b) => panic!("Cannot convert Bool to usize"),
            ClfTarget::String(_) => panic!("Cannot convert String to usize"),
        }
    }
}

pub trait IntoClfTargetIter {
    fn into_classifier_target_iter(self) -> Box<dyn Iterator<Item = ClfTarget>>;
}

impl<I> IntoClfTargetIter for I
where
    I: Iterator + 'static,
    I::Item: Into<ClfTarget>,
{
    fn into_classifier_target_iter(self) -> Box<dyn Iterator<Item = ClfTarget>> {
        Box::new(self.map(Into::into))
    }
}

/// Represents the probability distribution of classification targets.
///
/// ```
/// use std::collections::HashMap;
/// use light_river::common::{ClfTarget, ClfTargetProbabilities};
/// use num::Float;
///
/// let mut probs: ClfTargetProbabilities<f32> = HashMap::new();
/// probs.insert(ClfTarget::Bool(true), 0.7);
/// probs.insert(ClfTarget::Bool(false), 0.3);
/// ```
pub type ClfTargetProbabilities<F> = HashMap<ClfTarget, F>;

/// Represents the output of a classification model, which can be either a prediction or a probability distribution.
/// The probability distribution is represented by a HashMap of ClfTarget and Float values.
/// The prediction is represented by a ClfTarget.
/// # Example
/// ```
/// use light_river::common::{ClfTarget, ClassifierOutput};
/// use maplit::hashmap;
/// let probs: ClassifierOutput<f64> = ClassifierOutput::Probabilities( hashmap!{
///     ClfTarget::String("Cat".to_string()) => 0.7,
///     ClfTarget::String("Dog".to_string()) => 0.15,
///     ClfTarget::String("Cow".to_string()) => 0.15,
/// });
/// let pred = probs.get_predicition();
/// assert_eq!(pred, ClfTarget::String("Cat".to_string()));
/// ```
#[derive(Debug)]
pub enum ClassifierOutput<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    Probabilities(ClfTargetProbabilities<F>),
    Prediction(ClfTarget),
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> ClassifierOutput<F> {
    pub fn get_predicition(&self) -> ClfTarget {
        match self {
            ClassifierOutput::Prediction(y) => y.clone(),
            ClassifierOutput::Probabilities(y) => {
                // Find the key with the highest probabilities
                y.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0
                    .clone()
            }
        }
    }
    pub fn get_probabilities(&self) -> ClfTargetProbabilities<F> {
        // If we had only the prediction we set the probability to 1.0
        match self {
            ClassifierOutput::Prediction(y) => {
                let mut probs = ClfTargetProbabilities::new();
                probs.insert(y.clone(), F::from(1.0).unwrap());
                probs
            }
            ClassifierOutput::Probabilities(y) => y.clone(),
        }
    }
    pub fn merge(&self, other: ClassifierOutput<F>) -> Self {
        panic!("Not implementing this. Simply add the probabilities of the two vectors.")
    }
}

/// Represents the output of a regression model, is a prediction.
/// The prediction is represented by a RegTarget.
/// # Example
/// ```
/// use light_river::common::{RegressionOutput, RegTarget};
/// let target: RegTarget<f32> = 0.1;
/// let probs: RegressionOutput<f32> = RegressionOutput::Prediction(target);
/// let pred = probs.get_predicition();
/// assert_eq!(pred, target);
/// ```
#[derive(Debug)]
pub enum RegressionOutput<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    Prediction(RegTarget<F>),
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> RegressionOutput<F> {
    pub fn get_predicition(&self) -> RegTarget<F> {
        match self {
            RegressionOutput::Prediction(y) => y.clone(),
        }
    }
}

/// Represents a regression target using a Float value.
///
/// ```
/// use num::Float;
///
/// type RegTarget<F: Float> = F;
///
/// let target: RegTarget<f32> = 42.0;
/// ```
pub type RegTarget<F> = F;

/// Enum for all possible model targets (classification, regression, clustering, anomaly).
///
/// # Example
///
/// ```
/// use light_river::common::{ModelTarget, ClfTarget};
///
/// let target_classification = ModelTarget::Classification::<f32>(ClfTarget::Bool(true));
/// let target_regression = ModelTarget::Regression(42.0f32);
/// let target_clustering = ModelTarget::Clustering::<f32>(3);
/// let target_anomaly = ModelTarget::Anomaly(0.8f32);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelTarget<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    Classification(ClfTarget),
    Regression(RegTarget<F>),
    Clustering(usize),
    Anomaly(F),
}

/// Trait for implementing a classifier model.
///
/// Implement this trait for your classifier to use the `learn_one`, `predict_proba`, and
/// `predict_one` methods.
pub trait Classifier<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    // Temporary changed function parameters.
    // TODO: change back to original parameters.

    // fn learn_one(&mut self, x: &Observation<F>, y: ClfTarget);
    fn learn_one(&mut self, x: &Array1<F>, y: &ClfTarget);

    // fn predict_proba(&self, x: &Observation<F>) -> ClfTargetProbabilities<F>;
    fn predict_proba(&self, x: &Array1<F>) -> Array1<F>;

    // fn predict_one(&self, x: &Observation<F>) -> ClfTarget;
    fn predict_one(&mut self, x: &Array1<F>, y: &ClfTarget) -> F;
}

/// Trait for implementing a regression model.
///
/// Implement this trait for your regressor to use the `learn_one` and `predict_one` methods.
pub trait Regressor<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    // TODO: same as Classifier, roll back to previous function params.

    // fn learn_one(&mut self, x: &Observation<F>, y: RegTarget<F>);
    fn learn_one(&mut self, x: &Array1<F>, y: &RegTarget<F>);

    // fn predict_one(&self, x: &Observation<F>) -> RegTarget<F>;
    fn predict_one(&mut self, x: &Array1<F>, y: &RegTarget<F>) -> F;
}

/// Trait for implementing an anomaly detector model.
///
/// Implement this trait for your anomaly detector to use the `learn_one` and `score_one` methods.
pub trait AnomalyDetector<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    fn learn_one(&mut self, x: &Observation<F>);
    fn score_one(&self, x: &Observation<F>) -> F;
}

/// Trait for implementing a clustering model.
///
/// Implement this trait for your clustering model to use the `learn_one` and `predict_one` methods.
pub trait Clusterer<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    fn learn_one(&mut self, x: &Observation<F>);
    fn predict_one(&self, x: &Observation<F>) -> usize;
}

/// Represents a generic model which can be one of several types (classifier, regressor, anomaly detector, or clusterer).
pub enum ModelType<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    Classifier(Box<dyn Classifier<F>>),
    Regressor(Box<dyn Regressor<F>>),
    AnomalyDetector(Box<dyn AnomalyDetector<F>>),
    Clusterer(Box<dyn Clusterer<F>>),
}

// TODO: uncomment once traits Regressor and Classifier are fixed.

// impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> ModelType<F> {
//     /// Adapter method for learning a single observation.
//     pub fn learn_one(&mut self, x: &Observation<F>, y: ModelTarget<F>) {
//         match (self, y) {
//             (ModelType::Classifier(classifier), ModelTarget::Classification(target)) => {
//                 classifier.learn_one(x, target);
//             }
//             (ModelType::Regressor(regressor), ModelTarget::Regression(target)) => {
//                 regressor.learn_one(x, target);
//             }
//             (ModelType::AnomalyDetector(detector), ModelTarget::Anomaly(_)) => {
//                 detector.learn_one(x);
//             }
//             (ModelType::Clusterer(clusterer), ModelTarget::Clustering(_)) => {
//                 clusterer.learn_one(x);
//             }
//             _ => panic!("Mismatch between ModelType and ModelTarget"),
//         }
//     }
//     /// Adapter method for predicting of a single observation.
//     pub fn predict_one(&self, x: &Observation<F>) -> ModelTarget<F> {
//         match self {
//             ModelType::Classifier(classifier) => {
//                 ModelTarget::Classification(classifier.predict_one(x))
//             }
//             ModelType::Regressor(regressor) => ModelTarget::Regression(regressor.predict_one(x)),
//             ModelType::AnomalyDetector(detector) => ModelTarget::Anomaly(detector.score_one(x)),
//             ModelType::Clusterer(clusterer) => ModelTarget::Clustering(clusterer.predict_one(x)),
//         }
//     }
// }
