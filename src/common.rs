use std::{
    collections::HashMap,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

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
/// # Example
///
/// ```
/// use light_river::common::ClassifierTarget;
///
/// let target_bool = ClassifierTarget::Bool(true);
/// let target_int = ClassifierTarget::Int(1);
/// let target_string = ClassifierTarget::String("class".to_string());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClassifierTarget {
    Bool(bool),
    Int(i32),
    String(String),
}
// impl fmt::Display for ClassifierTarget {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         match self {
//             ClassifierTarget::Bool(b) => write!(f, "{}", b),
//             ClassifierTarget::Int(i) => write!(f, "{}", i),
//             ClassifierTarget::String(s) => write!(f, "{}", s),
//         }
//     }
// }
impl ClassifierTarget {
    pub fn from<T: Into<ClassifierTarget>>(item: T) -> Self {
        item.into()
    }
    /// Converts an iterator of a specific type to an iterator of `ClassifierTarget`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use light_river::common::ClassifierTarget; // Replace `your_crate_name` with the actual crate name.
    /// let strings = vec!["hello".to_string(), "world".to_string()].into_iter();
    /// let targets: Vec<_> = ClassifierTarget::from_iter(strings).collect();
    /// assert_eq!(
    ///     targets,
    ///     vec![
    ///         ClassifierTarget::String("hello".to_string()),
    ///         ClassifierTarget::String("world".to_string())
    ///     ]
    /// );
    ///
    /// let bools = vec![true, false].into_iter();
    /// let targets: Vec<_> = ClassifierTarget::from_iter(bools).collect();
    /// assert_eq!(
    ///     targets,
    ///     vec![ClassifierTarget::Bool(true), ClassifierTarget::Bool(false)]
    /// );
    /// ```
    pub fn from_iter<I: IntoClassifierTargetIter>(iter: I) -> Box<dyn Iterator<Item = Self>> {
        iter.into_classifier_target_iter()
    }
}

impl From<String> for ClassifierTarget {
    /// Converts a String into a ClassifierTarget::String variant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use light_river::common::ClassifierTarget;
    /// let s = String::from("hello");
    /// let target = ClassifierTarget::from(s);
    /// assert_eq!(target, ClassifierTarget::String("hello".to_string()));
    /// ```
    fn from(s: String) -> Self {
        ClassifierTarget::String(s)
    }
}

impl From<&str> for ClassifierTarget {
    /// Converts a &str into a ClassifierTarget::String variant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use light_river::common::ClassifierTarget;
    /// let target = ClassifierTarget::from("hello");
    /// assert_eq!(target, ClassifierTarget::String("hello".to_string()));
    /// ```
    fn from(s: &str) -> Self {
        ClassifierTarget::String(s.to_string())
    }
}

impl From<i32> for ClassifierTarget {
    /// Converts an i32 into a ClassifierTarget::Int variant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use light_river::common::ClassifierTarget;
    /// let target = ClassifierTarget::from(123);
    /// assert_eq!(target, ClassifierTarget::Int(123));
    /// ```
    fn from(i: i32) -> Self {
        ClassifierTarget::Int(i)
    }
}

impl From<bool> for ClassifierTarget {
    /// Converts a bool into a ClassifierTarget::Bool variant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use light_river::common::ClassifierTarget;
    /// let target = ClassifierTarget::from(true);
    /// assert_eq!(target, ClassifierTarget::Bool(true));
    /// ```
    fn from(b: bool) -> Self {
        ClassifierTarget::Bool(b)
    }
}
impl From<&bool> for ClassifierTarget {
    fn from(b: &bool) -> Self {
        ClassifierTarget::Bool(*b)
    }
}

impl From<&i32> for ClassifierTarget {
    fn from(i: &i32) -> Self {
        ClassifierTarget::Int(*i)
    }
}

impl From<&String> for ClassifierTarget {
    fn from(s: &String) -> Self {
        ClassifierTarget::String(s.clone())
    }
}

pub trait IntoClassifierTargetIter {
    fn into_classifier_target_iter(self) -> Box<dyn Iterator<Item = ClassifierTarget>>;
}

impl<I> IntoClassifierTargetIter for I
where
    I: Iterator + 'static,
    I::Item: Into<ClassifierTarget>,
{
    fn into_classifier_target_iter(self) -> Box<dyn Iterator<Item = ClassifierTarget>> {
        Box::new(self.map(Into::into))
    }
}

/// Represents the probability distribution of classification targets.
///
/// ```
/// use std::collections::HashMap;
/// use light_river::common::{ClassifierTarget, ClassifierTargetProbabilities};
/// use num::Float;
///
/// let mut probs: ClassifierTargetProbabilities<f32> = HashMap::new();
/// probs.insert(ClassifierTarget::Bool(true), 0.7);
/// probs.insert(ClassifierTarget::Bool(false), 0.3);
/// ```
pub type ClassifierTargetProbabilities<F> = HashMap<ClassifierTarget, F>;

/// Represents the output of a classification model, which can be either a prediction or a probability distribution.
/// The probability distribution is represented by a HashMap of ClassifierTarget and Float values.
/// The prediction is represented by a ClassifierTarget.
/// # Example
/// ```
/// use light_river::common::{ClassifierTarget, ClassifierOutput};
/// use maplit::hashmap;
/// let probs: ClassifierOutput<f64> = ClassifierOutput::Probabilities( hashmap!{
///     ClassifierTarget::String("Cat".to_string()) => 0.7,
///     ClassifierTarget::String("Dog".to_string()) => 0.15,
///     ClassifierTarget::String("Cow".to_string()) => 0.15,
/// });
/// let pred = probs.get_predicition();
/// assert_eq!(pred, ClassifierTarget::String("Cat".to_string()));
/// ```
#[derive(Debug)]
pub enum ClassifierOutput<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    Probabilities(ClassifierTargetProbabilities<F>),
    Prediction(ClassifierTarget),
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> ClassifierOutput<F> {
    pub fn get_predicition(&self) -> ClassifierTarget {
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
    pub fn get_probabilities(&self) -> ClassifierTargetProbabilities<F> {
        // If we had only the prediction we set the probability to 1.0
        match self {
            ClassifierOutput::Prediction(y) => {
                let mut probs = ClassifierTargetProbabilities::new();
                probs.insert(y.clone(), F::from(1.0).unwrap());
                probs
            }
            ClassifierOutput::Probabilities(y) => y.clone(),
        }
    }
    pub fn merge(&self, other: ClassifierOutput<F>) -> Self {
        unimplemented!()
        // match (self, other) {
        //     (ClassifierOutput::Probabilities(p1), ClassifierOutput::Probabilities(p2)) => {
        //         let mut combined = p1.clone();
        //         for (k, v) in p2 {
        //             *combined.entry(k).or_insert(F::zero()) += v;
        //         }
        //         ClassifierOutput::Probabilities(combined)
        //     },
        //     // Add more match arms as needed for other cases.
        //     _ => unimplemented!(),
        // }
    }
}

/// Represents the output of a regression model, is a prediction.
/// The prediction is represented by a RegressionTarget.
/// # Example
/// ```
/// use light_river::common::{RegressionOutput, RegressionTarget};
/// let target: RegressionTarget<f32> = 0.1;
/// let probs: RegressionOutput<f32> = RegressionOutput::Prediction(target);
/// let pred = probs.get_predicition();
/// assert_eq!(pred, target);
/// ```
#[derive(Debug)]
pub enum RegressionOutput<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    Prediction(RegressionTarget<F>),
}
impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> RegressionOutput<F> {
    pub fn get_predicition(&self) -> RegressionTarget<F> {
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
/// type RegressionTarget<F: Float> = F;
///
/// let target: RegressionTarget<f32> = 42.0;
/// ```
pub type RegressionTarget<F> = F;

/// Enum for all possible model targets (classification, regression, clustering, anomaly).
///
/// # Example
///
/// ```
/// use light_river::common::{ModelTarget, ClassifierTarget};
///
/// let target_classification = ModelTarget::Classification::<f32>(ClassifierTarget::Bool(true));
/// let target_regression = ModelTarget::Regression(42.0f32);
/// let target_clustering = ModelTarget::Clustering::<f32>(3);
/// let target_anomaly = ModelTarget::Anomaly(0.8f32);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelTarget<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    Classification(ClassifierTarget),
    Regression(RegressionTarget<F>),
    Clustering(i32),
    Anomaly(F),
}

/// Trait for implementing a classifier model.
///
/// Implement this trait for your classifier to use the `learn_one`, `predict_proba`, and
/// `predict_one` methods.
pub trait Classifier<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    fn learn_one(&mut self, x: &Observation<F>, y: ClassifierTarget);
    fn predict_proba(&self, x: &Observation<F>) -> ClassifierTargetProbabilities<F>;
    fn predict_one(&self, x: &Observation<F>) -> ClassifierTarget;
}

/// Trait for implementing a regression model.
///
/// Implement this trait for your regressor to use the `learn_one` and `predict_one` methods.
pub trait Regressor<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    fn learn_one(&mut self, x: &Observation<F>, y: RegressionTarget<F>);
    fn predict_one(&self, x: &Observation<F>) -> RegressionTarget<F>;
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
    fn predict_one(&self, x: &Observation<F>) -> i32;
}

/// Represents a generic model which can be one of several types (classifier, regressor, anomaly detector, or clusterer).
pub enum ModelType<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    Classifier(Box<dyn Classifier<F>>),
    Regressor(Box<dyn Regressor<F>>),
    AnomalyDetector(Box<dyn AnomalyDetector<F>>),
    Clusterer(Box<dyn Clusterer<F>>),
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> ModelType<F> {
    /// Adapter method for learning a single observation.
    pub fn learn_one(&mut self, x: &Observation<F>, y: ModelTarget<F>) {
        match (self, y) {
            (ModelType::Classifier(classifier), ModelTarget::Classification(target)) => {
                classifier.learn_one(x, target);
            }
            (ModelType::Regressor(regressor), ModelTarget::Regression(target)) => {
                regressor.learn_one(x, target);
            }
            (ModelType::AnomalyDetector(detector), ModelTarget::Anomaly(_)) => {
                detector.learn_one(x);
            }
            (ModelType::Clusterer(clusterer), ModelTarget::Clustering(_)) => {
                clusterer.learn_one(x);
            }
            _ => panic!("Mismatch between ModelType and ModelTarget"),
        }
    }
    /// Adapter method for predicting of a single observation.
    pub fn predict_one(&self, x: &Observation<F>) -> ModelTarget<F> {
        match self {
            ModelType::Classifier(classifier) => {
                ModelTarget::Classification(classifier.predict_one(x))
            }
            ModelType::Regressor(regressor) => ModelTarget::Regression(regressor.predict_one(x)),
            ModelType::AnomalyDetector(detector) => ModelTarget::Anomaly(detector.score_one(x)),
            ModelType::Clusterer(clusterer) => ModelTarget::Clustering(clusterer.predict_one(x)),
        }
    }
}
