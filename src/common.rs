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
type Observation<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> =
    HashMap<String, F>;

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClassifierTarget {
    Bool(bool),
    Int(i32),
    String(String),
}

/// Represents the probability distribution of classification targets.
///
/// ```
/// use std::collections::HashMap;
/// use light_river::common::{ClassifierTarget, ClassifierTargetProbabilities};
/// use num::Float;
///
/// type ClassifierTargetProbabilities<F: Float> = HashMap<ClassifierTarget, F>;
///
/// let mut probs: ClassifierTargetProbabilities<f32> = HashMap::new();
/// probs.insert(ClassifierTarget::Bool(true), 0.7);
/// probs.insert(ClassifierTarget::Bool(false), 0.3);
/// ```
type ClassifierTargetProbabilities<
    F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign,
> = HashMap<ClassifierTarget, F>;

/// Represents a regression target using a Float value.
///
/// ```
/// use num::Float;
///
/// type RegressionTarget<F: Float> = F;
///
/// let target: RegressionTarget<f32> = 42.0;
/// ```
type RegressionTarget<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> = F;

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
