use std::{
    collections::HashMap,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num::{Float, FromPrimitive};

type Observation<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> =
    HashMap<String, F>;

pub enum ClassifierTarget {
    Bool(bool),
    Int(i32),
    String(String),
}

type ClassifierTargetProbabilities<
    F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign,
> = HashMap<ClassifierTarget, F>;

type RegressionTarget<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> = F;
pub enum ModelTarget<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    Classification(ClassifierTarget),
    Regression(RegressionTarget<F>),
    Clustering(i32),
    Anomaly(F),
}

pub trait Classifier<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    fn learn_one(&mut self, x: &Observation<F>, y: ClassifierTarget);
    fn predict_proba(&self, x: &Observation<F>) -> ClassifierTargetProbabilities<F>;
    fn predict_one(&self, x: &Observation<F>) -> ClassifierTarget;
}

pub trait Regressor<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    fn learn_one(&mut self, x: &Observation<F>, y: RegressionTarget<F>);
    fn predict_one(&self, x: &Observation<F>) -> RegressionTarget<F>;
}

pub trait AnomalyDetector<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign>
{
    fn learn_one(&mut self, x: &Observation<F>);
    fn score_one(&self, x: &Observation<F>) -> F;
}

pub trait Clusterer<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    fn learn_one(&mut self, x: &Observation<F>);
    fn predict_one(&self, x: &Observation<F>) -> i32;
}

pub enum ModelType<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> {
    Classifier(Box<dyn Classifier<F>>),
    Regressor(Box<dyn Regressor<F>>),
    AnomalyDetector(Box<dyn AnomalyDetector<F>>),
    Clusterer(Box<dyn Clusterer<F>>),
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign> ModelType<F> {
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
