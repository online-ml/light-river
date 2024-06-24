use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::common::{ClfTarget, Observation, RegTarget};
use num::Float;

/// This enum allows you to choose whether to define a single target (Name) or multiple targets (MultipleNames).
/// # Exemple
/// ```
/// use light_river::stream::data_stream::Target;
/// use light_river::stream::iter_csv::IterCsv;
/// // This import makes it easier to create HashSets using a macro,
/// // but you can still create a HashSet in the traditional way even if you choose not to use this import.Use to create an hasmap
/// use maplit::hashset;
///
/// let content = "Name,Height,Weight,Score\nAlice,1.6,60.0,90.0\nBob,1.8,80.0,85.0";
/// //Single target
/// IterCsv::<f32, &[u8]>::new(content.as_bytes(), Some(Target::Name("Score".to_string())));
/// /// multiple targets
/// IterCsv::<f32, &[u8]>::new(content.as_bytes(), Some(Target::MultipleNames( hashset! {"Height".to_string(), "Weight".to_string()})));
/// ```
pub enum Target {
    Name(String),
    MultipleNames(HashSet<String>),
}

impl Target {
    pub fn contains(&self, name: &str) -> bool {
        match self {
            Target::Name(n) => n == name,
            Target::MultipleNames(names) => names.contains(&name.to_string()),
        }
    }
}

/// This enum allows you to have two types of data for your observation and targets: either a scalar or a string.
/// # Exemple
/// ```
/// use light_river::stream::data_stream::Data;
///
/// let scalar = Data::<f32>::Scalar(1.6);
/// let string = Data::<f32>::String("age".to_string());
/// ```
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Data<F: Float + std::str::FromStr> {
    Scalar(F),
    Int(i32),
    Bool(bool),
    String(String),
}

impl<F: Float + fmt::Display + std::str::FromStr> fmt::Display for Data<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Data::Scalar(v) => write!(f, "{}", v),
            Data::Int(v) => write!(f, "{}", v),
            Data::Bool(v) => write!(f, "{}", v),
            Data::String(v) => write!(f, "{}", v),
        }
    }
}

impl<F: Float + std::fmt::Display + std::str::FromStr> Data<F> {
    pub fn to_float(&self) -> Result<F, &str> {
        match self {
            Data::Scalar(v) => Ok(*v),
            Data::Int(v) => Ok(F::from(*v).unwrap()),
            Data::Bool(v) => Ok(F::from(*v as i32).unwrap()),
            Data::String(_) => Err("Cannot convert string to float"),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Data::Scalar(v) => v.to_string(),
            Data::Int(v) => v.to_string(),
            Data::Bool(v) => v.to_string(),
            Data::String(v) => v.clone(),
        }
    }
}

#[derive(Debug)]
pub enum DataStream<F: Float + std::str::FromStr> {
    X(HashMap<String, Data<F>>),
    XY(HashMap<String, Data<F>>, HashMap<String, Data<F>>),
}

impl<F: Float + fmt::Display + std::str::FromStr> fmt::Display for DataStream<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_hashmap<F: Float + fmt::Display + std::str::FromStr>(
            f: &mut fmt::Formatter<'_>,
            hm: &HashMap<String, Data<F>>,
            hm_name: &str,
        ) -> fmt::Result {
            write!(f, "{hm_name}: [")?;
            for (key, value) in hm {
                write!(f, " {}: {},", key, value)?;
            }
            write!(f, " ] ")
        }

        match self {
            DataStream::X(x) => fmt_hashmap(f, x, "X"),
            DataStream::XY(x, y) => {
                fmt_hashmap(f, x, "X")?;
                fmt_hashmap(f, y, "Y")
            }
        }
    }
}

impl<F: Float + std::str::FromStr + std::fmt::Display> DataStream<F> {
    /// **get_x()**: e.g. {"H.t": Scalar(0.1069), "H.Return": Scalar(0.0742), ...}
    ///
    /// **get_observation()**: e.g. {"H.t": 0.1069, "H.Return": 0.0742, ...}
    pub fn get_x(&self) -> &HashMap<String, Data<F>> {
        match self {
            DataStream::X(x) => x,
            DataStream::XY(x, _) => x,
        }
    }

    /// **get_y()**: e.g. {"subject": String("s002")}
    ///
    /// **to_classifier_target()**: e.g. String("s002")
    pub fn to_classifier_target(&self, target_key: &str) -> Result<ClfTarget, &str> {
        match self {
            DataStream::X(_) => Err("No y data"),
            // Use data to float
            DataStream::XY(_, y) => {
                let y = y.get(target_key).unwrap();
                Ok(ClfTarget::from(y.to_string()))
            }
        }
    }

    // TODO: update values in docstring
    /// **get_y()**: e.g. {"subject": ????????}
    ///
    /// **to_regression_target()**: e.g. ??????
    pub fn to_regression_target(&self, target_key: &str) -> Result<RegTarget<F>, &str> {
        match self {
            DataStream::X(_) => Err("No y data"),
            // Use data to float
            DataStream::XY(_, y) => {
                let y = y.get(target_key).unwrap();
                Ok(RegTarget::<F>::from(y.to_float().unwrap()).unwrap())
            }
        }
    }

    /// **get_y()**: e.g. {"subject": String("s002")}
    ///
    /// **to_classifier_target()**: e.g. String("s002")
    pub fn get_y(&self) -> Result<&HashMap<String, Data<F>>, &str> {
        match self {
            DataStream::X(_) => Err("No y data"),
            DataStream::XY(_, y) => Ok(y),
        }
    }

    /// **get_x()**: e.g. {"H.t": Scalar(0.1069), "H.Return": Scalar(0.0742), ...}
    ///
    /// **get_observation()**: e.g. {"H.t": 0.1069, "H.Return": 0.0742, ...}
    pub fn get_observation(&self) -> Observation<F> {
        match self {
            DataStream::X(x) | DataStream::XY(x, _) => {
                x.iter()
                    .filter_map(|(k, v)| match v.to_float() {
                        Ok(f_value) => Some((k.clone(), f_value)),
                        Err(_) => None, // Ignore non-convertible data types
                    })
                    .collect()
            }
        }
    }
}
