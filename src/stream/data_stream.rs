use std::collections::{HashMap, HashSet};

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
    String(String),
}

/// "This enum defines whether your DataSteam only contains observations (X) or both observations and one or more targets (XY)
pub enum DataStream<F: Float + std::str::FromStr> {
    X(HashMap<String, Data<F>>),
    XY(HashMap<String, Data<F>>, HashMap<String, Data<F>>),
}

impl<F: Float + std::str::FromStr> DataStream<F> {
    pub fn get_x(&self) -> &HashMap<String, Data<F>> {
        match self {
            DataStream::X(x) => x,
            DataStream::XY(x, _) => x,
        }
    }

    pub fn get_y(&self) -> Result<&HashMap<String, Data<F>>, &str> {
        match self {
            DataStream::X(_) => Err("No y data"),
            DataStream::XY(_, y) => Ok(y),
        }
    }
}
