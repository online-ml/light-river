use crate::datasets::utils;
use crate::stream::data_stream::Target;
use crate::stream::iter_csv::IterCsv;
use std::{fs::File, path::Path};

/// ChatGPT Generated synthetic dataset.
///
/// Add 'synthetic.csv' to project root directory.
pub struct Synthetic;
impl Synthetic {
    pub fn load_data() -> Result<IterCsv<f32, File>, Box<dyn std::error::Error>> {
        // let file_name = "syntetic_dataset_paper.csv";
        let file_name = "syntetic_dataset_int.csv";
        let file = File::open(file_name)?;
        let y_cols = Some(Target::Name("label".to_string()));
        match IterCsv::<f32, File>::new(file, y_cols) {
            Ok(x) => Ok(x),
            Err(e) => Err(Box::new(e)),
        }
    }
}
