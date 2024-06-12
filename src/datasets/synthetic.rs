use crate::stream::data_stream::Target;
use crate::stream::iter_csv::IterCsv;
use std::fs::File;
use std::path::Path;

use super::utils;

/// ChatGPT Generated synthetic dataset.
///
/// Add 'synthetic.csv' to project root directory.
pub struct Synthetic;
impl Synthetic {
    pub fn load_data() -> IterCsv<f32, File> {
        let url = "https://marcodifrancesco.com/assets/img/LightRiver/syntetic_dataset.csv";
        let file_name = "syntetic_dataset_v2.csv";
        if !Path::new(file_name).exists() {
            utils::download_csv_file(url, file_name);
        }
        let file = File::open(file_name).unwrap();
        let y_cols = Some(Target::Name("label".to_string()));
        IterCsv::<f32, File>::new(file, y_cols).unwrap()
    }
}
