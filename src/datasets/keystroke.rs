use crate::datasets::utils;
use crate::stream::data_stream::Target;
use crate::stream::iter_csv::IterCsv;
use std::{fs::File, path::Path};

/// CMU keystroke dataset.
///
/// Users are tasked to type in a password. The task is to determine which user is typing in the
/// password.
///
/// The only difference with the original dataset is that the "sessionIndex" and "rep" attributes
/// have been dropped.
///
/// The datasets is used for comparison with River python library.
///
/// References
/// ----------
/// [^1]: [Keystroke Dynamics - Benchmark Data Set](http://www.cs.cmu.edu/~keystroke/)
pub struct Keystroke;
impl Keystroke {
    pub fn load_data() -> IterCsv<f32, File> {
        let url = "http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv";
        let file_name = "keystroke.csv";
        if !Path::new(file_name).exists() {
            utils::download_csv_file(url, file_name);
        }
        let file = File::open(file_name).unwrap();
        let y_cols = Some(Target::Name("subject".to_string()));
        IterCsv::<f32, File>::new(file, y_cols).unwrap()
    }
}
