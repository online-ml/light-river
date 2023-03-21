pub struct CreditCard;
use std::{path::Path, fs::File};
use crate::datasets::utils;
use crate::stream::iter_csv::{IterCsv, Target};

impl CreditCard {
    pub fn load_credit_card_transactions() -> Result<IterCsv<f32, File>,  Box<dyn std::error::Error>> {
        let url = "https://maxhalford.github.io/files/datasets/creditcardfraud.zip";
        let file_name = "creditcard.csv";

        if !Path::new(file_name).exists() {
            utils::download_zip_file(url, file_name)?
        }
        let file = File::open(file_name).unwrap();
        
        match IterCsv::<f32, File>::new(file, Some(Target::Name("Class".to_string()))) {
            Ok(x) => Ok(x),
            Err(e) => Err(Box::new(e))
        }
    }
}