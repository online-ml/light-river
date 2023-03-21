use csv::{self, Reader, ReaderBuilder};
use num::Float;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

pub struct IterCsv<F: Float + std::str::FromStr, R: std::io::Read> {
    reader: Reader<R>,
    headers: csv::StringRecord,
    y_cols: Option<Target>,
    data_stream: PhantomData<DataStream<F>>,
}
pub enum Target {
    Name(String),
    MultipleNames(HashSet<String>),
}

impl Target {
    fn contains(&self, name: &str) -> bool {
        match self {
            Target::Name(n) => n == name,
            Target::MultipleNames(names) => names.contains(&name.to_string()),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Data<F: Float + std::str::FromStr> {
    Scalar(F),
    String(String),
}

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


impl<F: Float + std::str::FromStr, R: std::io::Read> IterCsv<F, R> {
    pub fn new(reader: R, y_cols: Option<Target>) -> Result<Self, csv::Error> {
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(reader);
        let headers = reader.headers()?.to_owned();
        Ok(Self {
            reader,
            headers,
            y_cols,
            data_stream: PhantomData,
        })
    }
}

impl<F: Float + std::str::FromStr, R: std::io::Read> Iterator for IterCsv<F, R> {
    type Item = Result<DataStream<F>, csv::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = self.reader.records().next()?;
        match record {
            Ok(record) => {
                let x_data: HashMap<String, Data<F>> = record
                    .iter()
                    .enumerate()
                    .filter_map(|(i, field)| {
                        let header = self.headers.get(i).unwrap().to_string();
                        if self.y_cols.is_some() && self.y_cols.as_ref().unwrap().contains(&header)
                        {
                            None
                        } else {
                            match field.parse::<F>() {
                                Ok(value) => Some((header, Data::Scalar(value))),
                                Err(_) => Some((header, Data::String(field.to_string()))),
                            }
                        }
                    })
                    .collect();
                let y_data = match &self.y_cols {
                    Some(cols) => {
                        let y_data: HashMap<String, Data<F>> = self
                            .headers
                            .iter()
                            .zip(record.iter())
                            .filter_map(|(header, field)| {
                                if cols.contains(header) {
                                    match field.parse::<F>() {
                                        Ok(value) => {
                                            Some((header.to_string(), Data::Scalar(value)))
                                        }
                                        Err(_) => Some((
                                            header.to_string(),
                                            Data::String(field.to_string()),
                                        )),
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Some(y_data)
                    }
                    None => None,
                };
                Some(Ok(match &y_data {
                    Some(y_data) => DataStream::XY(x_data, (*y_data).clone()),
                    None => DataStream::X(x_data),
                }))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maplit::{hashmap, hashset};
    use std::{collections::HashMap, fs::File};
    use tempfile::tempdir;

    fn result() -> Vec<HashMap<String, HashMap<String, Data<f32>>>> {
        vec![
            hashmap! {
                "x".to_string() => hashmap!{
                    "Name".to_string() => Data::<f32>::String("Alice".to_string()),
                    "Height".to_string() => Data::<f32>::Scalar(1.6),
                    "Weight".to_string() => Data::<f32>::Scalar(60.0),
                },
                "y".to_string() => hashmap!{
                    "Score".to_string() => Data::<f32>::Scalar(90.0)
                }
            },
            hashmap! {
                "x".to_string() => hashmap!{
                    "Name".to_string() => Data::<f32>::String("Bob".to_string()),
                    "Height".to_string() => Data::<f32>::Scalar(1.8),
                    "Weight".to_string() => Data::<f32>::Scalar(80.0),
                },
                "y".to_string() => hashmap!{
                    "Score".to_string() => Data::<f32>::Scalar(85.0),
                }
            },
        ]
    }

    #[test]
    fn test_with_target() {
        let content = "Name,Height,Weight,Score\nAlice,1.6,60.0,90.0\nBob,1.8,80.0,85.0";
        let result = result();

        let iter_csv =
            IterCsv::<f32, &[u8]>::new(content.as_bytes(), Some(Target::Name("Score".to_string())))
                .unwrap();

        for (i, line) in iter_csv.enumerate() {
            let line = line.unwrap();
            assert_eq!(line.get_x(), &result[i]["x"]);
            assert_eq!(line.get_y().unwrap(), &result[i]["y"]);
            assert!(line.get_y().is_ok());
        }
    }

    #[test]
    fn test_iter_without_target() {
        let content = "Name,Height,Weight\nAlice,1.6,60.0\nBob,1.8,80.0";
        let result = result();

        let iter_csv = IterCsv::<f32, &[u8]>::new(content.as_bytes(), None).unwrap();

        for (i, line) in iter_csv.enumerate() {
            let line = line.unwrap();
            assert_eq!(line.get_x(), &result[i]["x"]);
            assert!(line.get_y().is_err());
        }
    }

    #[test]
    fn test_iter_with_file() {
        let dir = tempdir().expect("failed to create temp dir");
        let file_path = dir.path().join("test.csv");
        std::fs::write(
            &file_path,
            "Name,Height,Weight\nAlice,1.6,60.0\nBob,1.8,80.0",
        )
        .expect("failed to write temp file");

        let file = File::open(file_path).unwrap();
        let result = result();

        let iter_csv = IterCsv::<f32, File>::new(file, None).unwrap();

        for (i, line) in iter_csv.enumerate() {
            let line = line.unwrap();
            assert_eq!(line.get_x(), &result[i]["x"]);
            assert!(line.get_y().is_err());
        }
    }

    #[test]
    fn test_iter_multiple_target() {
        let content = "Name,Height,Weight\nAlice,1.6,60.0\nBob,1.8,80.0";
        let result = vec![
            hashmap! {
                "x".to_string() => hashmap!{
                    "Name".to_string() => Data::<f32>::String("Alice".to_string()),
                },
                "y".to_string() => hashmap!{
                    "Height".to_string() => Data::<f32>::Scalar(1.6),
                    "Weight".to_string() => Data::<f32>::Scalar(60.0),
                }
            },
            hashmap! {
                "x".to_string() => hashmap!{
                    "Name".to_string() => Data::<f32>::String("Bob".to_string()),
                },
                "y".to_string() => hashmap!{
                    "Height".to_string() => Data::<f32>::Scalar(1.8),
                    "Weight".to_string() => Data::<f32>::Scalar(80.0),
                }
            },
        ];

        let iter_csv = IterCsv::<f32, &[u8]>::new(
            content.as_bytes(),
            Some(Target::MultipleNames(
                hashset! {"Height".to_string(), "Weight".to_string()},
            )),
        )
        .unwrap();

        for (i, line) in iter_csv.enumerate() {
            let line = line.unwrap();
            assert_eq!(line.get_x(), &result[i]["x"]);
            assert_eq!(line.get_y().unwrap(), &result[i]["y"]);
            assert!(line.get_y().is_ok());
        }
    }
}
