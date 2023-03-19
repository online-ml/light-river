use csv::{self, Reader, ReaderBuilder};
use num::Float;
use std::io::BufReader;
use std::{collections::HashMap, fs::File, path::Path};
enum Target {
    Name(String),
    MultipleNames(Vec<String>),
}
impl Target {
    fn contains(&self, name: &str) -> bool {
        match self {
            Target::Name(n) => n == name,
            Target::MultipleNames(names) => names.contains(&name.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
enum Data<F: Float + std::str::FromStr> {
    Scalar(F),
    String(String),
}

enum DataStream<F: Float + std::str::FromStr> {
    X(HashMap<String, Data<F>>),
    XY(HashMap<String, Data<F>>, HashMap<String, Data<F>>),
}

impl<F: Float + std::str::FromStr> DataStream<F> {
    fn get_x(&self) -> &HashMap<String, Data<F>> {
        match self {
            DataStream::X(x) => x,
            DataStream::XY(x, _) => x,
        }
    }
    fn get_y(&self) -> Result<&HashMap<String, Data<F>>, &str> {
        match self {
            DataStream::X(_) => Err("No y data"),
            DataStream::XY(_, y) => Ok(y),
        }
    }
}

enum PathOrReader<R> {
    Path(String),
    Reader(csv::Reader<R>),
}

struct IterCsv<F: Float + std::str::FromStr, R: std::io::Read> {
    reader: Reader<R>,
    headers: csv::StringRecord,
    y_cols: Option<Target>,
    data_stream: Option<DataStream<F>>,
}

impl<F: Float + std::str::FromStr, R: std::io::Read> IterCsv<F, R> {
    fn new(reader: R, y_cols: Option<Target>) -> Result<Self, csv::Error> {
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(reader);
        let headers = reader.headers()?.to_owned();
        let data_stream = None;
        Ok(Self {
            reader,
            headers,
            y_cols,
            data_stream,
        })
    }
    fn from_path<P: AsRef<Path>>(path: P, y_cols: Option<Target>) -> Result<Self, csv::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(reader);
        let iter_csv = IterCsv::<F, BufReader<File>>::new(reader, y_cols)?;
        Ok(iter_csv)
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
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn create_temp_file(content: &str) -> PathBuf {
        let dir = tempdir().expect("failed to create temp dir");
        let file_path = dir.path().join("test.csv");
        std::fs::write(&file_path, content).expect("failed to write temp file");
        file_path
    }
    #[test]
    fn test_iter_csv() {
        let path = "";
        //         let content = "Name,Height,Weight,Score
        // Alice,1.6,60.0,90.0
        // Bob,1.8,80.0,85.0
        // Charlie,1.7,70.0,92.5
        // David,1.9,90.0,87.0
        //         ";

        // let cursor = Cursor::new(content);
        // let iter_csv: IterCsv<f32, Cursor<&str>> =
        //     IterCsv::new(cursor, Some(Target::Name("Score".to_string()))).unwrap();

        let iter_csv: IterCsv<f32, Cursor<&str>> =
            IterCsv::new(path, Some(Target::Name("Score".to_string()))).unwrap();
        for data_stream in iter_csv {
            let x_data = data_stream.as_ref().unwrap().get_x();
            let y_data = data_stream.as_ref().unwrap().get_y();
            println!("{:?}", x_data);
            println!("{:?}", y_data);
        }
    }
}
