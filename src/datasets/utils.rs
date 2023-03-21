use std::io::Read;
use std::fs::File;
use std::path::Path;
use zip::ZipArchive;
use reqwest::blocking::Client;

pub fn download_zip_file(url: &str, file_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create a reqwest client
    let client = Client::new();

    // Send a GET request to the provided URL
    let response = client.get(url).send()?;

    // Read the response body into a byte vector
    let body = response.bytes()?;

    // Create a ZipArchive from the byte vector
    let mut zip_archive = ZipArchive::new(std::io::Cursor::new(body))?;

    // Get the index of the data.csv file in the zip archive
    let csv_index = zip_archive
        .file_names()
        .position(|name| name.ends_with(file_name))
        .ok_or(format!("{} not found in zip archive", file_name))?;

    let tmp_file_name = format!("tpm_{}", file_name);
    // Extract the data.csv file to a temporary file
    let mut csv_file = zip_archive.by_index(csv_index)?;
    let mut tmp_file = File::create(&tmp_file_name)?;
    std::io::copy(&mut csv_file, &mut tmp_file)?;

    // Rename the temporary file to data.csv
    let tmp_path = Path::new(&tmp_file_name);
    let data_path = Path::new(file_name);
    std::fs::rename(tmp_path, data_path)?;

    Ok(())
}