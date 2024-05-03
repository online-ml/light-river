use reqwest::blocking::Client;
use std::fs::File;
use std::io;
use std::path::Path;
use zip::ZipArchive;

pub(crate) fn download_zip_file(
    url: &str,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let response = client.get(url).send()?;
    let body = response.bytes()?;

    let mut zip_archive = ZipArchive::new(std::io::Cursor::new(body))?;

    let csv_index = zip_archive
        .file_names()
        .position(|name| name.ends_with(file_name))
        .ok_or(format!("{} not found in zip archive", file_name))?;

    let tmp_file_name = format!("tpm_{}", file_name);

    let mut csv_file = zip_archive.by_index(csv_index)?;
    let mut tmp_file = File::create(&tmp_file_name)?;
    std::io::copy(&mut csv_file, &mut tmp_file)?;

    let tmp_path = Path::new(&tmp_file_name);
    let data_path = Path::new(file_name);
    std::fs::rename(tmp_path, data_path)?;

    Ok(())
}

pub(crate) fn download_csv_file(url: &str, file_name: &str) {
    let resp = reqwest::blocking::get(url).expect("request failed");
    let body = resp.text().expect("body invalid");
    let mut out = File::create(file_name).expect("failed to create file");
    io::copy(&mut body.as_bytes(), &mut out).expect("failed to copy content");
}
