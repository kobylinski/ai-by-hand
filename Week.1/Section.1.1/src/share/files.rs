use chrono::Local;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub fn next_available_output_paths(
    extensions: &[&str],
    prefix: Option<&String>,
) -> HashMap<String, PathBuf> {
    fs::create_dir_all("output").expect("Failed to create output directory");

    let timestamp = Local::now().format("%Y-%m-%d").to_string();
    let base = match prefix {
        Some(p) => format!("{}.{}", p, timestamp),
        None => format!("data.{}", timestamp),
    };

    let regexes: Vec<(String, Regex)> = extensions
        .iter()
        .map(|&ext| {
            let pattern = format!(
                r"^{}\.(\d{{3}})\.{}$",
                regex::escape(&base),
                regex::escape(ext)
            );
            (ext.to_string(), Regex::new(&pattern).unwrap())
        })
        .collect();

    let max_enum = fs::read_dir("output")
        .expect("Failed to read output directory")
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|filename| {
            regexes
                .iter()
                .find_map(|(_, re)| re.captures(&filename))
                .and_then(|caps| caps.get(1))
                .and_then(|m| m.as_str().parse::<u32>().ok())
        })
        .max()
        .unwrap_or(0);

    let next_enum = format!("{:03}", max_enum + 1);

    extensions
        .iter()
        .map(|&ext| {
            let filename = format!("{}.{}.{}", base, &next_enum, ext);
            (ext.to_string(), Path::new("output").join(filename))
        })
        .collect()
}
