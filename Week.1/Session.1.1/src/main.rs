//! Linear Regression Data Generator
//!
//! This program generates synthetic data points following a linear pattern (y = 2x + 3)
//! with added random noise. The data is then exported to both CSV and JSON formats
//! and visualized using ASCII plotting.

use rand::Rng;
use serde::Serialize;
use share::{next_available_output_paths, plot, PlotType};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

mod share;

fn main() {
    // Get slope value from command line arguments or generate a random one
    let w = std::env::args().nth(2)
      .and_then(|s| s.parse::<f64>().ok())
      .unwrap_or_else(|| rand::random_range(1.0..10.0));

    // Get bias value from command line arguments or generate a random one
    let b = std::env::args().nth(2)
      .and_then(|s| s.parse::<f64>().ok())
      .unwrap_or_else(|| rand::random::<f64>() * 2.0 - 1.0);

    // Initialize random number generator
    let mut rng = rand::rng();

    // Pre-allocate vectors for efficiency
    let mut xs = Vec::with_capacity(50);
    let mut ys = Vec::with_capacity(50);

    // Generate 50 data points following y = 2x + 3 + noise
    for _ in 0..50 {
        // Generate x values in range [0, 10)
        let x = rng.random_range(0.0..10.0);
        // Add random noise in range [-1, 1)
        let noise = rng.random_range(-1.0..1.0);
        // Calculate y using linear function with noise
        let y = w * x + b + noise;
        xs.push(x);
        ys.push(y);
    }

    // Display data using plot
    println!("\nGenerated Data Points:\n");
    plot(PlotType::Scatter, &xs, &ys, Some("Scatter Plot of Random Data with Noise"));

    // Get unique file paths for outputs and export data
    let prefix = format!("data.{:.2}.{:.2}", w, b);
    let paths = next_available_output_paths(&["csv", "json"], Some(&prefix));
    export_to_csv(&xs, &ys, paths.get("csv").unwrap()).unwrap();
    export_to_json(&xs, &ys, paths.get("json").unwrap()).unwrap();

    println!("\nData exported to CSV and JSON files successfully.");
    println!("CSV Path: {}", paths.get("csv").unwrap().display());
    println!("JSON Path: {}", paths.get("json").unwrap().display());
    println!("\n");
}

/// Exports the generated data points to a CSV file.
///
/// # Arguments
/// * `xs` - Slice containing x-coordinates
/// * `ys` - Slice containing y-coordinates
/// * `path` - Path where the CSV file will be created
///
/// # Returns
/// * `std::io::Result<()>` - Success or error during file operations
fn export_to_csv(xs: &[f64], ys: &[f64], path: &PathBuf) -> std::io::Result<()> {
    // Create a buffered writer for efficient file writing
    let mut writer = BufWriter::new(File::create(path)?);
    // Write each (x,y) pair as a CSV row
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        writeln!(writer, "{},{}", x, y)?;
    }
    Ok(())
}

/// Data structure for serializing scatter plot data to JSON
#[derive(Serialize)]
struct ScatterData<'a> {
    /// x-coordinates of the data points
    x: &'a [f64],
    /// y-coordinates of the data points
    y: &'a [f64],
}

/// Exports the generated data points to a JSON file.
///
/// # Arguments
/// * `xs` - Slice containing x-coordinates
/// * `ys` - Slice containing y-coordinates
/// * `path` - Path where the JSON file will be created
///
/// # Returns
/// * `std::io::Result<()>` - Success or error during file operations
fn export_to_json(xs: &[f64], ys: &[f64], path: &PathBuf) -> std::io::Result<()> {
    // Create a data structure for JSON serialization
    let data = ScatterData { x: xs, y: ys };
    // Convert to pretty-printed JSON
    let json = serde_json::to_string_pretty(&data).unwrap();
    // Write JSON to file
    fs::write(path, json)?;
    Ok(())
}
