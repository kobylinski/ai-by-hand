use serde::Deserialize;
use serde_json;
use csv::ReaderBuilder;
use regex::Regex;
use std::path::PathBuf;

mod shared;
use shared::plot_scatter_ascii;

// Define structure for CSV data points - each row contains x,y coordinates
// Deserialize trait allows automatic parsing from CSV
#[derive(Debug, Deserialize)]
struct CsvData {
    x: f64,  // x-coordinate
    y: f64,  // y-coordinate
}

// Define structure for JSON data format - contains arrays of x,y coordinates
// Deserialize trait allows automatic parsing from JSON
#[derive(Debug, Deserialize)]
struct JsonData {
    x: Vec<f64>,  // Array of x-coordinates
    y: Vec<f64>,  // Array of y-coordinates
}

fn main() {
    // Get the data file path from command line arguments
    let path = std::env::args()
        .nth(1)
        .expect("Please provide path to data file");

    // Convert to PathBuf and verify file exists
    let path = PathBuf::from(&path);
    if !path.exists() {
        panic!("File does not exist: {}", path.display());
    }

    // Extract filename from path
    let filename = path.file_name()
      .and_then(|name| name.to_str())
      .expect("Invalid file name");

    println!("Processing file: {}", filename);

    // Define regex pattern to extract parameters from filename
    // Format: data.<w>.<b>.<timestamp>.<extension>
    let pattern = Regex::new(r"^data\.(\d+\.\d+)\.(-?\d+\.\d+)\.\d{4}\-\d{2}-\d{2}\.\d{3}\.(csv|json)$").unwrap();
    let result = pattern.captures(&filename);
    
    // Extract w (slope), b (intercept), and file extension from filename
    let (w, b, ext) = match result {
        Some(caps) => (
            caps.get(1).unwrap().as_str().parse::<f64>().unwrap(),
            caps.get(2).unwrap().as_str().parse::<f64>().unwrap(),
            caps.get(3).unwrap().as_str(),
        ),
        None => panic!("Invalid file name format"),
    };

    // Load data points from file based on extension
    let (x, y): (Vec<f64>, Vec<f64>) = match ext {
        "csv" => {
            // Read CSV file without headers
            let mut rdr = ReaderBuilder::new()
                .has_headers(false)
                .from_path(&path)
                .expect("Failed to read CSV file");
            let mut x = Vec::new();
            let mut y = Vec::new();
            // Parse each row into x,y coordinates
            for result in rdr.deserialize() {
                let record: CsvData = result.expect("Failed to deserialize CSV record");
                x.push(record.x);
                y.push(record.y);
            }
            (x, y)
        }
        "json" => {
            // Read and parse JSON file directly into vectors
            let file_content = std::fs::read_to_string(&path).expect("Failed to read JSON file");
            let result: JsonData = serde_json::from_str(&file_content).expect("Failed to deserialize JSON data");
            (result.x, result.y)
        }
        _ => panic!("Unsupported file extension"),
    };

    // Get learning rate from command line or use random value
    let learning_rate = std::env::args().nth(2)
      .and_then(|s| s.parse::<f64>().ok())
      .unwrap_or_else(|| rand::random_range(0.001..0.01));

    // Get number of epochs from command line or use random value
    let epochs = std::env::args().nth(3)
      .and_then(|s| s.parse::<u32>().ok())
      .unwrap_or_else(|| rand::random_range(1..100) * 100);

    // Set logging interval (default: every 10% of epochs)
    let log_interval = std::env::args().nth(4)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(epochs / 10);

    println!("w: {}, b: {}, learning_rate: {}, epochs: {}, log_interval: {}", w, b, learning_rate, epochs, log_interval);
    
    // Initialize model parameters and logging vectors
    let mut w_pred = 0.0; // predicted slope
    let mut b_pred = 0.0; // predicted intercept
    let mut log_epochs = Vec::new();  // store epoch numbers for plotting
    let mut log_losses = Vec::new();  // store MSE losses for plotting
    let mut log_losses_b = Vec::new(); // store MAE losses for plotting

    // Training loop
    for epoch in 1..epochs {
        // Calculate gradients for w and b
        let dw = gradient_w(&x, &y, w_pred, b_pred);
        let db = gradient_b(&x, &y, w_pred, b_pred);

        // Update parameters using gradient descent
        w_pred -= learning_rate * dw;
        b_pred -= learning_rate * 3.0 * db;  // Note: 3.0 multiplier for faster convergence

        // Log progress at specified intervals
        if epoch % log_interval == 0 || epoch == 1 || epoch == epochs {
            // Calculate Mean Squared Error loss
            let loss = x.iter().zip(&y)
                .map(|(&xi, &yi)| {
                    let y_hat = w_pred * xi + b_pred;
                    (y_hat - yi).powi(2)
                })
                .sum::<f64>() / (x.len() as f64);
            
            // Calculate Mean Absolute Error loss
            let loss_b = x.iter().zip(&y)
                .map(|(&xi, &yi)| {
                    let y_hat = w_pred * xi + b_pred;
                    (y_hat - yi).abs()
                })
                .sum::<f64>() / (x.len() as f64);

            // Store values for plotting
            log_epochs.push(epoch as f64);
            log_losses.push(loss);
            log_losses_b.push(loss_b);

            // Print progress
            println!(
                "Epoch {:>4} | Loss: {:>10.6} | w: {:>8.4} | b: {:>8.4}",
                epoch, loss, w_pred, b_pred
            );
        }
    }

    // Plot training progress
    plot_scatter_ascii(&log_epochs, &log_losses, Some("Loss vs Epoch"));
    plot_scatter_ascii(&log_epochs, &log_losses_b, Some("Loss B vs Epoch"));
}

// Calculate gradient for weight parameter (slope)
// Uses derivative of MSE with respect to w
fn gradient_w(x: &[f64], y: &[f64], w: f64, b: f64) -> f64 {
    x.iter().zip(y).map(|(xi, yi)|xi* ((w * xi + b) - yi)).sum::<f64>() * (2.0 / x.len() as f64)
}

// Calculate gradient for bias parameter (intercept)
// Uses derivative of MSE with respect to b
fn gradient_b(x: &[f64], y: &[f64], w: f64, b: f64) -> f64 {
    x.iter().zip(y).map(|(xi, yi)| (w * xi + b) - yi).sum::<f64>() * (2.0 / x.len() as f64)
}