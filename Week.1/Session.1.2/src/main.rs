// Import required external crates
use serde::Deserialize;  // For deserializing data from files
use serde_json;          // For JSON parsing
use csv::ReaderBuilder;  // For CSV file handling

// Import local plotting module
mod share;
use share::{plot, PlotType};

// Define structure for CSV data points - each row contains x,y coordinates
#[derive(Debug, Deserialize)]
struct CsvData {
    x: f64,  // x-coordinate
    y: f64,  // y-coordinate
}

// Define structure for JSON data format - contains arrays of x,y coordinates
#[derive(Debug, Deserialize)]
struct JsonData {
    x: Vec<f64>,  // Array of x-coordinates
    y: Vec<f64>,  // Array of y-coordinates
}

fn main() {
    // Get file path from command line arguments
    let path = std::env::args()
        .nth(1)
        .expect("Please provide path to data file");
    
    // Extract file extension to determine file type
    let extension = std::path::Path::new(&path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    // Get bias parameter from command line or generate random value between -1 and 1
    let b = std::env::args().nth(2)
      .and_then(|s| s.parse::<f64>().ok())
      .unwrap_or_else(|| rand::random::<f64>() * 2.0 - 1.0);

    // Load data from file based on file extension
    let (x, y): (Vec<f64>, Vec<f64>) = match extension {
        // Handle CSV file format
        "csv" => {
            let mut rdr = ReaderBuilder::new()
                .has_headers(false)
                .from_path(path)
                .expect("Failed to read CSV file");
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            // Read each row and extract x,y values
            for result in rdr.deserialize() {
                let record: CsvData = result.expect("Failed to deserialize CSV record");
                xs.push(record.x);
                ys.push(record.y);
            }
            (xs, ys)
        }
        // Handle JSON file format
        "json" => {
            let file = std::fs::File::open(path).expect("Failed to open JSON file");
            let data: JsonData = serde_json::from_reader(file).expect("Failed to deserialize JSON");
            (data.x, data.y)
        }
        // Error for unsupported file types
        _ => panic!("Unsupported file format: {}", extension),
    };

    // Initialize vectors to store weights and corresponding mean squared errors
    let mut weights = Vec::with_capacity(10);
    let mut means = Vec::with_capacity(10);

    // Test different weight values from 0 to 19
    for w in 0..20 {
        // Calculate predictions using current weight and bias
        let predict = predict(&x, w as f64, b);
        // Calculate mean squared error for current predictions
        let mse = mean_squared_error(&predict, &y);
        weights.push(w as f64);
        means.push(mse);
    }

    // Plot the relationship between weights and mean squared errors
    plot(PlotType::Line,&weights, &means, Some("Mean Squared Error vs Weight"));
    
    // Find and print the weight that resulted in the lowest MSE
    println!("Best weight: {}", weights[means.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0]);
    // Print the lowest MSE achieved
    println!("Best MSE: {}", means.iter().cloned().fold(f64::INFINITY, f64::min));
    // Print the bias value used in the linear model
    println!("Bias (b): {}", b);
}

/// Predicts y-values using linear function y = wx + b
/// 
/// # Arguments
/// * `x` - Input values (features)
/// * `w` - Weight parameter for the linear model
/// * `b` - Bias parameter for the linear model
fn predict(x: &Vec<f64>, w: f64, b: f64) -> Vec<f64> {
    x.iter().map(|xi| w * xi + b).collect()
}

/// Calculates the mean squared error between predicted and true values
/// 
/// # Arguments
/// * `y_pred` - Vector of predicted values
/// * `y_true` - Vector of actual (true) values
/// 
/// # Panics
/// Panics if the predicted and true vectors have different lengths
fn mean_squared_error(y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64 {
    if y_pred.len() != y_true.len() {
        panic!("Predicted and true values must have the same length");
    }
    // Calculate average of squared differences between predicted and true values
    let sum: f64 = y_pred.iter().zip(y_true).map(|(pred, true_val)| (pred - true_val).powi(2)).sum();
    sum / y_pred.len() as f64
}