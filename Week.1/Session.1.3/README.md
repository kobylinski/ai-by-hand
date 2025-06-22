# Linear Regression with Gradient Descent

This section implements gradient descent optimization to find the optimal parameters (weight w and bias b) for our linear regression model from previous sections.

## Dictionary

- **MSE (Mean Squared Error)**: A metric that measures the average squared difference between predicted and actual values. Used as the loss function for optimization.
- **MAE (Mean Absolute Error)**: A metric that measures the average absolute difference between predicted and actual values. Used as an additional monitoring metric.
- **Gradient Descent**: An optimization algorithm that iteratively adjusts parameters to minimize a loss function by moving in the direction of steepest descent.
- **Learning Rate**: A hyperparameter that controls how much we adjust our parameters in the direction of the gradient. Too high can cause overshooting, too low makes convergence slow.
- **Epoch**: One complete pass through the training data.

## Prerequisites

- Rust (latest stable version)
- Cargo (comes with Rust)
- Data from Section 1.1 (CSV or JSON file with x,y coordinates)

## Installation

1. Clone the repository (if you haven't already)
2. Navigate to this directory:
```bash
cd "Week 1/Section 1.3"
```
3. Build the project:
```bash
cargo build
```

## Running the Application

Basic usage:
```bash
cargo run -- <path-to-data-file> [learning_rate] [epochs]
```

### Parameters

- `<path-to-data-file>`: Required. Path to input data file (CSV or JSON) from Section 1.1
- `learning_rate`: Optional. Learning rate for gradient descent (default: 0.01)
- `epochs`: Optional. Number of training iterations (default: 1000)

Examples:
```bash
# Run with default parameters
cargo run -- data.csv

# Run with custom learning rate and epochs
cargo run -- data.json 0.005 2000
```

## Input Data Format

The program accepts data in two formats:

### CSV Format
```csv
x1,y1
x2,y2
...
```

### JSON Format
```json
{
    "x": [x1, x2, ...],
    "y": [y1, y2, ...]
}
```

## Output

The program provides:
1. Real-time training progress showing:
   - Current epoch
   - MSE loss
   - Current weight (w) and bias (b) values
2. ASCII plots visualizing:
   - MSE loss vs epoch
   - MAE loss vs epoch

## Implementation Details

### Gradient Descent

The implementation uses:
- **MSE Loss**: L = (1/n) Σ(y_pred - y_true)²
- **Weight Gradient**: ∂L/∂w = (2/n) Σ(x_i * (w*x_i + b - y_i))
- **Bias Gradient**: ∂L/∂b = (2/n) Σ(w*x_i + b - y_i)

Parameters are updated using:
```
w = w - learning_rate * ∂L/∂w
b = b - learning_rate * ∂L/∂b
```

### Special Features

- Adaptive bias gradient scaling (3.0x multiplier) for faster convergence
- Progress logging at customizable intervals
- Dual loss tracking (MSE and MAE) for better training monitoring
- ASCII visualization of training progress

## Troubleshooting

1. If convergence is slow:
   - Increase the learning rate
   - Increase the number of epochs
   - Check if your data is properly normalized

2. If loss is unstable:
   - Decrease the learning rate
   - Check for outliers in your data

3. If you get "File not found":
   - Verify the path to your input data file
   - Make sure the file has the correct extension (.csv or .json)