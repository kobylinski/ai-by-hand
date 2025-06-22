
# Linear Prediction Function Implementation

This program implements a forward function (y = wx + b) for linear regression and computes Mean Squared Error (MSE) between predicted and actual values. It demonstrates basic concepts of linear regression prediction and loss calculation.

## Dictionary

**MSE** - Mean Squared Error, a metric used to measure the average squared difference between predicted and actual values in machine learning models.

## Prerequisites

- Rust (latest stable version)
- Cargo (comes with Rust)
- Data from Section 1.1 (CSV or JSON file)

## Installation

1. Clone the repository (if you haven't already)
2. Navigate to this directory:
```bash
cd "Week 1/Section 1.2"
```
3. Build the project:
```bash
cargo build
```

## Running the Application

To run the program:
```bash
cargo run
```

## Input Data

The program expects input data from Section 1.1. It can read either:
- CSV file from the previous section's output directory
- JSON file from the previous section's output directory

## Program Features

1. **Forward Function Implementation**
   - Implements y = wx + b prediction function
   - Configurable weight (w) and bias (b) parameters
   - Processes input data points to generate predictions

2. **MSE Calculation**
   - Computes Mean Squared Error between predictions and actual values
   - Shows loss value to evaluate prediction quality
   - Helps understand model accuracy

3. **Multiple Parameter Tests**
   - Try different values for w and b
   - Observe how different parameters affect predictions
   - Compare MSE values across different configurations

## Output

The program provides:

1. **Terminal Output**:
   - Parameter values (w and b) being tested
   - MSE calculation results
   - Summary statistics of predictions

2. **Visual Representation**:
   - ASCII plot showing actual vs. predicted values
   - Clear visualization of how well the model fits the data

## Understanding the Results

1. **MSE Interpretation**:
   - Lower MSE indicates better fit
   - Compare MSE across different parameter settings
   - Use this to understand which parameters work better

2. **Visual Analysis**:
   - Look at the ASCII plot to see prediction line
   - Compare how well it fits the actual data points
   - Observe over/underfitting patterns

## Tips for Use

1. Start with simple parameter values (e.g., w=1, b=0)
2. Gradually adjust parameters to see how MSE changes
3. Pay attention to both MSE values and visual representation
4. Try to find parameters that minimize MSE

## Troubleshooting

If you encounter issues:
1. Verify input data file exists and is accessible
2. Check parameter values are within reasonable ranges
3. Ensure you have required dependencies installed
4. Check terminal output for any error messages