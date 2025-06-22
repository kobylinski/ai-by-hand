# Linear Regression Data Generator

This program generates synthetic data points following a linear pattern (y = 2x + 3) with added random noise. The data is visualized using ASCII plotting and exported to both CSV and JSON formats.

## Prerequisites

- Rust (latest stable version)
- Cargo (comes with Rust)

## Installation

1. Clone the repository (if you haven't already)
2. Navigate to this directory:
```bash
cd "Week 1/Section 1.1"
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

## Output

The program generates three types of output:

1. **ASCII Plot**: Displayed directly in the terminal, showing the generated data points
2. **CSV File**: Created in the `output` directory with format `data.YYYY-MM-DD.HH-MM-SS.NNN.csv`
3. **JSON File**: Created in the `output` directory with format `data.YYYY-MM-DD.HH-MM-SS.NNN.json`

### Output Format

- CSV files contain two columns: x and y values
- JSON files contain two arrays: "x" and "y" coordinates
- Files are automatically named with timestamps to prevent overwriting

### Example Output Location

Check the `output` directory for the generated files. The most recent files will have the latest timestamp in their names.

## Verification

After running the program:
1. Look at the ASCII plot in your terminal
2. Check the `output` directory for the newest `.csv` and `.json` files
3. You can view the CSV file in any spreadsheet application
4. You can view the JSON file in any text editor or JSON viewer