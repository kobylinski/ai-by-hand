use std::env;
use term_size::dimensions;
use atty::is as is_terminal;
use base64::{prelude::BASE64_STANDARD, Engine as _};
use charming::{
  component::{Axis, Title}, 
  element::{AxisLabel, AxisType, LineStyle, TextStyle, SplitLine}, 
  series::{Line, Scatter}, 
  Chart, 
  ImageFormat, 
  ImageRenderer
};
use std::error::Error;

/// Enum to represent different types of plots.
/// - `Scatter`: For scatter plots.
/// - `Line`: For line plots.
#[derive(Debug, Copy, Clone)]
pub enum PlotType {
  Scatter,
  Line
}

/// Enum to represent different plot engines based on the terminal type.
/// - `Ascii`: For ASCII-based plotting in terminals.
/// - `Kitty`: For plotting in Kitty terminals using the Kitty graphics protocol.
/// - `ImgCat`: For plotting in terminals that support ImgCat (like iTerm2).
/// - `None`: For unsupported terminals.
#[derive(Debug, Copy, Clone)]
pub enum PlotEngine {
  Ascii,
  Kitty,
  ImgCat,
  None
}

/// Function to check if the terminal is a Kitty terminal
/// by checking the environment variable `TERM` or `KITTY_PID`.
/// Returns true if the terminal is Kitty, false otherwise.
pub fn is_kitty() -> bool {
  env::var("TERM")
      .map(|term| term.contains("kitty"))
      .unwrap_or(false) || env::var("KITTY_PID").is_ok()
}

/// Function to check if the terminal is an ImgCat terminal
/// by checking the environment variable `TERM_PROGRAM`.
pub fn is_img_cat() -> bool {
  env::var("TERM_PROGRAM")
      .map(|term| term.contains("iTerm"))
      .unwrap_or(false)
}

/// Detects the terminal type and returns the appropriate PlotEngine.
/// If the terminal is not supported, it returns `PlotEngine::None`.
/// This function checks if the terminal is a Kitty terminal, ImgCat 
/// terminal, or falls back to ASCII.
pub fn detect_terminal() -> PlotEngine {
  if !is_terminal(atty::Stream::Stdout) {
    return PlotEngine::None;
  }
  if is_kitty() {
    return PlotEngine::Kitty;
  }
  if is_img_cat() {
    return PlotEngine::ImgCat;
  }
  return PlotEngine::Ascii;
}

pub fn plot(
    plot_type: PlotType,
    xs: &[f64],
    ys: &[f64],
    title: Option<&str>,
) {
  if !is_terminal(atty::Stream::Stdout) {
    eprintln!("Error: Plotting requires a terminal.");
    return;
  }
  let (cols, _) = dimensions().unwrap_or((80, 24));
  let width_px = (cols * 8) as u32;
  let height_px = (cols * 8) as u32;
  match detect_terminal() {
    PlotEngine::Kitty => {
      match render_plot_to_png(plot_type, xs, ys, title, width_px, height_px) {
        Ok(png) => {
          let b64 = BASE64_STANDARD.encode(&png);
          print!("\x1B_Gf=100,a=T,t=d;{}\x1B\\", b64);
        }
        Err(err) => 
          eprintln!("Error rendering plot: {}", err)
         
      }
    }
    PlotEngine::ImgCat => {
      match render_plot_to_png(plot_type, xs, ys, title, width_px, height_px) {
        Ok(png) => {
          let b64 = BASE64_STANDARD.encode(&png);
          print!(
            "\x1b]1337;File=inline=1;width={}px;height={}px;preserveAspectRatio=1;:{}\x07\n",
            width_px, height_px, b64
          )
        }
        Err(err) => eprintln!("Error rendering plot: {}", err),
      }
    }
    _ => match plot_type {
      PlotType::Scatter => plot_scatter_ascii(xs, ys, title),
      PlotType::Line => plot_line_ascii(xs, ys, title),
    },
  };
}

pub fn plot_scatter_ascii(xs: &[f64], ys: &[f64], title: Option<&str>) {
    let (term_width, _) = dimensions().unwrap_or((80, 24));
    let width = term_width.saturating_sub(10) - 5; // room for labels
    let height = 20;

    let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut grid = vec![vec![' '; width]; height];

    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let x_pos = (((x - x_min) / (x_max - x_min)) * (width as f64 - 1.0)).round() as usize;
        let y_pos = (((y - y_min) / (y_max - y_min)) * (height as f64 - 1.0)).round() as usize;
        let y_pos = height.saturating_sub(1 + y_pos); // Flip Y

        if y_pos < height && x_pos < width {
            grid[y_pos][x_pos] = '*';
        }
    }

    let y_tick_interval = (y_max - y_min) / height as f64;

    println!("\n");
    if let Some(title) = title {
        println!("{}{}", " ".repeat((width - title.len()) / 2), title);
    }
    println!("{}{}", " ".repeat(width), ""); // Empty line for spacing

    for (i, row) in grid.iter().enumerate() {
        let y_value = y_min + (height - 1 - i) as f64 * y_tick_interval;
        print!("{:>8.2} │", y_value); // Y-axis + label

        for ch in row {
            print!("{}", ch);
        }

        println!();
    }

    // X-axis base line with arrow
    print!("         └"); // Y-origin corner
    for _ in 0..(width - 1) {
        print!("─");
    }
    println!("→"); // X-axis arrow

    // X-axis value labels (every ~10 chars)
    print!("        ");
    let x_tick_interval = (x_max - x_min) / width as f64;
    for i in 0..width {
        if i % 10 == 0 {
            let x_value = x_min + i as f64 * x_tick_interval;
            print!("{:^10.2}", x_value);
        }
    }
    println!();
}
/// Renders a simple ASCII line plot by connecting data points
/// Fallback ASCII line plot (currently identical to scatter ASCII)
pub fn plot_line_ascii(xs: &[f64], ys: &[f64], title: Option<&str>) {
  plot_scatter_ascii(xs, ys, title);
}

/// Draws a scatter or line chart using the Charming crate and returns it as PNG bytes in memory.
///
/// # Arguments
/// * `plot_type` - The type of the chart (Scatter or Line).
/// * `xs` - Slice of x-coordinates.
/// * `ys` - Slice of y-coordinates.
/// * `title` - Optional chart title.
///
/// # Returns
/// A `Result` containing the PNG image as a `Vec<u8>` or an error.
pub fn render_plot_to_png(plot_type: PlotType, xs: &[f64], ys: &[f64], title: Option<&str>, width: u32, height: u32) -> Result<Vec<u8>, Box<dyn Error>> {
    // Ensure xs and ys have the same length
    assert_eq!(xs.len(), ys.len(), "x and y coordinates must have the same length");

    // Create title component
    let title_text = title.unwrap_or_default().to_string();
    let title_component = Title::new()
      .text(&title_text)
      .text_style(
        TextStyle::new()
          .font_size(36)
          .color("#fff")
        )
      .left("center");

    // Create data points for the series (x and y as f64 pairs)
    let data: Vec<Vec<f64>> = xs.iter().copied().zip(ys.iter().copied()).map(|(x,y)| vec![x,y]).collect();

    // Create the chart and configure it
    let mut chart = Chart::new()
        .title(title_component)
        .x_axis(Axis::new()
            .type_(AxisType::Value)// Numerical x-axis data
            .axis_label(
              AxisLabel::new()
                .font_size(28)
                .color("#fff")
            )
            .split_line(
              SplitLine::new()
                .line_style(LineStyle::new().color("#fff").width(1.0).opacity(0.5))
            )
        )
        .y_axis(Axis::new()
            .type_(AxisType::Value) // Ensure y-axis is numerical
            .axis_label(AxisLabel::new().font_size(28).color("#fff"))
            .split_line(
              SplitLine::new()
                .line_style(LineStyle::new().color("#fff").width(1.0).opacity(0.5))
            )
        );

    // Add series based on plot type
    match plot_type {
        PlotType::Scatter => {
            chart = chart.series(
                Scatter::new()
                    .name("Data")
                    .symbol_size(20)
                    .data(data) // Vec<Vec<f64>>
            );
        }
        PlotType::Line => {
            chart = chart.series(
                Line::new()
                    .name("Data")
                    .data(data) // Use (x, y) pairs
            );
        }
    }

    // Render chart to SVG string
    let mut renderer = ImageRenderer::new(width, height);
    let png_bytes = renderer.render_format(ImageFormat::Png, &chart)
        .map_err(|e| format!("Failed to render chart: {}", e))?;
    

    Ok(png_bytes)
}