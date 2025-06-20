use term_size::dimensions;

pub fn plot_scatter_ascii(xs: &[f64], ys: &[f64], title: Option<&str>) {
    let (term_width, _) = dimensions().unwrap_or((80, 24));
    let width = term_width.saturating_sub(10); // room for labels
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
