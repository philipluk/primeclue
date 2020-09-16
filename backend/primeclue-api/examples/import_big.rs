use primeclue::data::importer::{build_data_set, split_to_vec, ClassRequest};
use primeclue::error::PrimeclueErr;
use primeclue::user::Settings;
use std::fs;
use std::path::PathBuf;

// This example will import a file. This is done mainly to avoid browser GUI with big files.
// Usage:
// cargo run --release --example import_big
fn main() -> Result<(), PrimeclueErr> {
    // Data to import
    let path = "/tmp/training.csv";
    let name = "mnist_fashion_training";
    // Get string content
    println!("Reading file");
    let content = fs::read(PathBuf::from(path))?;
    let content = String::from_utf8(content)
        .map_err(|e| format!("Error converting file content: {:?}", e))?;
    // A bit hideous way to import all but last columns
    println!("Building column list");
    let line = &split_to_vec(&content, ",", false)[0];
    let len = line.len();
    let mut import_columns = vec![];
    for _ in 0..len - 1 {
        import_columns.push(true);
    }
    import_columns.push(false);

    let class_request = ClassRequest {
        content,
        expression: "".to_string(),
        class_column: len,
        separator: ",".to_string(),
        ignore_first_row: false,
        rows_per_set: 1,
        import_columns,
        data_name: name.to_string(),
        custom_reward_penalty_columns: false,
        reward_column: 0,
        penalty_column: 0,
    };
    println!("Building data set");
    let data_set = build_data_set(&class_request)?;
    let path = Settings::new()?.data_dir().join(name);
    println!("Saving to {:?}", path);
    data_set.save_to_disk(&path, |p| {
        if p % 100 == 0 {
            println!("Saved {} points", p);
        }
        Ok(())
    })?;
    println!("Saved {} points", data_set.len());
    println!("Import successful");
    Ok(())
}
