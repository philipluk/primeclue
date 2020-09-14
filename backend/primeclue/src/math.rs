pub fn std_dev(values: &[f32]) -> Vec<f32> {
    let avg = values.iter().sum::<f32>() / values.len() as f32;
    let st_dev = (values.iter().map(|v| (v - avg).powf(2.0)).sum::<f32>()
        / (values.len() - 1) as f32)
        .sqrt();
    values.iter().map(|v| (v - avg) / st_dev).collect()
}

pub fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|v1, v2| v1.partial_cmp(&v2).unwrap());
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle] + values[middle - 1]) / 2.0
    } else {
        values[middle]
    }
}
