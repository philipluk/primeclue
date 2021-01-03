// SPDX-License-Identifier: AGPL-3.0-or-later
/*
   Primeclue: Machine Learning and Data Mining
   Copyright (C) 2020 Łukasz Wojtów

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

use primeclue::data::data_set::{DataSet, DataView};
use primeclue::error::PrimeclueErr;
use primeclue::exec::classifier::Classifier;
use primeclue::exec::score::Objective;
use primeclue::exec::training_group::TrainingGroup;
use primeclue::math::median;
use std::env;
use std::ops::Add;
use std::path::PathBuf;
use std::time::Instant;

// Usage:
// cargo run --release --example gpw_median_check </path/to/imported/data> <minutes>
fn main() -> Result<(), PrimeclueErr> {
    // Get path and time from command line args
    let path = env::args()
        .skip(1)
        .next()
        .ok_or("Need path to data as first arg".to_string())
        .map_err(PrimeclueErr::from)?;
    let seconds = env::args()
        .skip(2)
        .next()
        .ok_or("Need time in seconds as last arg".to_string())
        .map_err(PrimeclueErr::from)?
        .parse::<usize>()
        .map_err(|_| "Time must be an integer".to_string())?;

    let count = 20;
    let mut results = Vec::with_capacity(count);
    for run in 0..count {
        if let Some(result) = check_once(&path, seconds) {
            results.push(result);
            println!("Loop #{} of {} result: {} ", run + 1, count, result);
        } else {
            println!("No result for loop #{}", run + 1);
        }
    }
    if results.is_empty() {
        return Ok(());
    }
    let median = median(&mut results);
    println!("Median result for {:?}: {}", path, median);
    Ok(())
}

fn check_once(path: &str, seconds: usize) -> Option<f32> {
    // Read data from disk. Data must be in Primeclue's format, i.e. imported to `data.ssd` file.
    let data = DataSet::read_from_disk(&PathBuf::from(path)).unwrap();

    // Use this to remove some data, for example stocks not in uptrend
    let data = data.filter(|p| p.data().0.get(0, 7) > 0.0);

    // Split data into random parts. Only training and verification sets are used for training,
    // testing set in used to display result to the user.
    // Here data is split with "marker" to decide which points go to testing set. Data is
    // imported with 'date' column which is used to recognize points in years 2017 and later.
    let (training_data, verification_data, test_data) =
        data.split_with_test_data_marker(|p| p.data().0.get(0, 0) >= 2017_00_00 as f32);

    // Get some loop break condition. Here it's time limit, regardless of result or generation count
    let end_time = Instant::now().add(std::time::Duration::from_secs(seconds as u64));

    // Get training object that later will be used to get classifier. Third argument is objective that
    // we want to maximize for. Other types are accuracy (percentage) or cost.
    let mut training =
        TrainingGroup::new(training_data, verification_data, Objective::AUC, 20, &[0]).ok()?;

    // Actual training happens here
    while Instant::now().lt(&end_time) {
        training.next_generation();
    }
    // Get classifier after training has finished. It will fail if there is no classifier for any of the classes
    let classifier = training.classifier().ok()?;

    // Use the following code to save classifier to disk and use it later for classification
    // let p = PathBuf::from(path);
    // let name = p.file_name().unwrap().to_str().unwrap();
    // classifier.save(&p, name).unwrap();

    // Get classifier's score on unseen data
    // Some(classifier.score(&test_data)?.auc)

    // Use the following code to get average profit for all "true" label predictions
    average_per_true(&test_data, &classifier)
}

fn average_per_true(test_data: &DataView, classifier: &Classifier) -> Option<f32> {
    let predictions = classifier.classify(&test_data);
    let mut true_count = 0;
    let mut sum_profit = 0.0;
    let mut positive = 0;
    let mut negative = 0;

    for (point, prediction) in test_data.outcomes().iter().zip(predictions) {
        if prediction == "true" {
            true_count += 1;
            let change = point.reward() + point.penalty();
            sum_profit += change;
            if change > 0.0 {
                positive += 1;
            } else {
                negative += 1;
            }
        }
    }
    println!("true_count: {}, positive: {}, negative: {}", true_count, positive, negative);
    if true_count > 0 && !sum_profit.is_nan() {
        Some(sum_profit / true_count as f32)
    } else {
        None
    }
}
