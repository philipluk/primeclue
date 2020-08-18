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

use primeclue::data::data_set::DataSet;
use primeclue::error::PrimeclueErr;
use primeclue::exec::score::Objective;
use primeclue::exec::training_group::TrainingGroup;
use std::env;
use std::ops::Add;
use std::path::PathBuf;
use std::time::Instant;

// Usage:
// cargo run --release --example time_limit </path/to/imported/data> <minutes>
fn main() -> Result<(), PrimeclueErr> {
    // Get path and time from command line args
    let path = env::args()
        .skip(1)
        .next()
        .ok_or("Need path to data as first arg".to_string())
        .map_err(PrimeclueErr::from)?;
    let minutes = env::args()
        .skip(2)
        .next()
        .ok_or("Need time in minutes as second arg".to_string())
        .map_err(PrimeclueErr::from)?
        .parse::<u64>()
        .map_err(|_| "Time must be an integer".to_string())?;

    // Read data from disk. Data must be in Primeclue's format, i.e. imported to `data.ssd` file.
    let data = DataSet::read_from_disk(&PathBuf::from(path))?;

    // Split data into random training set and testing set. Only training set is used for training,
    // testing set in used to display result to the user.
    let (training_data, verification_data, test_data) = data.shuffle().into_views_split();

    // Get some loop break condition. Here it's time limit, regardless of result or generation count
    let end_time = Instant::now().add(std::time::Duration::from_secs(60 as u64 * minutes));

    // Get training object that later will be used to get classifier. Last argument is score type that
    // we want to maximize for. Other types are accuracy (percentage) or cost.
    let mut training =
        TrainingGroup::new(training_data, verification_data, Objective::AUC, 10, &vec![])?;

    // Actual training happens here
    while Instant::now().lt(&end_time) {
        training.next_generation();
        if let Some(stats) = training.stats() {
            println!("{:?}", stats);
        }
    }

    // Get classifier after training has finished. It will fail if there is no classifier for any of the classes
    let classifier = training.classifier()?;

    // Get classifier's score on unseen data
    match classifier.execute_for_score(&test_data) {
        None => println!("No result on test data"),
        Some(score) => println!(
            "Finished after {} generations, score on unseen data: {}",
            training.generation(),
            score
        ),
    }
    Ok(())
}
