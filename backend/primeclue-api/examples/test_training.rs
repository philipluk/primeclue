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
use primeclue::data::outcome::Class;
use primeclue::data::{Input, Outcome, Point};
use primeclue::exec::score::Objective;
use primeclue::exec::training_group::TrainingGroup;
use primeclue::rand::GET_RNG;
use rand::Rng;
use std::collections::HashMap;

// Run with : cargo run --release --example test_training
// Average time to success: 798

fn main() {
    let mut sum = 0.0;
    let count = 100;
    for attempt in 1..count + 1 {
        sum += training_success(attempt);
        println!("Average time to success after {} attempts: {}", attempt, sum / attempt as f64);
    }
    println!("Average time to success: {}", sum / count as f64);
}

fn training_success(attempt: usize) -> f64 {
    let mut classes = HashMap::new();
    classes.insert(Class::new(0), "A".to_owned());
    classes.insert(Class::new(1), "B".to_owned());
    classes.insert(Class::new(2), "C".to_owned());
    classes.insert(Class::new(3), "D".to_owned());
    let string_classes = classes.iter().map(|(c, s)| (s.clone(), *c)).collect::<HashMap<_, _>>();
    let mut data_set = DataSet::new(classes);
    let mut rng = GET_RNG();
    for _ in 0..4 * 250 {
        let a = rng.gen_range(0, 100);
        let b = rng.gen_range(0, 100);
        let c = rng.gen_range(0, 100);

        let output = if a % 15 == 0 {
            "A"
        } else if (b + 2) % 5 == 0 {
            "B"
        } else if (c + 5) % 3 == 0 {
            "C"
        } else {
            "D"
        };

        let point = Point::new(
            Input::from_vector(vec![vec![a as f32, b as f32, c as f32]]).unwrap(),
            Outcome::new(*string_classes.get(output).unwrap(), 1.0, -1.0),
        );
        data_set.add_data_point(point).unwrap();
    }

    let (training_data, verification_data, test_data) = data_set.shuffle().into_views_split();

    let mut training = TrainingGroup::new(
        training_data,
        verification_data,
        Objective::Accuracy,
        10,
        &Vec::new(),
    )
    .unwrap();
    let required_score = 90.0;
    let start = std::time::Instant::now();
    loop {
        training.next_generation();
        if let Ok(classifier) = training.classifier() {
            if let Some(score) = classifier.applied_score(&test_data) {
                if let Some(stats) = training.stats() {
                    println!(
                        "Testing training #{}, generation: {}, unseen: {}",
                        attempt, stats.generation, score.accuracy
                    );
                }
                if score.accuracy > required_score {
                    let elapsed =
                        std::time::Instant::now().duration_since(start).as_millis() as f64;
                    return elapsed;
                }
            }
        }
    }
}
