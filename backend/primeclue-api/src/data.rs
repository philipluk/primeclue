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

use crate::executor::{Status, StatusCallback};
use crate::user::{read_files, Settings, DATA_DIR, DELETE_IN_PROGRESS};
use primeclue::data::data_set::DataSet;
use primeclue::data::expression::{parse, OutcomeProducer};
use primeclue::data::outcome::Class;
use primeclue::data::{Input, Outcome, Point};
use primeclue::error::PrimeclueErr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Deref;
use std::path::Path;
use std::{fs, thread};

enum ClassProducer {
    Binary(OutcomeProducer),
    Column(usize, HashMap<String, Class>),
}

impl ClassProducer {
    fn class(&self, data: &[Vec<&str>], row: usize) -> Result<Option<Class>, PrimeclueErr> {
        match self {
            ClassProducer::Binary(producer) => {
                Ok(producer.classify(data, row)?.map(Class::from))
            }
            ClassProducer::Column(column, classes) => {
                let v = data[row][*column];
                Ok(Some(*classes.get(v).unwrap()))
            }
        }
    }

    fn all_classes(&self) -> HashMap<Class, String> {
        match self {
            ClassProducer::Binary(_) => {
                let mut classes = HashMap::new();
                classes.insert(Class::from(false), "false".to_string());
                classes.insert(Class::from(true), "true".to_string());
                classes
            }
            ClassProducer::Column(_, current) => {
                let mut classes = HashMap::new();
                for (k, v) in current {
                    classes.insert(*v, k.to_owned());
                }
                classes
            }
        }
    }
}

pub(crate) fn remove(name: &str) -> Result<(), PrimeclueErr> {
    let settings = Settings::new()?;
    let org_path = Path::new(settings.base_dir()).join(DATA_DIR).join(name);
    let remove_path = Path::new(settings.base_dir())
        .join(DATA_DIR)
        .join(format!("{}_{}", DELETE_IN_PROGRESS, name));
    fs::rename(org_path, remove_path.clone())?;
    thread::spawn(move || {
        fs::remove_dir_all(&remove_path).unwrap();
    });
    Ok(())
}

pub(crate) fn list() -> Result<Vec<String>, PrimeclueErr> {
    let settings = Settings::new()?;
    let data_path = settings.data_dir();
    let rd = fs::read_dir(&data_path)?;
    read_files(rd)
}

pub(crate) fn import(
    r: OutcomesRequest,
    status_callback: &StatusCallback,
) -> Result<String, PrimeclueErr> {
    let data = build_data_set(&r)?;
    let settings = Settings::new()?;
    let total = data.len();
    let callback = |count| {
        status_callback(Status::Progress(
            count as f64 / total as f64,
            format!("Saved data set {}", count),
        ));
        Ok(())
    };
    save_data(r.data_name, &data, &settings, callback)?;
    Ok("Done".to_string())
}

#[derive(Deserialize, Debug)]
pub(crate) struct OutcomesRequest {
    content: String,
    expression: String,
    class_column: usize,
    separator: String,
    ignore_first_row: bool,
    rows_per_set: usize,
    import_columns: Vec<bool>,
    data_name: String,
    custom_reward_penalty_columns: bool,
    reward_column: usize,
    penalty_column: usize,
}

impl OutcomesRequest {
    fn extract_reward_penalty(&self, row: &[&str]) -> Result<(f32, f32), PrimeclueErr> {
        if self.custom_reward_penalty_columns {
            let reward = row.get(self.reward_column - 1).ok_or_else(|| {
                PrimeclueErr::from(format!("Unable to access {}'nth column", self.reward_column))
            })?;
            let reward = reward.trim().parse().map_err(|_| {
                PrimeclueErr::from(format!("Unable to parse '{}' to reward", reward))
            })?;
            let penalty = row.get(self.penalty_column - 1).ok_or_else(|| {
                PrimeclueErr::from(format!(
                    "Unable to access {}'nth column",
                    self.penalty_column
                ))
            })?;
            let penalty = penalty.trim().parse().map_err(|_| {
                PrimeclueErr::from(format!("Unable to parse '{}' to penalty", penalty))
            })?;
            Ok((reward, penalty))
        } else {
            Ok((1.0, -1.0))
        }
    }
}

#[derive(Deserialize, Debug)]
struct Rewards {
    correct_true: f32,
    incorrect_true: f32,
    correct_false: f32,
    incorrect_false: f32,
}

#[derive(Serialize, Debug)]
pub(crate) struct OutcomesResponse {
    // TODO refactor to ClassesResponse
    outcomes: Vec<String>, // TODO refactor to classes, change frontend
}

//TODO refactor to ClassRequest - change frontend
pub(crate) fn outcomes(r: &OutcomesRequest) -> Result<OutcomesResponse, PrimeclueErr> {
    let data = split_to_vec(&r.content, &r.separator, r.ignore_first_row);
    let mut classes = Vec::with_capacity(data.len());
    let class_producer = class_producer(r, &data)?;
    if r.ignore_first_row {
        classes.push(String::new());
    }
    for row in 0..data.len() {
        let class = class_producer.class(&data, row)?;
        classes.push(class.map_or(String::new(), |c| c.to_string()))
    }
    Ok(OutcomesResponse { outcomes: classes })
}

pub(crate) fn split_to_vec<'a>(
    content: &'a str,
    separator: &str,
    ignore_first_row: bool,
) -> Vec<Vec<&'a str>> {
    let mut rows: Vec<&str> =
        content.split('\n').map(str::trim).filter(|s| !s.is_empty()).collect();
    if ignore_first_row && !rows.is_empty() {
        rows.remove(0);
    }
    rows.iter().map(|r| r.split(separator).collect()).collect()
}

pub(crate) fn build_numbers_row(
    use_columns: &[bool],
    row_num: usize,
    row: &[&str],
) -> Result<Vec<f32>, PrimeclueErr> {
    let to_import = use_columns
        .iter()
        .zip(row)
        .filter_map(|(&keep, &s)| if keep { Some(s) } else { None })
        .collect::<Vec<&str>>();
    let mut num_row = vec![];
    for value in to_import {
        let n = value.trim().parse().map_err(|err| {
            format!("Unable to parse '{}' to number: row {}: {}", value, row_num + 1, err)
        })?;
        num_row.push(n);
    }
    Ok(num_row)
}

fn build_class_map(
    data: &[Vec<&str>],
    column: usize,
) -> Result<HashMap<String, Class>, PrimeclueErr> {
    let mut classes = HashMap::new();
    for (i, row) in data.iter().enumerate() {
        let v = row
            .get(column)
            .ok_or_else(|| PrimeclueErr::from(format!("No column {} in row {}", column, i)))?
            .deref();
        if !v.is_empty() && !classes.contains_key(v) {
            classes.insert(v.to_string(), Class::new(classes.len() as u16));
        }
    }
    Ok(classes)
}

fn class_producer(
    r: &OutcomesRequest,
    data: &[Vec<&str>],
) -> Result<ClassProducer, PrimeclueErr> {
    if !r.expression.is_empty() {
        Ok(ClassProducer::Binary(parse(&r.expression, data)?))
    } else {
        let column = r.class_column - 1;
        let classes = build_class_map(&data, column)?;
        Ok(ClassProducer::Column(column, classes))
    }
}

fn build_data_set(r: &OutcomesRequest) -> Result<DataSet, PrimeclueErr> {
    let data = split_to_vec(&r.content, &r.separator, r.ignore_first_row);
    let class_producer = class_producer(&r, &data)?;
    let mut numbers = vec![];
    let mut data_set = DataSet::new(class_producer.all_classes());
    for (row_num, row) in data.iter().enumerate() {
        numbers.push(build_numbers_row(&r.import_columns, row_num, row)?);
        if row_num + 1 < r.rows_per_set {
            continue;
        }
        if let Some(outcome) = class_producer.class(&data, row_num)? {
            let (reward, penalty) = r.extract_reward_penalty(row)?;
            data_set.add_data_point(build_data_point(
                r,
                &mut numbers,
                row_num,
                outcome,
                reward,
                penalty,
            )?)?;
        }
    }
    Ok(data_set)
}

fn build_data_point(
    r: &OutcomesRequest,
    numbers: &mut Vec<Vec<f32>>,
    row_num: usize,
    outcome: Class,
    reward: f32,
    penalty: f32,
) -> Result<Point, PrimeclueErr> {
    let id = create_input_data(row_num, numbers, r.rows_per_set)?;
    let pd = Outcome::new(outcome, reward, penalty);
    Ok(Point::new(id, pd))
}

fn save_data<F>(
    name: String,
    data: &DataSet,
    settings: &Settings,
    callback: F,
) -> Result<(), PrimeclueErr>
where
    F: FnMut(usize) -> Result<(), PrimeclueErr>,
{
    let path = Path::new(settings.base_dir()).join(DATA_DIR).join(name);
    data.save_to_disk(&path, callback)
}

fn create_input_data(
    line: usize,
    floats: &[Vec<f32>],
    rows_per_set: usize,
) -> Result<Input, PrimeclueErr> {
    let start_line = 1 + line - rows_per_set;

    let mut id = Input::new();
    for floats_line in floats[start_line..=line].iter() {
        id.add_row(floats_line.clone()).map_err(|e| {
            PrimeclueErr::from(format!("Unable to import line: {:?}: {}", floats_line, e))
        })?;
    }
    Ok(id)
}
