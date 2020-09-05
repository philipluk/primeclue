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

use crate::data::Data;
use crate::data::InputShape;
use crate::error::PrimeclueErr;
use crate::serialization::{Deserializable, Serializable, Serializator};

#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    data: Data<f32>,
}
impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

impl Input {
    #[must_use]
    pub fn new() -> Input {
        Input { data: Data::new() }
    }

    pub fn from_vector(data: Vec<Vec<f32>>) -> Result<Input, PrimeclueErr> {
        let mut input_data = Input::new();

        for (id, row) in data.into_iter().enumerate() {
            input_data.add_row(row).map_err(|e| {
                PrimeclueErr::from(format!("Unable to import row {}: {} ", id + 1, e))
            })?;
        }
        Ok(input_data)
    }

    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> f32 {
        *self.data.get(row, column)
    }

    pub fn add_row(&mut self, row: Vec<f32>) -> Result<usize, PrimeclueErr> {
        self.data.add_row(row)
    }

    #[must_use]
    pub fn input_shape(&self) -> &InputShape {
        self.data.input_shape()
    }

    #[must_use]
    pub fn row(&self, r: usize) -> Vec<f32> {
        self.data.row(r).into_iter().copied().collect()
    }
}

impl Serializable for Input {
    fn serialize(&self, s: &mut Serializator) {
        s.add(self.data.input_shape());
        for i in 0..self.input_shape().rows() {
            let row = self.row(i);
            row.serialize(s);
        }
    }
}

impl Deserializable for Input {
    fn deserialize(s: &mut Serializator) -> Result<Input, String> {
        let mut id = Input::new();
        let input_shape = InputShape::deserialize(s)?;
        for _ in 0..input_shape.rows() {
            let row = Vec::deserialize(s)?;
            id.add_row(row).map_err(|e| e.to_string())?;
        }
        Ok(id)
    }
}
