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

use crate::error::PrimeclueErr;
use crate::rand::GET_RNG;
use crate::serialization::{Deserializable, Serializable, Serializator};
use rand::Rng;
use std::fmt::{Debug, Error, Formatter};

#[derive(Clone)]
pub struct Data<T> {
    size: Size,
    data: Vec<T>,
}

impl<T: Debug> Debug for Data<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "Size: {}, Data: {:?}", self.size, self.data)
    }
}

impl<T: PartialEq> PartialEq for Data<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.size == other.size
    }
}

impl<T> Default for Data<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Data<T> {
    #[must_use]
    pub fn new() -> Data<T> {
        Data { size: Size::new(0, 0), data: Vec::new() }
    }

    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> &T {
        &self.data[row * self.size.columns + column]
    }

    pub fn add_row(&mut self, row: Vec<T>) -> Result<usize, PrimeclueErr> {
        if !self.data.is_empty() && self.size.columns() != row.len() {
            PrimeclueErr::result(format!(
                "Invalid number of columns, found: {}, required: {}",
                row.len(),
                self.size.columns()
            ))
        } else {
            self.size.update(self.size.rows() + 1, row.len());
            self.data.extend(row.into_iter());
            Ok(self.size.rows())
        }
    }

    #[must_use]
    pub fn size(&self) -> &Size {
        &self.size
    }

    #[must_use]
    pub fn row(&self, r: usize) -> Vec<&T> {
        let start = r * self.size.columns();
        let end = start + self.size.columns();
        self.data[start..end].iter().collect()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    rows: usize,
    columns: usize,
}

impl std::fmt::Display for Size {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "rows: {}, columns: {}", self.rows, self.columns)
    }
}

impl Size {
    #[must_use]
    pub fn new(rows: usize, columns: usize) -> Size {
        Size { rows, columns }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn update(&mut self, rows: usize, columns: usize) {
        self.rows = rows;
        self.columns = columns;
    }

    pub fn random_row_column(&self, forbidden_cols: &[usize]) -> (usize, usize) {
        let mut rng = GET_RNG();
        loop {
            let col = rng.gen_range(0, self.columns);
            if !forbidden_cols.contains(&col) {
                return (rng.gen_range(0, self.rows), col);
            }
        }
    }
}

impl Serializable for Size {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.rows, &self.columns]);
    }
}

impl Deserializable for Size {
    fn deserialize(s: &mut Serializator) -> Result<Size, String> {
        let next = s.next_token()?;
        let rows = next.parse().map_err(|e| format!("Unable to parse {}: {}", next, e))?;
        let next = s.next_token()?;
        let columns = next.parse().map_err(|e| format!("Unable to parse {}: {}", next, e))?;
        Ok(Size::new(rows, columns))
    }
}

#[cfg(test)]
mod test {
    use crate::data::{Data, Size};
    use crate::rand::GET_RNG;
    use rand::Rng;

    #[test]
    fn add_row() {
        let mut d = Data::new();
        d.add_row(vec![1.0, 2.0]).unwrap();
        let r = d.add_row(vec![1.0, 2.0, 3.0]);
        assert!(r.is_err());
        let r = d.add_row(vec![1.0]);
        assert!(r.is_err())
    }

    #[test]
    fn random_with_forbidden() {
        let mut rng = GET_RNG();
        let columns = 100;
        let size = Size::new(1, columns);
        for _ in 0..100 {
            let mut forbidden = vec![];
            for _ in 1..rng.gen_range(2, columns) {
                forbidden.push(rng.gen_range(0, columns))
            }
            for _ in 0..100 {
                let (_, column) = size.random_row_column(&forbidden);
                assert!(!forbidden.contains(&column))
            }
        }
    }
}
