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

#![forbid(unsafe_code)]
#![warn(
anonymous_parameters,
missing_copy_implementations,
missing_debug_implementations,
// missing_docs,
nonstandard_style,
rust_2018_idioms,
single_use_lifetimes,
trivial_casts,
trivial_numeric_casts,
unreachable_pub,
unused_extern_crates,
unused_qualifications,
variant_size_differences,
)]

#[macro_use]
extern crate lazy_static;

pub mod data;
pub mod error;
pub mod exec;
pub mod math;
pub mod rand;
pub mod serialization;
pub mod user;
