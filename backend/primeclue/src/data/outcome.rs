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

use crate::serialization::{Deserializable, Serializable, Serializator};
use serde::export::Formatter;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;

#[derive(Clone, Copy, Default, PartialEq, Debug, Eq, Hash, serde::Serialize)]
pub struct Class(u16);

impl Class {
    pub fn new(v: u16) -> Self {
        Class(v)
    }

    pub fn from(b: bool) -> Self {
        if b {
            Class::new(1)
        } else {
            Class::new(0)
        }
    }
}

impl Display for Class {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Serializable for Class {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.0)
    }
}

impl Deserializable for Class {
    fn deserialize(s: &mut Serializator) -> Result<Self, String> {
        let v = u16::deserialize(s)?;
        Ok(Class::new(v))
    }
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Outcome {
    class: Class,
    reward: f32,
    penalty: f32,
}

impl Outcome {
    #[must_use]
    pub fn new(class: Class, reward: f32, penalty: f32) -> Outcome {
        Outcome { class, reward, penalty }
    }

    pub fn calculate_cost(&self, guess: bool, class: Class) -> f32 {
        if class == self.class {
            if guess {
                self.reward
            } else {
                self.penalty
            }
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn class(&self) -> Class {
        self.class
    }

    pub fn set_reward_penalty(&mut self, reward: f32, penalty: f32) {
        self.reward = reward;
        self.penalty = penalty;
    }

    #[must_use]
    pub fn reward(&self) -> f32 {
        self.reward
    }

    #[must_use]
    pub fn penalty(&self) -> f32 {
        self.penalty
    }
}

impl Serializable for Outcome {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.class, &self.reward, &self.penalty]);
    }
}

impl Deserializable for Outcome {
    fn deserialize(s: &mut Serializator) -> Result<Outcome, String> {
        let class = Class::deserialize(s)?;
        let reward = f32::deserialize(s)?;
        let penalty = f32::deserialize(s)?;
        Ok(Outcome { class, reward, penalty })
    }
}

#[must_use]
pub fn sort_guesses(guesses: Vec<f32>, outcomes: &[Outcome]) -> Vec<(f32, Outcome)> {
    let mut outcomes =
        guesses.into_iter().zip(outcomes.iter().copied()).collect::<Vec<(f32, Outcome)>>();
    outcomes.sort_unstable_by(|(first, _), (second, _)| {
        first.partial_cmp(second).unwrap_or(Ordering::Greater)
    });
    outcomes
}

#[cfg(test)]
mod test {
    use crate::data::outcome::Class;
    use crate::data::Outcome;

    #[test]
    fn set_reward_penalty() {
        let mut p = Outcome::new(Class::new(0), 1.0, -1.0);
        let new = 3.0;
        p.set_reward_penalty(new, -new);
        assert_eq!(new, p.reward());
        assert_eq!(-new, p.penalty())
    }

    #[test]
    fn calc_reward_penalty() {
        let class_true = Class::new(1);
        let class_false = Class::new(0);
        let reward = 1.0f32;
        let penalty = -1.0f32;
        let outcome_true = Outcome::new(class_true, reward, penalty);
        let outcome_false = Outcome::new(class_false, reward, penalty);
        let guess_true = true;
        let guess_false = false;
        assert_eq!(outcome_true.calculate_cost(guess_true, class_true), reward);
        assert_eq!(outcome_true.calculate_cost(guess_false, class_true), penalty);
        assert_eq!(outcome_true.calculate_cost(guess_true, class_false), 0.0);
        assert_eq!(outcome_true.calculate_cost(guess_false, class_false), 0.0);

        assert_eq!(outcome_false.calculate_cost(guess_true, class_false), reward);
        assert_eq!(outcome_false.calculate_cost(guess_false, class_false), penalty);
        assert_eq!(outcome_false.calculate_cost(guess_true, class_true), 0.0);
        assert_eq!(outcome_false.calculate_cost(guess_false, class_true), 0.0);
    }
}
