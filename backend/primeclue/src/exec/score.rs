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

use crate::data::outcome::Class;
use crate::data::Outcome;
use crate::serialization::{Deserializable, Serializable, Serializator};
use core::fmt;
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;

#[derive(Debug, PartialEq, Copy, Clone, serde::Deserialize, serde::Serialize)]
pub enum Objective {
    Cost,
    AUC,
    Accuracy,
}

impl Serializable for Objective {
    fn serialize(&self, s: &mut Serializator) {
        let var = match self {
            Objective::Cost => "Cost",
            Objective::AUC => "AUC",
            Objective::Accuracy => "Accuracy",
        };
        s.add_str(var);
    }
}

impl Deserializable for Objective {
    fn deserialize(s: &mut Serializator) -> Result<Objective, String> {
        let t = s.next_token()?;
        match t.as_ref() {
            "Cost" => Ok(Objective::Cost),
            "AUC" => Ok(Objective::AUC),
            "Accuracy" => Ok(Objective::Accuracy),
            _ => Err(format!("Invalid token for ScoreType: {}", t)),
        }
    }
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Objective::Cost => "Cost",
            Objective::AUC => "AUC",
            Objective::Accuracy => "Accuracy",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Threshold {
    value: f32,
}

impl Threshold {
    pub fn bool(self, f: f32) -> Option<bool> {
        if !f.is_finite() {
            None
        } else {
            Some(f >= self.value)
        }
    }

    pub fn new(f: f32) -> Self {
        Threshold { value: f }
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

impl Serializable for Threshold {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.value)
    }
}

impl Deserializable for Threshold {
    fn deserialize(s: &mut Serializator) -> Result<Self, String> {
        Ok(Threshold { value: f32::deserialize(s)? })
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Score {
    objective: Objective,
    class: Class,
    value: f32,
    threshold: Threshold,
}

impl Score {
    #[must_use]
    pub fn new(objective: Objective, class: Class, value: f32, threshold: Threshold) -> Score {
        Score { objective, class, value, threshold }
    }

    #[must_use]
    pub fn threshold(&self) -> Threshold {
        self.threshold
    }

    #[must_use]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    #[must_use]
    pub fn class(&self) -> Class {
        self.class
    }

    #[must_use]
    pub fn value(&self) -> f32 {
        self.value
    }

    #[must_use]
    pub fn from(objective: Objective, class: Class, value: f32, threshold: Threshold) -> Self {
        Score { objective, class, value, threshold }
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.objective == other.objective {
            let diff = (self.value / other.value - 1.0).abs() > 0.001;
            if diff {
                self.value.partial_cmp(&other.value)
            } else {
                Some(Equal)
            }
        } else {
            None
        }
    }
}

impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            None => false,
            Some(ord) => ord == Equal,
        }
    }
}

impl Serializable for Score {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.objective, &self.class, &self.value, &self.threshold]);
    }
}

impl Deserializable for Score {
    fn deserialize(s: &mut Serializator) -> Result<Score, String> {
        let objective = Objective::deserialize(s)?;
        let class = Class::deserialize(s)?;
        let value = f32::deserialize(s)?;
        let threshold = Threshold::deserialize(s)?;
        Ok(Score { objective, class, value, threshold })
    }
}

#[must_use]
pub fn calc_score(
    outcomes: &[(f32, Outcome)],
    threshold: Threshold,
    class: Class,
    objective: Objective,
) -> Score {
    let value = match objective {
        Objective::AUC => calculate_auc(&outcomes, class),
        Objective::Accuracy => calculate_accuracy(threshold, &outcomes, class),
        Objective::Cost => calculate_cost(threshold, &outcomes, class),
    };
    Score { objective, class, value, threshold }
}

fn calculate_auc(outcomes: &[(f32, Outcome)], class: Class) -> f32 {
    let mut incorrect_count = 0_usize;
    let mut correct_count = 0_usize;
    let mut total_incorrect = 0_usize;
    for (_, outcome) in outcomes {
        if outcome.class() == class {
            correct_count += 1;
            total_incorrect += incorrect_count;
        } else {
            incorrect_count += 1;
        }
    }
    total_incorrect as f32 / (correct_count * incorrect_count) as f32
}

fn calculate_accuracy(threshold: Threshold, outcomes: &[(f32, Outcome)], class: Class) -> f32 {
    let mut correct = 0;
    let mut total = 0;
    for (guess, outcome) in outcomes.iter() {
        if let Some(guess_bool) = threshold.bool(*guess) {
            total += 1;
            if (outcome.class() == class && guess_bool)
                || (outcome.class() != class && !guess_bool)
            {
                correct += 1;
            }
        }
    }
    correct as f32 / total as f32
}

#[must_use]
pub fn threshold(outcomes: &[(f32, Outcome)], class: Class) -> Threshold {
    let none_class_count =
        outcomes.iter().map(|(_, o)| o.class()).filter(|c| c != &class).count();
    if none_class_count == 0 {
        Threshold::new(outcomes[0].0)
    } else if none_class_count == outcomes.len() {
        Threshold::new(2.0 * outcomes[outcomes.len() - 1].0.abs())
    } else {
        Threshold::new(outcomes[none_class_count].0)
    }
}

fn calculate_cost(threshold: Threshold, outcomes: &[(f32, Outcome)], class: Class) -> f32 {
    let mut cost = 0.0;
    for (guess, outcome) in outcomes.iter() {
        if let Some(b) = threshold.bool(*guess) {
            cost += outcome.calculate_score(b, class);
        }
    }
    cost
}

#[cfg(test)]
mod test {
    use crate::data::outcome::Class;
    use crate::data::Outcome;
    use crate::exec::score::Objective::{Accuracy, Cost, AUC};
    use crate::exec::score::{threshold, Score, Threshold};
    use crate::serialization::serializator::test::test_serialization;
    use std::cmp::Ordering::Equal;

    #[test]
    fn serialize_tree_score() {
        test_serialization(Accuracy);
        test_serialization(Score {
            objective: Cost,
            class: Class::from(true),
            value: 0.0,
            threshold: Threshold::new(1.0),
        });
        test_serialization(Score {
            objective: AUC,
            class: Class::from(false),
            value: 1.0,
            threshold: Threshold::new(0.0),
        });
        test_serialization(Score {
            objective: Accuracy,
            class: Class::new(3),
            value: -11.5,
            threshold: Threshold::new(-10.3),
        });
    }

    #[test]
    fn cmp_incompatible_score() {
        let class = Class::new(0);
        let s1 = Score { objective: AUC, class, value: 1.0, threshold: Threshold::new(0.0) };
        let s2 = Score { objective: Cost, class, value: 2.0, threshold: Threshold::new(0.0) };
        assert_eq!(s1.partial_cmp(&s2), None);
        assert_eq!(s1 > s2, false);
        assert_eq!(s1 < s2, false);
        assert_eq!(s1 == s2, false);
    }

    #[test]
    fn cmp_close_score_small() {
        let class = Class::new(0);
        let s1 = Score { objective: AUC, class, value: 0.7000, threshold: Threshold::new(0.0) };
        let s2 = Score { objective: AUC, class, value: 0.7005, threshold: Threshold::new(0.0) };
        assert_eq!(s1.partial_cmp(&s2).unwrap(), Equal);
        assert_eq!(s1, s2);
    }

    #[test]
    fn cmp_close_score_big() {
        let class = Class::new(0);
        let s1 =
            Score { objective: AUC, class, value: 1_000_000.0, threshold: Threshold::new(0.0) };
        let s2 =
            Score { objective: AUC, class, value: 1_000_001.0, threshold: Threshold::new(0.0) };
        assert_eq!(s1.partial_cmp(&s2).unwrap(), Equal);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_threshold() {
        // Classification uses >= comparison
        // So if threshold is 3.14 then everything that is greater or equal 3.14 will be classified as true
        // Outcomes must be sorted in ascending order of first part of tuple
        let outcomes = vec![
            (-10.0, Outcome::new(Class::from(false), 1.0, -1.0)),
            (-3.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-1.0, Outcome::new(Class::from(true), 1.0, -1.0)),
        ];
        let t = threshold(&outcomes, Class::from(true)); // if false_count == 1 then return 2nd value
        assert_eq!(t.value(), -3.0);

        let outcomes = vec![
            (-10.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-3.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-1.0, Outcome::new(Class::from(true), 1.0, -1.0)),
        ];
        let t = threshold(&outcomes, Class::from(false)); // if none false then return 2 * abs(last value)
        assert_eq!(t.value(), 2.0);

        let outcomes = vec![
            (1.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (3.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (10.0, Outcome::new(Class::from(true), 1.0, -1.0)),
        ];
        let t = threshold(&outcomes, Class::from(false)); // if none false then return double last value
        assert_eq!(t.value(), 20.0);

        let outcomes = vec![
            (-10.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-3.0, Outcome::new(Class::from(true), 1.0, -1.0)),
            (-1.0, Outcome::new(Class::from(true), 1.0, -1.0)),
        ];
        let t = threshold(&outcomes, Class::from(true)); // if none false then return first value
        assert_eq!(t.value(), -10.0);
    }
}
