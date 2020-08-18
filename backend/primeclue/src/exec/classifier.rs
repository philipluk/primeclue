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

use crate::data::data_set::DataView;
use crate::data::outcome::Class;
use crate::data::Size;
use crate::error::PrimeclueErr;
use crate::exec::scored_tree::ScoredTree;
use crate::serialization::{Deserializable, Serializable, Serializator};
use serde::export::Formatter;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;

#[derive(Copy, Clone, Debug, Serialize)]
pub struct AppliedScore {
    pub accuracy: f32,
    pub cost: f32,
}

impl Display for AppliedScore {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}% / {:.1} cost", self.accuracy, self.cost)
    }
}

#[derive(Debug, PartialEq)]
pub struct Classifier {
    classes: HashMap<Class, String>,
    trees: Vec<ScoredTree>,
}

impl Classifier {
    pub fn new(
        classes: HashMap<Class, String>,
        trees: Vec<ScoredTree>,
    ) -> Result<Self, PrimeclueErr> {
        if classes.is_empty() {
            PrimeclueErr::result("Class map is empty".to_string())
        } else if trees.is_empty() {
            PrimeclueErr::result("Tree vectors is empty".to_string())
        } else if classes.len() != trees.len() {
            PrimeclueErr::result(format!(
                "Class / Tree vectors lens don't match: {} vs {}",
                classes.len(),
                trees.len()
            ))
        } else {
            Ok(Classifier { classes, trees })
        }
    }

    pub fn get_classes(&self) -> &HashMap<Class, String> {
        &self.classes
    }

    pub fn data_size(&self) -> &Size {
        self.trees[0].data_size()
    }

    pub fn average_score(&self) -> Option<f32> {
        let mut val = 0.0;
        for tree in &self.trees {
            val += tree.score().value();
        }
        Some(val / self.trees.len() as f32)
    }

    pub fn node_count(&self) -> usize {
        self.trees.iter().map(|t| t.node_count()).sum()
    }

    pub fn sorted_trees(&self) -> Vec<&ScoredTree> {
        let mut scores = self.trees.iter().collect::<Vec<_>>();
        scores.sort_unstable_by(|&t1, &t2| t1.partial_cmp(&t2).unwrap_or(Ordering::Equal));
        scores
    }

    pub fn classify(&self, data: &DataView) -> Vec<&str> {
        let trees = self.sorted_trees();
        let mut responses = vec![""; data.cells().get(0, 0).len()];
        for tree in trees {
            let values = tree.execute(&data);
            let class_string = self.classes.get(&tree.score().class()).unwrap();
            for (value, response) in values.iter().zip(responses.iter_mut()) {
                if let Some(guess) = tree.guess(*value) {
                    if guess {
                        *response = class_string;
                    }
                }
            }
        }
        responses
    }

    pub fn execute_for_score(&self, data: &DataView) -> Option<f32> {
        let mut trees = self.sorted_trees();
        trees.remove(0);
        let mut test_score = 0.0;
        for tree in &trees {
            test_score += tree.execute_for_score(&data)?.value();
        }
        Some(test_score / trees.len() as f32)
    }

    pub fn applied_score(&self, data: &DataView) -> Option<AppliedScore> {
        let results = self.classify(data);
        let mut correct = 0;
        let mut total = 0;
        let mut reward = 0.0;
        let mut penalty = 0.0;
        for (actual, outcome) in results.iter().zip(data.outcomes()) {
            if actual.is_empty() {
                continue;
            }
            total += 1;
            let expected = data.class_map().get(&outcome.class())?;
            if actual == expected {
                correct += 1;
                reward += outcome.reward();
            } else {
                penalty += outcome.penalty();
            }
        }
        let accuracy = 100.0 * correct as f32 / total as f32;
        let cost = reward + penalty;
        Some(AppliedScore { accuracy, cost })
    }
}

impl Serializable for Classifier {
    fn serialize(&self, s: &mut Serializator) {
        s.add(&self.classes);
        s.add(&self.trees)
    }
}

impl Deserializable for Classifier {
    fn deserialize(s: &mut Serializator) -> Result<Self, String> {
        let classes = HashMap::deserialize(s)?;
        let trees = Vec::deserialize(s)?;
        Ok(Classifier { classes, trees })
    }
}

#[cfg(test)]
mod test {
    use crate::data::data_set::test::create_simple_data;
    use crate::data::outcome::Class;
    use crate::data::Size;
    use crate::exec::classifier::Classifier;
    use crate::exec::score::Objective::AUC;
    use crate::exec::score::{Score, Threshold};
    use crate::exec::scored_tree::ScoredTree;
    use crate::exec::training_group::TrainingGroup;
    use crate::exec::tree::Tree;
    use crate::serialization::serializator::test::test_serialization;
    use std::collections::HashMap;

    #[test]
    fn serialize_classifier() {
        let forbidden_cols = vec![];
        for _ in 0..10 {
            let (d1, d2, _) = create_simple_data().shuffle().into_views_split();
            let mut training_group =
                TrainingGroup::new(d1, d2, AUC, 5, &forbidden_cols).unwrap();
            loop {
                training_group.next_generation();
                if let Ok(classifier) = training_group.classifier() {
                    test_serialization(classifier);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_empty_classifier() {
        let classes = HashMap::new();
        let r = Classifier::new(classes, vec![]);
        assert!(r.is_err());

        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "true".to_string());
        classes.insert(Class::new(1), "false".to_string());
        let r = Classifier::new(classes, vec![]);
        assert!(r.is_err());

        let t = Tree::new(&Size::new(1, 1), 3, &vec![], 0.5, 0.5);
        let classes = HashMap::new();
        let r = Classifier::new(
            classes,
            vec![ScoredTree::new(t, Score::new(AUC, Class::new(0), 0.9, Threshold::new(0.0)))],
        );
        assert!(r.is_err());
    }

    #[test]
    fn invalid_class_count() {
        let mut trees = vec![];
        for i in 0..3 {
            let t = Tree::new(&Size::new(1, 1), 3, &vec![], 0.5, 0.5);
            trees.push(ScoredTree::new(
                t,
                Score::new(AUC, Class::new(i), 0.9, Threshold::new(0.0)),
            ))
        }
        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "true".to_string());
        classes.insert(Class::new(1), "false".to_string());
        let r = Classifier::new(classes, trees);
        assert!(r.is_err());

        let mut trees = vec![];
        for i in 0..2 {
            let t = Tree::new(&Size::new(1, 1), 3, &vec![], 0.5, 0.5);
            trees.push(ScoredTree::new(
                t,
                Score::new(AUC, Class::new(i), 0.9, Threshold::new(0.0)),
            ))
        }
        let mut classes = HashMap::new();
        classes.insert(Class::new(0), "true".to_string());
        classes.insert(Class::new(1), "false".to_string());
        classes.insert(Class::new(2), "null".to_string());
        let r = Classifier::new(classes, trees);
        assert!(r.is_err());
    }
}
