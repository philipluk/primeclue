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
use crate::exec::score::{Objective, Score};
use crate::exec::scored_tree::ScoredTree;
use crate::exec::tree::Tree;
use crate::rand::GET_RNG;
use rand::seq::IteratorRandom;
use rand::Rng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use serde::export::fmt::Error;
use serde::export::Formatter;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem::replace;
use std::ops::Add;
use std::time::Duration;
use std::time::SystemTime;

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct GroupId(u64);

pub struct ClassTraining {
    next_id: GroupId,
    wasted_generations: usize,
    objective: Objective,
    size: usize,
    node_limit: usize,
    data_size: Size,
    forbidden_cols: Vec<usize>,
    best_tree: Option<ScoredTree>,
    class: Class,
    groups: HashMap<GroupId, ClassGroup>,
}

impl Debug for ClassTraining {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{:?}", self.class)
    }
}

impl ClassTraining {
    #[must_use]
    pub fn new(
        size: usize,
        data_size: &Size,
        forbidden_cols: Vec<usize>,
        objective: Objective,
        class: Class,
    ) -> Self {
        let groups = HashMap::new();
        ClassTraining {
            next_id: GroupId(1),
            wasted_generations: 0,
            size,
            data_size: *data_size,
            forbidden_cols,
            groups,
            node_limit: 5_000_000,
            best_tree: None,
            objective,
            class,
        }
    }

    pub fn class(&self) -> &Class {
        &self.class
    }

    pub fn training_score(&self) -> Option<f32> {
        self.best_tree.as_ref().map(|t| t.score().value())
    }

    #[must_use]
    pub fn best_tree(&self) -> Option<&ScoredTree> {
        self.best_tree.as_ref()
    }

    pub fn next_generation(&mut self, training_data: &DataView, verification_data: &DataView) {
        self.fill_up();
        let objective = self.objective;
        let class = self.class;
        let length = self.size;
        let forbidden_cols = &self.forbidden_cols;
        self.groups.par_iter_mut().for_each(|(_, group)| {
            let end_time = SystemTime::now().add(Duration::from_secs(1));
            while SystemTime::now().lt(&end_time) {
                group.breed(forbidden_cols, length);
                group.execute_and_score(objective, training_data, class);
                if group.scored.is_empty() {
                    break;
                }
                group.remove_weak_trees(length);
            }
        });
        self.remove_empty_groups();
        self.select_best(verification_data);
        self.keep_node_limit();
        self.groups.shrink_to_fit();
    }

    fn remove_empty_groups(&mut self) {
        self.groups.retain(|_, p| !p.scored.is_empty());
    }

    fn keep_node_limit(&mut self) {
        let mut sizes =
            self.groups.values().map(|p| (p.id, p.nodes_count())).collect::<Vec<_>>();
        let sum = sizes.iter().map(|(_, s)| s).sum::<usize>();
        if sum > self.node_limit {
            sizes.sort_by(|(_, s1), (_, s2)| s1.cmp(s2));
            let mut so_far = 0;
            for (id, size) in sizes {
                if so_far + size > self.node_limit {
                    self.groups.remove(&id);
                } else {
                    so_far += size;
                }
            }
        }
    }

    fn fill_up(&mut self) {
        while self.groups.len() < self.size * 2 {
            let id = self.next_id;
            self.next_id.0 += 1;
            let group = generate_group(self, id, &self.forbidden_cols);
            self.groups.insert(group.id, group);
        }
    }

    fn select_best(&mut self, data: &DataView) {
        let mut sorted_scores = self.sorted_by_score(data);
        sorted_scores.reverse();
        self.assign_best_tree(&sorted_scores);
        self.remove_bad_groups(&mut sorted_scores);
    }

    fn remove_bad_groups(&mut self, sorted_scores: &mut Vec<(GroupId, Score)>) {
        if self.groups.len() <= self.size {
            return;
        }
        let mut new_group_map = HashMap::with_capacity(self.size);
        for _ in 0..self.size {
            if !sorted_scores.is_empty() {
                let (first, _) = sorted_scores.remove(0);
                new_group_map.insert(first, self.groups.remove(&first).unwrap());
            }
        }
        self.groups = new_group_map;
    }

    fn assign_best_tree(&mut self, sorted_scores: &[(GroupId, Score)]) {
        if !sorted_scores.is_empty() {
            let mut best_now =
                ScoredTree::best_tree(&self.groups.get(&sorted_scores[0].0).unwrap().scored)
                    .unwrap()
                    .clone();
            let score_value = (sorted_scores[0].1.value() + best_now.score().value()) / 2.0;
            let score = Score::new(
                best_now.score().objective(),
                best_now.score().class(),
                score_value,
                best_now.score().threshold(),
            );
            best_now.set_score(score);
            if self.best_tree.is_none() || (&best_now > self.best_tree.as_ref().unwrap()) {
                self.best_tree = Some(best_now);
                self.wasted_generations = 0;
            } else {
                self.wasted_generations += 1;
            }
        }
    }

    fn sorted_by_score(&mut self, data: &DataView) -> Vec<(GroupId, Score)> {
        let mut scores = Vec::with_capacity(self.groups.len());
        for g in self.groups.values() {
            if let Some(tree) = ScoredTree::best_tree(&g.scored) {
                if let Some(score) = tree.execute_for_score(data) {
                    scores.push((g.id, score))
                }
            }
        }
        scores.sort_unstable_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap());
        scores
    }
}

pub struct ClassGroup {
    id: GroupId,
    fresh: Vec<Tree>,
    scored: Vec<ScoredTree>,
}

impl Debug for ClassGroup {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", self.id.0)
    }
}

impl ClassGroup {
    fn create_random(
        group_size: usize,
        data_size: &Size,
        id: GroupId,
        forbidden_cols: &[usize],
    ) -> Self {
        let mut rng = GET_RNG();
        let max_branch_length = rng.gen_range(2, 30);
        let data_prob = rng.gen_range(0.01, 0.99);
        let branch_prob = rng.gen_range(0.01, 0.99);
        let tree =
            Tree::new(data_size, max_branch_length, forbidden_cols, branch_prob, data_prob);
        ClassGroup::create_from_tree(group_size, id, tree, forbidden_cols)
    }

    fn create_from_tree(
        group_size: usize,
        id: GroupId,
        tree: Tree,
        forbidden_cols: &[usize],
    ) -> ClassGroup {
        let mut trees = Vec::with_capacity(group_size);
        trees.push(tree);
        while trees.len() < group_size {
            let mut t = trees[0].clone();
            t.change_weights();
            t.mutate(forbidden_cols);
            trees.push(t);
        }
        ClassGroup { id, fresh: trees, scored: Vec::new() }
    }

    fn total_score(&self) -> f32 {
        self.scored.iter().map(|t| t.score().value()).sum()
    }

    fn breed(&mut self, forbidden_cols: &[usize], count: usize) {
        if let Some(child) = ScoredTree::best_tree(&self.scored).map(|t| t.tree()) {
            while self.fresh.len() < count {
                let mut child = child.clone();
                child.mutate(forbidden_cols);
                child.change_weights();
                self.fresh.push(child);
            }
        }
    }

    fn random_parent(&self, total_score: f32) -> Option<&Tree> {
        let mut rng = GET_RNG();
        let rand_score = total_score * rng.gen_range(0.0, 1.0);
        let mut parent_score = 0.0;
        for id in 0..self.scored.len() {
            parent_score += self.scored[id].score().value();
            if parent_score >= rand_score {
                return Some(self.scored[id].tree());
            }
        }
        None
    }

    fn remove_weak_trees(&mut self, length: usize) {
        self.scored.sort_unstable_by(|t1, t2| t1.partial_cmp(&t2).unwrap_or(Equal));
        self.scored.reverse();
        self.scored.truncate(length);
    }

    fn execute_and_score(&mut self, objective: Objective, data: &DataView, class: Class) {
        let len = self.fresh.len();
        let trees = replace(&mut self.fresh, Vec::with_capacity(len));
        for tree in trees {
            if let Some(score) = tree.execute_for_score(data, class, objective) {
                self.scored.push(ScoredTree::new(tree, score))
            }
        }
    }

    #[must_use]
    fn nodes_count(&self) -> usize {
        self.scored.iter().map(|t| t.node_count()).sum::<usize>()
            + self.fresh.iter().map(|t| t.node_count()).sum::<usize>()
    }
}

fn generate_group(
    training: &ClassTraining,
    id: GroupId,
    forbidden_cols: &[usize],
) -> ClassGroup {
    let mut rng = GET_RNG();
    let random_chance =
        (1.0 - (1.0 / (training.wasted_generations + 1) as f64)).min(0.1).max(0.9);
    loop {
        if rng.gen_bool(random_chance) {
            return ClassGroup::create_random(
                training.size,
                &training.data_size,
                id,
                forbidden_cols,
            );
        } else if let Some(class_group) = create_mutated(training, id, forbidden_cols) {
            return class_group;
        }
    }
}

fn create_mutated(
    training: &ClassTraining,
    id: GroupId,
    forbidden_cols: &[usize],
) -> Option<ClassGroup> {
    let mut rng = GET_RNG();
    let group = training.groups.values().choose(&mut rng)?;
    let total_score = group.total_score();
    let mut tree = group.random_parent(total_score)?.clone();
    tree.mutate(forbidden_cols);
    Some(ClassGroup::create_from_tree(training.size, id, tree, forbidden_cols))
}
