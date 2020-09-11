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
use crate::data::outcome::{sort_guesses, Class};
use crate::data::InputShape;
use crate::exec::node::Weighted;
use crate::exec::score::{calc_score, Objective, Score};
use crate::rand::GET_RNG;
use crate::serialization::{Deserializable, Serializable, Serializator};
use rand::Rng;

#[derive(Debug, PartialEq, Clone)]
pub struct Tree {
    node: Weighted,
    input_shape: InputShape,
    node_count: usize,
}

impl Serializable for Tree {
    fn serialize(&self, s: &mut Serializator) {
        s.add_items(&[&self.node, &self.input_shape, &self.node_count]);
    }
}

impl Deserializable for Tree {
    fn deserialize(s: &mut Serializator) -> Result<Tree, String> {
        let node = Weighted::deserialize(s)?;
        let input_shape = InputShape::deserialize(s)?;
        let node_count = usize::deserialize(s)?;
        Ok(Tree { node, input_shape, node_count })
    }
}

impl Tree {
    pub fn new(
        input_shape: &InputShape,
        max_branch_length: usize,
        forbidden_cols: &[usize],
        branch_prob: f64,
        data_prob: f64,
    ) -> Tree {
        let node = Weighted::new(
            0,
            input_shape,
            branch_prob,
            max_branch_length,
            forbidden_cols,
            data_prob,
        );
        let node_count = node.node_count();
        Tree { node, input_shape: *input_shape, node_count }
    }

    pub fn change_weights(&mut self) {
        let mut rng = GET_RNG();
        let count = rng.gen_range(0, (self.node_count() as f32).sqrt() as i32);
        for _ in 0..count {
            let node = self.select_random_node();
            node.change_weight(rng.gen_range(-3.0, 3.0));
        }
    }

    pub fn mutate(&mut self, forbidden_cols: &[usize]) {
        let input_shape = self.input_shape;
        let node = self.select_random_node();
        node.mutate(&input_shape, forbidden_cols);
    }

    pub fn select_node(&mut self, node_id: usize) -> &mut Weighted {
        self.node.select_node(node_id, self.node_count)
    }

    pub fn select_random_node(&mut self) -> &mut Weighted {
        let node_id = GET_RNG().gen_range(0, self.node_count());
        self.select_node(node_id)
    }

    #[must_use]
    pub fn serializator(&self) -> Serializator {
        let mut s = Serializator::new();
        s.add(self);
        s
    }

    #[must_use]
    pub fn input_shape(&self) -> &InputShape {
        &self.input_shape
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    #[must_use]
    pub fn execute_for_score(
        &self,
        data: &DataView,
        class: Class,
        objective: Objective,
    ) -> Option<Score> {
        if data.cells().get(0, 0).len() < 2 {
            None
        } else {
            let guesses = self.execute(data);
            let mut change = false;
            for v in &guesses {
                if !v.is_finite() {
                    return None;
                }
                change = change || (*v - guesses[0]).abs() > 0.001;
            }
            if !change {
                return None;
            }
            // TODO clean up: move threshold calculation and sort to objective.calc_score()
            let outcomes = sort_guesses(guesses, data.outcomes());
            let threshold = objective.threshold(&outcomes, class);
            Some(calc_score(&outcomes, threshold, class, objective))
        }
    }

    pub(crate) fn execute(&self, data: &DataView) -> Vec<f32> {
        self.node.execute(data.cells())
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::data::InputShape;
    use crate::exec::functions::{MATH_CONSTANTS, ONE_ARG_FUNCTIONS, TWO_ARG_FUNCTIONS};
    use crate::exec::node::{Node, Weighted};
    use crate::exec::tree::Tree;
    use crate::rand::GET_RNG;
    use crate::serialization::serializator::test::test_serialization;
    use rand::Rng;

    #[test]
    fn serialize_tree() {
        let input_shape = InputShape::new(10, 20);
        for _ in 0..10_000 {
            let max_branch_length = GET_RNG().gen_range(2, 15);
            let tree = Tree::new(&input_shape, max_branch_length, &Vec::new(), 0.5, 0.5);
            test_serialization(tree);
        }
    }

    fn sample_tree() -> Tree {
        let n1 = Node::DataValue(0, 0);
        let w1 = Weighted::from(n1);
        let n2 = Node::DataValue(0, 0);
        let w2 = Weighted::from(n2);
        let n3 = Node::MathConstant(&MATH_CONSTANTS[0]);
        let w3 = Weighted::from(n3);
        let n4 = Node::SingleArgFunction(&ONE_ARG_FUNCTIONS[0], w1);
        let w4 = Weighted::from(n4);
        let n5 = Node::DoubleArgFunction(&TWO_ARG_FUNCTIONS[2], w2, w3);
        let w5 = Weighted::from(n5);
        let n6 = Node::DoubleArgFunction(&TWO_ARG_FUNCTIONS[3], w4, w5);
        let node_count = n6.node_count();
        Tree { node: Weighted::from(n6), input_shape: InputShape::new(1, 1), node_count }
    }

    #[test]
    fn count_nodes() {
        let tree = sample_tree();
        assert_eq!(tree.node_count(), 6);
    }

    pub(crate) fn create_short_tree() -> Tree {
        let n1 = Node::DataValue(0, 0);
        let w1 = Weighted::from(n1);
        let node_count = w1.node_count();
        Tree { node: w1, input_shape: InputShape::new(1, 1), node_count }
    }

    pub(crate) fn create_long_tree() -> Tree {
        let n1 = Node::DataValue(0, 0);
        let w1 = Weighted::from(n1);
        let w = Weighted::from(Node::SingleArgFunction(&ONE_ARG_FUNCTIONS[0], w1));
        let node_count = w.node_count();
        Tree { node: w, input_shape: InputShape::new(1, 1), node_count }
    }
}
