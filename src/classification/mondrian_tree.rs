use crate::classification::alias::FType;
use crate::classification::mondrian_node::{Node, Stats};

use ndarray::Array1;

use num::{Float, FromPrimitive};
use rand::prelude::*;
use rand_distr::{Distribution, Exp};

use std::collections::HashSet;

use std::fmt;

use std::usize;

#[derive(Clone)]
pub struct MondrianTreeClassifier<F: FType> {
    n_features: usize,
    n_labels: usize,
    rng: ThreadRng,
    nodes: Vec<Node<F>>,
    root: Option<usize>,
}

impl<F: FType + fmt::Display> fmt::Display for MondrianTreeClassifier<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "")?;
        if let Some(root) = self.root {
            self.recursive_repr(Some(root), f, "", true)
        } else {
            Ok(())
        }
    }
}

impl<F: FType + fmt::Display> MondrianTreeClassifier<F> {
    /// Helper method to recursively format node details.
    fn recursive_repr(
        &self,
        node_idx: Option<usize>,
        f: &mut fmt::Formatter<'_>,
        prefix: &str,
        is_last: bool,
    ) -> fmt::Result {
        if let Some(idx) = node_idx {
            let node = &self.nodes[idx];
            let node_prefix = if is_last { "└─" } else { "├─" };
            let child_prefix = if is_last { "  " } else { "│ " };
            let feature = if self.nodes[idx].feature > 100 {
                String::from("_")
            } else {
                self.nodes[idx].feature.to_string()
            };
            writeln!(
                f,
                "{}{}Node {}: time={:.3}, min={:?}, max={:?}, thrs={:.2}, f={}, counts={}, is_leaf={}",
                // "{}{}Node {}: left={:?}, right={:?}, parent={:?}, time={:.3}, min={:?}, max={:?}, thrs={:.2}, f={}, counts={}",
                prefix,
                node_prefix,
                idx,
                // node.left,
                // node.right,
                // node.parent,
                node.time,
                node.range_min.to_vec(),
                node.range_max.to_vec(),
                node.threshold,
                feature,
                node.stats.counts,
                node.is_leaf,
            )?;

            let mut children = vec![];
            if let Some(left) = node.left {
                children.push((left, node.right.is_none())); // Left child
            }
            if let Some(right) = node.right {
                children.push((right, true)); // Right child
            }

            for (i, (child, last)) in children.into_iter().enumerate() {
                self.recursive_repr(Some(child), f, &(prefix.to_owned() + child_prefix), last)?;
            }
        }
        Ok(())
    }
}

impl<F: FType> MondrianTreeClassifier<F> {
    pub fn new(n_features: usize, n_labels: usize) -> Self {
        MondrianTreeClassifier::<F> {
            n_features,
            n_labels,
            rng: rand::thread_rng(),
            nodes: vec![],
            root: None,
        }
    }

    fn create_node(
        &mut self,
        x: &Array1<F>,
        y: usize,
        parent: Option<usize>,
        time: F,
        is_leaf: bool,
    ) -> usize {
        let mut node = Node::<F> {
            parent,
            time, // F::from(1e9).unwrap(), // Very large value
            is_leaf,
            range_min: x.clone(),
            range_max: x.clone(),
            feature: usize::MAX,
            threshold: F::infinity(),
            left: None,
            right: None,
            stats: Stats::new(self.n_labels, self.n_features),
        };
        node.update_leaf(x, y);
        self.nodes.push(node);
        let node_idx = self.nodes.len() - 1;
        node_idx
    }

    fn create_node_empty(&mut self, parent: Option<usize>, time: F) -> usize {
        let node = Node::<F> {
            parent,
            time, // F::from(1e9).unwrap(), // Very large value
            is_leaf: true,
            range_min: Array1::from_elem(self.n_features, F::infinity()),
            range_max: Array1::from_elem(self.n_features, -F::infinity()),
            feature: usize::MAX,
            threshold: F::infinity(),
            left: None,
            right: None,
            stats: Stats::new(self.n_labels, self.n_features),
        };
        self.nodes.push(node);
        let node_idx = self.nodes.len() - 1;
        node_idx
    }

    fn test_tree(&self) {
        // TODO: move to test
        for node_idx in 0..self.nodes.len() {
            // TODO: check if self.root is None, if so tree should be empty
            if node_idx == self.root.unwrap() {
                // Root node
                debug_assert!(self.nodes[node_idx].parent.is_none(), "Root has a parent.");
            } else {
                // Non-root node
                debug_assert!(
                    !self.nodes[node_idx].parent.is_none(),
                    "Non-root node has no parent"
                )
            }
        }

        let children_l: Vec<usize> = self.nodes.iter().filter_map(|node| node.left).collect();
        let children_r: Vec<usize> = self.nodes.iter().filter_map(|node| node.right).collect();
        let children = [children_l.clone(), children_r.clone()].concat();
        let mut seen = HashSet::new();
        let has_duplicates = children.iter().any(|item| !seen.insert(item));
        debug_assert!(
            !has_duplicates,
            "Multiple nodes share one child. Children left: {:?}, Children right: {:?}",
            children_l, children_r
        );
    }

    fn compute_split_time(
        &self,
        time: F,
        exp_sample: F,
        node_idx: usize,
        y: usize,
        extensions_sum: F,
    ) -> F {
        // println!("is_dirac() - counts: {}, node: {}", self.nodes[node_idx].stats.counts, node_idx);
        if self.nodes[node_idx].is_dirac(y) {
            // println!(
            //     "compute_split_time() - node: {node_idx} - extensions_sum: {:?} - same class: {}",
            //     extensions_sum,
            //     self.nodes[node_idx].is_dirac(y)
            // );
            return F::zero();
        }

        if extensions_sum > F::zero() {
            let split_time = time + exp_sample;

            // From River: If the node is a leaf we must split it
            if self.nodes[node_idx].is_leaf {
                // println!(
                //     "compute_split_time() - node: {node_idx} - extensions_sum: {:?} - split is_leaf",
                //     extensions_sum
                // );
                return split_time;
            }

            // From River: Otherwise we apply Mondrian process dark magic :)
            // 1. We get the creation time of the childs (left and right is the same)
            let child_idx = self.nodes[node_idx].left.unwrap();
            let child_time = self.nodes[child_idx].time;
            // 2. We check if splitting time occurs before child creation time
            if split_time < child_time {
                // println!(
                //     "compute_split_time() - node: {node_idx} - extensions_sum: {:?} - split mid tree",
                //     extensions_sum
                // );
                return split_time;
            }
            // println!("compute_split_time() - node: {node_idx} - extensions_sum: {:?} - not increased enough to split (mid node)", extensions_sum);
        } else {
            // println!(
            //     "compute_split_time() - node: {node_idx} - extensions_sum: {:?} - not outside box",
            //     extensions_sum
            // );
        }

        F::zero()
    }

    fn go_downwards(&mut self, node_idx: usize, x: &Array1<F>, y: usize) -> usize {
        let time = self.nodes[node_idx].time;
        let node_range_min = &self.nodes[node_idx].range_min;
        let node_range_max = &self.nodes[node_idx].range_max;
        let extensions = {
            let e_min = (node_range_min - x).mapv(|v| F::max(v, F::zero()));
            let e_max = (x - node_range_max).mapv(|v| F::max(v, F::zero()));
            &e_min + &e_max
        };
        // 'T' in River
        let exp_sample = {
            let lambda = extensions.sum();
            let exp_dist = Exp::new(lambda.to_f32().unwrap()).unwrap();
            let exp_sample = F::from_f32(exp_dist.sample(&mut self.rng)).unwrap();
            // DEBUG: shadowing with Exp expected value
            let exp_sample = F::one() / lambda;
            exp_sample
        };
        let split_time = self.compute_split_time(time, exp_sample, node_idx, y, extensions.sum());
        if split_time > F::zero() {
            // Split the current node: if leaf we two leafs, otherwise
            // we add a new node along the path
            let feature = {
                let cumsum = extensions
                    .iter()
                    .scan(F::zero(), |acc, &x| {
                        *acc = *acc + x;
                        Some(*acc)
                    })
                    .collect::<Array1<F>>();
                let e_sample = F::from_f32(self.rng.gen::<f32>()).unwrap() * extensions.sum();
                // DEBUG: shadowing with expected value
                let e_sample = F::from_f32(0.5).unwrap() * extensions.sum();
                cumsum.iter().position(|&val| val > e_sample).unwrap()
            };

            let mut is_right_extension = x[feature] > node_range_min[feature];
            // DEBUG: just to test the is not leaf case
            is_right_extension = false;

            let (lower_bound, upper_bound) = if is_right_extension {
                (
                    node_range_max[feature].to_f32().unwrap(),
                    x[feature].to_f32().unwrap(),
                )
            } else {
                (
                    x[feature].to_f32().unwrap(),
                    node_range_min[feature].to_f32().unwrap(),
                )
            };
            // let threshold: F = F::from_f32(self.rng.gen_range(lower_bound..upper_bound)).unwrap();
            // DEBUG: split in the middle
            let threshold = F::from_f32((lower_bound + upper_bound) / 2.0).unwrap();
            // println!(
            //     "Threshold: {threshold} - Lower: {}, Upper: {}",
            //     lower_bound, upper_bound
            // );

            let mut range_min = node_range_min.clone();
            let mut range_max = node_range_max.clone();
            range_min.zip_mut_with(x, |a, &b| *a = F::min(*a, b));
            range_max.zip_mut_with(x, |a, &b| *a = F::max(*a, b));

            if self.nodes[node_idx].is_leaf {
                // println!("go_downwards() - split_time > 0 (is leaf)");
                let leaf_full = self.create_node(x, y, Some(node_idx), split_time, true);
                let leaf_empty = self.create_node_empty(Some(node_idx), split_time);
                // if x[feature] <= threshold {
                if is_right_extension {
                    self.nodes[node_idx].left = Some(leaf_empty);
                    self.nodes[node_idx].right = Some(leaf_full);
                } else {
                    self.nodes[node_idx].left = Some(leaf_full);
                    self.nodes[node_idx].right = Some(leaf_empty);
                }
                self.nodes[node_idx].range_min = range_min;
                self.nodes[node_idx].range_max = range_max;
                self.nodes[node_idx].threshold = threshold;
                self.nodes[node_idx].feature = feature;
                self.nodes[node_idx].is_leaf = false;
                self.update_downwards(node_idx);
                return node_idx;
            } else {
                // println!("go_downwards() - split_time > 0 (not leaf)");
                let parent_node = Node {
                    parent: self.nodes[node_idx].parent,
                    time: self.nodes[node_idx].time,
                    is_leaf: false,
                    range_min,
                    range_max,
                    feature,
                    threshold,
                    left: None,
                    right: None,
                    stats: Stats::new(self.n_labels, self.n_features),
                };
                self.nodes.push(parent_node);
                let parent_idx = self.nodes.len() - 1;

                // === Changed "create_node_empty" to "create_node"
                // TODO: check is_leaf, sometimes is true, sometimes false??
                let sibling_idx = self.create_node(x, y, Some(parent_idx), split_time, true);

                // println!(
                //     "grandpa: {:?}, parent: {:?}, child: {:?}, sibling: {:?}",
                //     self.nodes[node_idx].parent, parent_idx, node_idx, sibling_idx
                // );
                // Node 1. Grandpa: self.nodes[node_idx].parent
                // └─Node 3. (new) Parent: parent_idx
                //   ├─Node 2. Child: node_idx
                //   └─Node 4. (new) Sibling: sibling_idx
                if is_right_extension {
                    self.nodes[parent_idx].left = Some(node_idx);
                    self.nodes[parent_idx].right = Some(sibling_idx);
                } else {
                    self.nodes[parent_idx].left = Some(sibling_idx);
                    self.nodes[parent_idx].right = Some(node_idx);
                }
                // 'stats' copied from River
                self.nodes[parent_idx].stats = self.nodes[node_idx].stats.clone();
                self.nodes[node_idx].parent = Some(parent_idx);

                // self.nodes[node_idx].time = split_time;

                // I'm 0% sure if this "if" is required.
                if self.nodes[node_idx].is_leaf {
                    // {
                    // From River: reset child
                    self.nodes[node_idx].range_min =
                        Array1::from_elem(self.n_features, F::infinity());
                    self.nodes[node_idx].range_max =
                        Array1::from_elem(self.n_features, -F::infinity());
                    self.nodes[node_idx].stats = Stats::new(self.n_labels, self.n_features);
                }
                // self.update_downwards(parent_idx);
                // From River: added "update_leaf" after "update_downwards"
                self.nodes[parent_idx].update_leaf(x, y);
                return parent_idx;
            }
        } else {
            // No split, just update the node. If leaf add to count, else call recursively next child node.

            let node = &mut self.nodes[node_idx];
            node.range_min.zip_mut_with(x, |a, b| *a = F::min(*a, *b));
            node.range_max.zip_mut_with(x, |a, b| *a = F::max(*a, *b));

            if node.is_leaf {
                // println!("go_downwards() - split_time < 0 (is leaf)");
                node.update_leaf(x, y);
            } else {
                // println!("go_downwards() - split_time < 0 (not leaf)");
                if x[node.feature] <= node.threshold {
                    let node_left = node.left.unwrap();
                    let node_left_new = Some(self.go_downwards(node_left, x, y));
                    let node = &mut self.nodes[node_idx];
                    node.left = node_left_new;
                } else {
                    let node_right = node.right.unwrap();
                    let node_right_new = Some(self.go_downwards(node_right, x, y));
                    let node = &mut self.nodes[node_idx];
                    node.right = node_right_new;
                };
                // "update_downwards" was not in Nel215 implementation, added because of Python implementation
                // Later changed from "update_downwards" to "update_leaf"
                // self.update_downwards(node_idx);
                self.nodes[node_idx].update_leaf(x, y);
            }
            return node_idx;
        }
    }

    /// Update 'node stats' by merging 'right child stats + left child stats'.
    fn update_downwards(&mut self, node_idx: usize) {
        // From River:
        // Ranges (range_min, range_max) are already up to date

        // println!("update_downwards() - node_idx: {}", node_idx);

        let node = &self.nodes[node_idx];
        let left_s = &self.nodes[node.left.unwrap()].stats;
        let right_s = &self.nodes[node.right.unwrap()].stats;
        let merge_s = node.update_internal(left_s, right_s);

        let node = &mut self.nodes[node_idx];
        node.stats = merge_s;
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one.
    ///
    /// Function in River/LightRiver: "learn_one()"
    pub fn partial_fit(&mut self, x: &Array1<F>, y: usize) {
        self.root = match self.root {
            None => Some(self.create_node(x, y, None, F::zero(), true)),
            Some(root_idx) => Some(self.go_downwards(root_idx, x, y)),
        };
        // println!("partial_fit() tree post {}===========", self);
    }

    fn fit(&self) {
        unimplemented!("Make the program first work with 'partial_fit', then implement this")
    }

    /// Note: In Nel215 codebase should work on multiple records. Here it only works
    /// as public interface for predict().
    pub fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        // println!("predict_proba() - tree size: {}", self.nodes.len());
        // self.test_tree();
        self.predict(x, self.root.unwrap(), F::one())
    }

    fn predict(&self, x: &Array1<F>, node_idx: usize, p_not_separated_yet: F) -> Array1<F> {
        let node = &self.nodes[node_idx];
        // Probability 'p' of the box not splitting.
        //     eta (box dist): larger distance, more prob of splitting
        //     d (time delta with parent): more dist with parent, more prob of splitting
        let p = {
            let d = node.time - self.get_parent_time(node_idx);
            let dist_max = (x - &node.range_max).mapv(|v| F::max(v, F::zero()));
            let dist_min = (&node.range_min - x).mapv(|v| F::max(v, F::zero()));
            let eta = dist_min.sum() + dist_max.sum();
            F::one() - (-d * eta).exp()
        };
        debug_assert!(!p.is_nan(), "Found probability of splitting NaN. This is probably because range_max and range_min are [inf, inf]");

        // Generate a result for the current node using its statistics.
        let res = node.stats.create_result(x, p_not_separated_yet * p);

        let w = p_not_separated_yet * (F::one() - p);
        if node.is_leaf {
            let res2 = node.stats.create_result(x, w);
            return res + res2;
        } else {
            let child_idx = if x[node.feature] <= node.threshold {
                node.left
            } else {
                node.right
            };
            let child_res = self.predict(x, child_idx.unwrap(), w);
            return res + child_res;
        }
    }

    fn get_parent_time(&self, node_idx: usize) -> F {
        // If node is root, time is 0
        match self.nodes[node_idx].parent {
            Some(parent_idx) => self.nodes[parent_idx].time,
            None => F::from_f32(0.0).unwrap(),
        }
    }

    pub fn get_tree_size(&self) -> usize {
        self.nodes.len()
    }
}
