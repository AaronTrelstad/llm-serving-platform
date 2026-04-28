pub struct BTreeNode {
    keys: Vec<(u64, String)>,
    children: Vec<Box<BTreeNode>>,
    is_leaf: bool,
}

impl BTreeNode {
    pub fn new(is_leaf: bool) -> Self {
        Self {
            keys: Vec::new(),
            children: Vec::new(),
            is_leaf,
        }
    }
}

pub struct BTree {
    root: Box<BTreeNode>,
    order: usize,
}

impl BTree {
    pub fn new(order: usize) -> Self {
        Self {
            root: Box::new(BTreeNode::new(true)),
            order,
        }
    }

    pub fn insert(&mut self, timestamp: u64, job_id: String) {
        if self.root.keys.len() == 2 * self.order - 1 {
            let old_root = std::mem::replace(&mut self.root, Box::new(BTreeNode::new(false)));
            self.root.children.push(old_root);
            Self::split(&mut self.root, 0, self.order);
        }
        Self::insert_non_full(&mut self.root, timestamp, job_id, self.order);
    }

    pub fn range(&self, start: u64, end: u64) -> Vec<String> {
        let mut results = Vec::new();
        Self::range_search(&self.root, start, end, &mut results);
        results
    }

    fn split(parent: &mut BTreeNode, index: usize, order: usize) {
        let child = &mut parent.children[index];
        let mut new_node = BTreeNode::new(child.is_leaf);
        let mid_key = child.keys[order - 1].clone();
        new_node.keys = child.keys.split_off(order);
        child.keys.pop();

        if !child.is_leaf {
            new_node.children = child.children.split_off(order);
        }

        parent.keys.insert(index, mid_key);
        parent.children.insert(index + 1, Box::new(new_node));
    }

    fn insert_non_full(node: &mut BTreeNode, timestamp: u64, job_id: String, order: usize) {
        let mut index = node.keys.len() as isize - 1;

        if node.is_leaf {
            while index >= 0 && node.keys[index as usize].0 > timestamp {
                index -= 1;
            }
            node.keys.insert((index + 1) as usize, (timestamp, job_id));
        } else {
            while index >= 0 && node.keys[index as usize].0 > timestamp {
                index -= 1;
            }
            let mut child_index = (index + 1) as usize;

            if node.children[child_index].keys.len() == 2 * order - 1 {
                Self::split(node, child_index, order);
                if timestamp > node.keys[child_index].0 {
                    child_index += 1;
                }
            }

            Self::insert_non_full(&mut node.children[child_index], timestamp, job_id, order);
        }
    }

    fn range_search(node: &BTreeNode, start: u64, end: u64, results: &mut Vec<String>) {
        let mut index = 0;

        while index < node.keys.len() && node.keys[index].0 < start {
            if !node.is_leaf {
                Self::range_search(&node.children[index], start, end, results);
            }

            index += 1;
        }

        while index < node.keys.len() && node.keys[index].0 <= end {
            if !node.is_leaf {
                Self::range_search(&node.children[index], start, end, results);
            }

            results.push(node.keys[index].1.clone());
            index += 1;
        }

        if !node.is_leaf {
            Self::range_search(&node.children[index], start, end, results);
        }
    }
}
