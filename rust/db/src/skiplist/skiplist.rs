use crossbeam_epoch::{Atomic, Guard, Owned, Shared};
use std::sync::atomic::Ordering;

const MAX_LEVEL: usize = 16;
const PROBABILITY: f64 = 0.5;

pub struct Node {
    key:   Vec<u8>,
    value: Vec<u8>,
    next:  Vec<Atomic<Node>>,
}

impl Node {
    pub fn new(key: Vec<u8>, value: Vec<u8>, level: usize) -> Self {
        Self {
            key,
            value,
            next: (0..level).map(|_| Atomic::null()).collect(),
        }
    }

    fn head() -> Self {
        Self::new(vec![], vec![], MAX_LEVEL)  
    }
}

pub struct SkipList {
    head: Atomic<Node>,
    size: std::sync::atomic::AtomicUsize,
}

impl SkipList {
    pub fn new() -> Self {
        Self {
            head: Atomic::new(Node::head()),
            size: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        todo!();
    }

    pub fn insert(&self, key: Vec<u8>, value: Vec<u8>) {
        todo!();
    }

    pub fn scan(&self, start: &[u8], end: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        todo!();
    }
}
