use db::btree::btree::BTree;

#[test]
fn test_btree_insert_and_range_all() {
    let mut tree = BTree::new(4);
    tree.insert(100, "a".to_string());
    tree.insert(200, "b".to_string());
    tree.insert(300, "c".to_string());

    let results = tree.range(0, 301);
    assert_eq!(results.len(), 3);
}

#[test]
fn test_btree_range_is_inclusive_on_end() {
    let mut tree = BTree::new(4);
    tree.insert(100, "a".to_string());
    tree.insert(200, "b".to_string());
    tree.insert(300, "c".to_string());

    // end=300 is inclusive, so [300] is included
    let results = tree.range(0, 300);
    assert_eq!(results.len(), 3);
    assert!(results.contains(&"a".to_string()));
    assert!(results.contains(&"b".to_string()));
    assert!(results.contains(&"c".to_string()));
}

#[test]
fn test_btree_range_empty_when_no_keys_match() {
    let mut tree = BTree::new(4);
    tree.insert(1000, "a".to_string());
    tree.insert(2000, "b".to_string());

    assert!(tree.range(3000, 5000).is_empty());
    assert!(tree.range(0, 500).is_empty());
}

#[test]
fn test_btree_handles_many_inserts_with_splits() {
    let mut tree = BTree::new(4);
    for i in 1u64..=50 {
        tree.insert(i * 100, format!("val_{}", i));
    }

    let results = tree.range(0, 5001);
    assert_eq!(results.len(), 50);
}

#[test]
fn test_btree_duplicate_timestamps_both_stored() {
    let mut tree = BTree::new(4);
    tree.insert(1000, "job-a".to_string());
    tree.insert(1000, "job-b".to_string());

    let results = tree.range(0, 2000);
    assert_eq!(results.len(), 2);
    assert!(results.contains(&"job-a".to_string()));
    assert!(results.contains(&"job-b".to_string()));
}

#[test]
fn test_btree_range_subset() {
    let mut tree = BTree::new(4);
    for i in 1u64..=10 {
        tree.insert(i * 1000, format!("item_{}", i));
    }

    // Range [3000, 7000): should include 3000, 4000, 5000, 6000, 7000
    let results = tree.range(3000, 7000);
    assert_eq!(results.len(), 5);
    assert!(results.contains(&"item_3".to_string()));
    assert!(results.contains(&"item_4".to_string()));
    assert!(results.contains(&"item_5".to_string()));
    assert!(results.contains(&"item_6".to_string()));
    assert!(results.contains(&"item_7".to_string()));
}
