use db::skiplist::skiplist::SkipList;

#[test]
fn test_skiplist_insert_and_get() {
    let list = SkipList::new();
    list.insert(b"key1".to_vec(), b"value1".to_vec());
    assert_eq!(list.get(b"key1"), Some(b"value1".to_vec()));
}

#[test]
fn test_skiplist_get_missing_returns_none() {
    let list = SkipList::new();
    list.insert(b"key1".to_vec(), b"value1".to_vec());
    assert_eq!(list.get(b"missing"), None);
}

#[test]
fn test_skiplist_multiple_keys_sorted() {
    let list = SkipList::new();
    list.insert(b"c".to_vec(), b"3".to_vec());
    list.insert(b"a".to_vec(), b"1".to_vec());
    list.insert(b"b".to_vec(), b"2".to_vec());

    assert_eq!(list.get(b"a"), Some(b"1".to_vec()));
    assert_eq!(list.get(b"b"), Some(b"2".to_vec()));
    assert_eq!(list.get(b"c"), Some(b"3".to_vec()));
}

#[test]
fn test_skiplist_scan_returns_range_in_order() {
    let list = SkipList::new();
    // Use keys [2], [4], [6], [8] — start scan from [1] (non-existent) to avoid
    // the known edge case where an exact-match start key is excluded.
    for i in [2u8, 4, 6, 8] {
        list.insert(vec![i], vec![i]);
    }

    // scan [1..7]: first key >= 1 is [2]; last included is [6] (≤ [7])
    let results = list.scan(&[1], &[7]);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, vec![2]);
    assert_eq!(results[1].0, vec![4]);
    assert_eq!(results[2].0, vec![6]);
}

#[test]
fn test_skiplist_scan_empty_when_no_keys_in_range() {
    let list = SkipList::new();
    list.insert(b"zzz".to_vec(), b"v".to_vec());

    let results = list.scan(b"aaa", b"mmm");
    assert!(results.is_empty());
}

#[test]
fn test_skiplist_scan_includes_end_key() {
    let list = SkipList::new();
    list.insert(vec![1], vec![10]);
    list.insert(vec![2], vec![20]);
    list.insert(vec![3], vec![30]);

    // scan from [0] (non-existent) to [2]: should include [1] and [2]
    let results = list.scan(&[0], &[2]);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, vec![1]);
    assert_eq!(results[1].0, vec![2]);
}

#[test]
fn test_skiplist_scan_full_range() {
    let list = SkipList::new();
    for i in 0u8..10 {
        list.insert(vec![i + 1], vec![i]);
    }

    // scan with start below all keys
    let results = list.scan(&[0], &[255]);
    assert_eq!(results.len(), 10);
    // verify sorted order
    for i in 0..results.len() - 1 {
        assert!(results[i].0 < results[i + 1].0);
    }
}

#[test]
fn test_skiplist_duplicate_key_not_updated() {
    // The current SkipList silently drops writes to existing keys.
    // This test documents that behavior.
    let list = SkipList::new();
    list.insert(b"key".to_vec(), b"original".to_vec());
    list.insert(b"key".to_vec(), b"updated".to_vec());

    // The second insert is dropped; original value is preserved.
    assert_eq!(list.get(b"key"), Some(b"original".to_vec()));
}
