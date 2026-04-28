use db::lsm::memtable::MemTable;

#[test]
fn test_memtable_insert_and_get() {
    let table = MemTable::new();
    table.insert(b"key".to_vec(), b"value".to_vec()).unwrap();
    assert_eq!(table.get(b"key"), Some(b"value".to_vec()));
}

#[test]
fn test_memtable_get_missing_returns_none() {
    let table = MemTable::new();
    assert_eq!(table.get(b"missing"), None);
}

#[test]
fn test_memtable_size_tracking() {
    let table = MemTable::new();
    assert_eq!(table.size(), 0);

    table.insert(b"k".to_vec(), b"v".to_vec()).unwrap(); // 1 + 1 = 2
    assert_eq!(table.size(), 2);

    table.insert(b"key2".to_vec(), b"val2".to_vec()).unwrap(); // 4 + 4 = 8
    assert_eq!(table.size(), 10);
}

#[test]
fn test_memtable_is_full_false_when_small() {
    let table = MemTable::new();
    table.insert(b"k".to_vec(), b"v".to_vec()).unwrap();
    assert!(!table.is_full());
}

#[test]
fn test_memtable_freeze_rejects_subsequent_inserts() {
    let table = MemTable::new();
    table.insert(b"before".to_vec(), b"v".to_vec()).unwrap();
    table.freeze();
    let result = table.insert(b"after".to_vec(), b"v".to_vec());
    assert!(result.is_err());
    // Data written before freeze is still readable
    assert_eq!(table.get(b"before"), Some(b"v".to_vec()));
}

#[test]
fn test_memtable_iter_returns_all_entries_sorted() {
    let table = MemTable::new();
    table.insert(b"c".to_vec(), b"3".to_vec()).unwrap();
    table.insert(b"a".to_vec(), b"1".to_vec()).unwrap();
    table.insert(b"b".to_vec(), b"2".to_vec()).unwrap();

    let entries = table.iter();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].0, b"a");
    assert_eq!(entries[1].0, b"b");
    assert_eq!(entries[2].0, b"c");
}

#[test]
fn test_memtable_scan_range() {
    let table = MemTable::new();
    for i in [1u8, 3, 5, 7, 9] {
        table.insert(vec![i], vec![i * 10]).unwrap();
    }

    // scan [2..6]: first key >= 2 is [3]; last included ≤ [6] is [5]
    let results = table.scan(&[2], &[6]);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, vec![3]);
    assert_eq!(results[1].0, vec![5]);
}
