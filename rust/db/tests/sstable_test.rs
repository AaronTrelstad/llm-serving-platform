use db::lsm::memtable::MemTable;
use db::lsm::sstable::SSTable;
use tempfile::tempdir;

#[test]
fn test_sstable_write_and_get() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.sst");

    let memtable = MemTable::new();
    memtable.insert(b"key1".to_vec(), b"value1".to_vec()).unwrap();
    memtable.insert(b"key2".to_vec(), b"value2".to_vec()).unwrap();

    let mut sst = SSTable::write(path, &memtable).unwrap();

    assert_eq!(sst.get(b"key1").unwrap(), Some(b"value1".to_vec()));
    assert_eq!(sst.get(b"key2").unwrap(), Some(b"value2".to_vec()));
    assert_eq!(sst.get(b"missing").unwrap(), None);
}

#[test]
fn test_sstable_open_reads_persisted_data() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("persist.sst");

    let memtable = MemTable::new();
    memtable.insert(b"hello".to_vec(), b"world".to_vec()).unwrap();
    memtable.insert(b"foo".to_vec(), b"bar".to_vec()).unwrap();
    SSTable::write(path.clone(), &memtable).unwrap();

    // Re-open from disk
    let mut sst = SSTable::open(path).unwrap();
    assert_eq!(sst.get(b"hello").unwrap(), Some(b"world".to_vec()));
    assert_eq!(sst.get(b"foo").unwrap(), Some(b"bar".to_vec()));
    assert_eq!(sst.get(b"absent").unwrap(), None);
}

#[test]
fn test_sstable_bloom_filter_rejects_absent_keys() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bloom.sst");

    let memtable = MemTable::new();
    for i in 0u32..100 {
        memtable
            .insert(i.to_le_bytes().to_vec(), b"v".to_vec())
            .unwrap();
    }
    let mut sst = SSTable::write(path, &memtable).unwrap();

    // A key with a completely different byte pattern is almost certainly absent
    assert_eq!(sst.get(b"definitely_not_a_stored_key").unwrap(), None);
}

#[test]
fn test_sstable_iter_returns_all_records() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("iter.sst");

    let memtable = MemTable::new();
    memtable.insert(b"a".to_vec(), b"1".to_vec()).unwrap();
    memtable.insert(b"b".to_vec(), b"2".to_vec()).unwrap();
    memtable.insert(b"c".to_vec(), b"3".to_vec()).unwrap();

    let mut sst = SSTable::write(path, &memtable).unwrap();
    let records = sst.iter().unwrap();

    assert_eq!(records.len(), 3);
    // iter follows insertion order of BufWriter which mirrors memtable.iter() — sorted
    assert_eq!(records[0], (b"a".to_vec(), b"1".to_vec()));
    assert_eq!(records[1], (b"b".to_vec(), b"2".to_vec()));
    assert_eq!(records[2], (b"c".to_vec(), b"3".to_vec()));
}

#[test]
fn test_sstable_large_values() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("large.sst");

    let memtable = MemTable::new();
    let large_value = vec![0xABu8; 4096];
    memtable
        .insert(b"bigkey".to_vec(), large_value.clone())
        .unwrap();

    let mut sst = SSTable::write(path, &memtable).unwrap();
    assert_eq!(sst.get(b"bigkey").unwrap(), Some(large_value));
}
