use db::lsm::compaction::Compaction;
use db::lsm::memtable::MemTable;
use db::lsm::sstable::SSTable;
use tempfile::tempdir;

/// Build a key in the [type_byte][timestamp_be][identifier] format that Series writes
/// and compaction expects.
fn job_key(timestamp: u64, job_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9 + job_id.len());
    key.push(0u8); // InferenceJob
    key.extend_from_slice(&timestamp.to_be_bytes());
    key.extend_from_slice(job_id.as_bytes());
    key
}

fn metric_key(timestamp: u64, worker_id: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9 + worker_id.len());
    key.push(1u8); // GPUMetric
    key.extend_from_slice(&timestamp.to_be_bytes());
    key.extend_from_slice(worker_id.as_bytes());
    key
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[test]
fn test_compaction_merges_multiple_sstables_into_one() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    let now = now_secs();

    // Build 3 SSTables, each with 2 fresh records
    let mut sstables = Vec::new();
    for i in 0..3usize {
        let path = data_dir.join(format!("sstable_{:03}.sst", i));
        let memtable = MemTable::new();
        memtable
            .insert(job_key(now, &format!("job-{}", i * 2)), b"v".to_vec())
            .unwrap();
        memtable
            .insert(job_key(now, &format!("job-{}", i * 2 + 1)), b"v".to_vec())
            .unwrap();
        sstables.push(SSTable::write(path, &memtable).unwrap());
    }

    let compaction = Compaction::new();
    let mut result = compaction.compact(sstables, &data_dir).unwrap();

    assert_eq!(result.len(), 1, "compaction should produce a single SSTable");
    let records = result[0].iter().unwrap();
    assert_eq!(records.len(), 6, "all 6 records should be present after merge");
}

#[test]
fn test_compaction_deduplicates_same_key_across_sstables() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    let now = now_secs();

    // Two SSTables with the same key — compaction should keep one copy
    let mut sstables = Vec::new();
    for i in 0..2usize {
        let path = data_dir.join(format!("sstable_{:03}.sst", i));
        let memtable = MemTable::new();
        memtable
            .insert(job_key(now, "job-dup"), b"value".to_vec())
            .unwrap();
        sstables.push(SSTable::write(path, &memtable).unwrap());
    }

    let compaction = Compaction::new();
    let mut result = compaction.compact(sstables, &data_dir).unwrap();
    let records = result[0].iter().unwrap();
    assert_eq!(records.len(), 1);
}

#[test]
fn test_compaction_drops_expired_records() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    let now = now_secs();

    // TTL is 7 days. old_ts is 8 days ago → should be dropped.
    let old_ts = now - 8 * 24 * 3600;
    // new_ts is 1 day ago → should survive.
    let new_ts = now - 1 * 24 * 3600;

    let path = data_dir.join("sstable_000.sst");
    let memtable = MemTable::new();
    memtable
        .insert(job_key(old_ts, "job-old"), b"expired".to_vec())
        .unwrap();
    memtable
        .insert(job_key(new_ts, "job-new"), b"alive".to_vec())
        .unwrap();

    let sstables = vec![SSTable::write(path, &memtable).unwrap()];

    let compaction = Compaction::new();
    let mut result = compaction.compact(sstables, &data_dir).unwrap();
    let records = result[0].iter().unwrap();

    assert_eq!(records.len(), 1, "only the non-expired record should survive");
    // Verify the surviving record has the recent timestamp in its key
    let ts = u64::from_be_bytes(records[0].0[1..9].try_into().unwrap());
    assert_eq!(ts, new_ts);
}

#[test]
fn test_compaction_keeps_metric_records_within_ttl() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    let now = now_secs();

    let fresh_ts = now - 3600; // 1 hour ago — well within 7-day TTL

    let path = data_dir.join("sstable_000.sst");
    let memtable = MemTable::new();
    memtable
        .insert(metric_key(fresh_ts, "worker-1"), b"metric-data".to_vec())
        .unwrap();
    memtable
        .insert(metric_key(fresh_ts, "worker-2"), b"metric-data".to_vec())
        .unwrap();

    let sstables = vec![SSTable::write(path, &memtable).unwrap()];

    let compaction = Compaction::new();
    let mut result = compaction.compact(sstables, &data_dir).unwrap();
    let records = result[0].iter().unwrap();

    assert_eq!(records.len(), 2, "fresh metric records should be kept");
}

#[test]
fn test_compaction_removes_old_sstable_files() {
    let dir = tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    let now = now_secs();

    let paths: Vec<_> = (0..2)
        .map(|i| data_dir.join(format!("sstable_{:03}.sst", i)))
        .collect();

    let mut sstables = Vec::new();
    for path in &paths {
        let memtable = MemTable::new();
        memtable
            .insert(job_key(now, "job-x"), b"v".to_vec())
            .unwrap();
        sstables.push(SSTable::write(path.clone(), &memtable).unwrap());
    }

    let compaction = Compaction::new();
    compaction.compact(sstables, &data_dir).unwrap();

    for path in &paths {
        assert!(!path.exists(), "old SSTable file should be deleted: {:?}", path);
    }
}
