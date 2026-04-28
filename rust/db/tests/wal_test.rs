use db::wal::wal::{WAL, WALRecord, WALRecordType};
use tempfile::tempdir;

#[test]
fn test_wal_append_and_recover() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.wal");

    let mut wal = WAL::open(path.clone()).unwrap();
    wal.append(WALRecord {
        sequence:    0,
        timestamp:   1000,
        record_type: WALRecordType::InferenceJob,
        key:         b"job_001".to_vec(),
        value:       b"test_value".to_vec(),
    }).unwrap();
    wal.append(WALRecord {
        sequence:    0,
        timestamp:   2000,
        record_type: WALRecordType::GPUMetric,
        key:         b"worker_01".to_vec(),
        value:       b"metric_value".to_vec(),
    }).unwrap();
    wal.force_sync().unwrap();
    drop(wal);

    let mut wal2 = WAL::open(path).unwrap();
    let records = wal2.recover().unwrap();

    assert_eq!(records.len(), 2);
    assert_eq!(records[0].key, b"job_001");
    assert_eq!(records[0].timestamp, 1000);
    assert_eq!(records[1].key, b"worker_01");
    assert_eq!(records[1].timestamp, 2000);
}

#[test]
fn test_wal_sequence_monotonic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.wal");
    let mut wal = WAL::open(path).unwrap();

    for i in 0..10 {
        wal.append(WALRecord {
            sequence:    0,
            timestamp:   i,
            record_type: WALRecordType::InferenceJob,
            key:         format!("key_{}", i).into_bytes(),
            value:       b"value".to_vec(),
        }).unwrap();
    }

    wal.force_sync().unwrap();
    let records = wal.recover().unwrap();

    for i in 0..records.len() - 1 {
        assert!(records[i].sequence < records[i + 1].sequence);
    }
}

#[test]
fn test_wal_truncate() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.wal");
    let mut wal = WAL::open(path.clone()).unwrap();

    wal.append(WALRecord {
        sequence:    0,
        timestamp:   1000,
        record_type: WALRecordType::InferenceJob,
        key:         b"job_001".to_vec(),
        value:       b"value".to_vec(),
    }).unwrap();
    wal.force_sync().unwrap();
    wal.truncate().unwrap();
    drop(wal);

    let mut wal2 = WAL::open(path).unwrap();
    let records = wal2.recover().unwrap();
    assert_eq!(records.len(), 0);
}

#[test]
fn test_wal_crash_recovery() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.wal");
    let mut wal = WAL::open(path.clone()).unwrap();

    for i in 0..5u64 {
        wal.append(WALRecord {
            sequence:    0,
            timestamp:   i * 1000,
            record_type: WALRecordType::InferenceJob,
            key:         format!("job_{:03}", i).into_bytes(),
            value:       b"value".to_vec(),
        }).unwrap();
    }
    wal.force_sync().unwrap();

    drop(wal);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap();
    let len = file.metadata().unwrap().len();
    file.set_len(len - 3).unwrap(); 
    drop(file);

    let mut wal2 = WAL::open(path).unwrap();
    let records = wal2.recover().unwrap();
    assert!(records.len() >= 4);  
}
