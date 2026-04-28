use db::lsm::bloom::BloomFilter;

#[test]
fn test_bloom_inserted_keys_always_found() {
    let mut filter = BloomFilter::new(1000, 0.01);
    let keys: Vec<Vec<u8>> = (0u32..100).map(|i| i.to_le_bytes().to_vec()).collect();

    for key in &keys {
        filter.insert(key);
    }
    for key in &keys {
        assert!(filter.contains(key), "false negative for key {:?}", key);
    }
}

#[test]
fn test_bloom_false_positive_rate_within_bounds() {
    let mut filter = BloomFilter::new(1000, 0.01);

    // Insert 1000 keys in range 0..1000
    for i in 0u32..1000 {
        filter.insert(&i.to_le_bytes());
    }

    // Check 1000 keys that were never inserted (range 10000..11000)
    let mut false_positives = 0;
    for i in 10000u32..11000 {
        if filter.contains(&i.to_le_bytes()) {
            false_positives += 1;
        }
    }

    // Configured FPR is 1%; allow 3× headroom for statistical variance
    assert!(
        false_positives < 30,
        "false positive rate too high: {}/1000",
        false_positives
    );
}

#[test]
fn test_bloom_empty_filter_contains_nothing() {
    let filter = BloomFilter::new(1000, 0.01);
    assert!(!filter.contains(b"anything"));
}

#[test]
fn test_bloom_large_filter_no_false_negatives() {
    let mut filter = BloomFilter::new(100_000, 0.001);
    let keys: Vec<Vec<u8>> = (0u32..10_000).map(|i| i.to_le_bytes().to_vec()).collect();

    for key in &keys {
        filter.insert(key);
    }
    for key in &keys {
        assert!(filter.contains(key));
    }
}
