# Database Design

This is a time-series database made in Rust from scratch. It's designed for append heavy workloads (inference jobs, GPU metrics, etc.) and supports writes, reads and range queries.

---

## Components

### WAL (Write-Ahead Log)

The WAL provides crash saftey. Every write is appended to disk and fsynced before touching memory. On a crash, the WAL is replayed to rebuild the memtable.

### Memtable
