use std::fs::{File, OpenOptions};
use std::io::{Read, Result, Seek, SeekFrom, Write};
use std::path::PathBuf;

pub struct WAL {
    writer: std::io::BufWriter<File>,
    sequence: u64,
    pending_sync: u64,
    sync_interval: u64,
    buf: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub enum WALRecordType {
    InferenceJob,
    GPUMetric,
    ChatMessages,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct WALRecord {
    pub sequence: u64,
    pub timestamp: u64,
    pub record_type: WALRecordType,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

impl WAL {
    pub fn open(path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;

        let writer = std::io::BufWriter::with_capacity(64 * 1024, file);

        let mut wal = Self {
            writer,
            sequence: 0,
            pending_sync: 0,
            sync_interval: 100,
            buf: Vec::with_capacity(1024),
        };
        wal.recover()?;
        Ok(wal)
    }

    pub fn append(&mut self, record: WALRecord) -> Result<()> {
        self.sequence += 1;
        let record = WALRecord {
            sequence: self.sequence,
            ..record
        };

        self.buf.clear();
        self.buf.extend_from_slice(&[0u8; 8]);
        bincode::serialize_into(&mut self.buf, &record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let len = (self.buf.len() - 8) as u64;
        self.buf[0..8].copy_from_slice(&len.to_le_bytes());

        self.writer.write_all(&self.buf)?;

        self.pending_sync += 1;
        if self.pending_sync >= self.sync_interval {
            self.writer.flush()?;
            self.writer.get_mut().sync_all()?;
            self.pending_sync = 0;
        }

        Ok(())
    }

    pub fn recover(&mut self) -> Result<Vec<WALRecord>> {
        self.writer.flush()?;
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;

        let mut records = Vec::new();

        loop {
            let mut len_buf = [0u8; 8];
            match file.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let len = u64::from_le_bytes(len_buf) as usize;
            let mut buf = vec![0u8; len];
            match file.read_exact(&mut buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let record: WALRecord = bincode::deserialize(&buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

            self.sequence = self.sequence.max(record.sequence);
            records.push(record);
        }

        Ok(records)
    }

    pub fn truncate(&mut self) -> Result<()> {
        self.writer.flush()?;
        let file = self.writer.get_mut();
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.sync_all()?;
        Ok(())
    }

    pub fn force_sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_mut().sync_all()?;
        self.pending_sync = 0;
        Ok(())
    }
}
