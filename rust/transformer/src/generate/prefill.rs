use super::kv_transfer::{SerializedKV, serialize_kv_caches};
use crate::model::Llama;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use candle_core::{Device, Tensor};

pub struct PrefillResult {
    pub job_id: String,
    pub first_token: u32,
    pub n_tokens: usize,
    pub prefill_worker: String,
    pub serialized_kv: Vec<SerializedKV>,
}

pub struct PrefillWorker {
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub worker_id: String,
}

impl PrefillWorker {
    pub fn new(model: Llama, tokenizer: Tokenizer, device: Device, worker_id: String) -> Self {
        Self {
            model,
            tokenizer,
            device,
            worker_id,
        }
    }

    pub fn prefill(&self, job_id: &str, prompt: &str) -> Result<PrefillResult> {
        let token_ids = self.tokenizer.encode(prompt, true)?;
        let n_tokens = token_ids.len();

        let tokens = Tensor::from_vec(token_ids, (1, n_tokens), &self.device)?;

        let mut kv_caches = self.model.empty_kv_caches();
        let logits = self.model.forward(&tokens, &mut kv_caches, 0)?;

        let first_token = logits
            .squeeze(0)?
            .squeeze(0)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        let serialized_kv = serialize_kv_caches(&kv_caches)?;

        Ok(PrefillResult {
            job_id: job_id.to_string(),
            first_token,
            n_tokens,
            prefill_worker: self.worker_id.clone(),
            serialized_kv,
        })
    }
}
