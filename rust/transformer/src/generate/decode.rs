use super::kv_transfer::{SerializedKV, deserialize_kv_caches};
use crate::model::Llama;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use candle_core::{Device, Tensor};

pub struct DecodeResult {
    pub job_id: String,
    pub output: String,
    pub n_tokens: usize,
    pub decode_worker: String,
}

pub struct DecodeWorker {
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub worker_id: String,
}

impl DecodeWorker {
    pub fn new(model: Llama, tokenizer: Tokenizer, device: Device, worker_id: String) -> Self {
        Self {
            model,
            tokenizer,
            device,
            worker_id,
        }
    }

    pub fn decode(
        &self,
        job_id: &str,
        first_token: u32,
        n_prompt: usize,
        serialized_kv: Vec<SerializedKV>,
        max_tokens: usize,
    ) -> Result<DecodeResult> {
        let mut kv_caches =
            deserialize_kv_caches(serialized_kv, self.model.n_layers(), &self.device)?;

        let mut output_tokens = vec![first_token];
        let mut current_token = first_token;
        let mut pos = n_prompt;

        for _ in 0..max_tokens {
            if current_token == self.tokenizer.eos {
                break;
            }

            let token_tensor = Tensor::from_vec(vec![current_token], (1, 1), &self.device)?;

            let logits = self.model.forward(&token_tensor, &mut kv_caches, pos)?;

            current_token = logits
                .squeeze(0)?
                .squeeze(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;

            output_tokens.push(current_token);
            pos += 1;
        }

        let output = self.tokenizer.decode(&output_tokens)?;

        Ok(DecodeResult {
            job_id: job_id.to_string(),
            output,
            n_tokens: output_tokens.len(),
            decode_worker: self.worker_id.clone(),
        })
    }
}
