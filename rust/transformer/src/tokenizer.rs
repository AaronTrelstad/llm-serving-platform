use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HFTokenizer;

pub struct Tokenizer {
    inner: HFTokenizer,
    pub bos: u32,
    pub eos: u32,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = HFTokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            inner,
            bos: 128000,
            eos: 128001,
        })
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut ids = encoding.get_ids().to_vec();

        if add_bos {
            ids.insert(0, self.bos);
        }

        Ok(ids)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn decode_token(&self, token: u32) -> Result<String> {
        self.decode(&[token])
    }
}
