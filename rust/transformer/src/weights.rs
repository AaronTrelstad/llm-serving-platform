use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

use crate::attention::GroupedQueryAttention;
use crate::block::LlamaBlock;
use crate::config::LlamaConfig;
use crate::ffn::SwiGLU;
use crate::model::Llama;
use crate::rmsnorm::RMSNorm;
use crate::rope::RoPE;

pub fn load_weights(
    model_dir: &Path,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let index_str = std::fs::read_to_string(&index_path)
        .context("failed to read model.safetensors.index.json")?;
    let index: serde_json::Value = serde_json::from_str(&index_str)?;

    let weight_map = index["weight_map"]
        .as_object()
        .context("invalid index format")?;

    let mut files: Vec<String> = weight_map
        .values()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    files.sort();
    files.dedup();

    let mut weights: HashMap<String, Tensor> = HashMap::new();

    for file in &files {
        let path = model_dir.join(file);

        let file = std::fs::File::open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let st = SafeTensors::deserialize(&mmap).context("failed to parse safetensors")?;

        for (name, view) in st.tensors() {
            let tensor = Tensor::from_raw_buffer(
                view.data(),
                candle_core::DType::BF16, 
                &view.shape().iter().map(|&d| d as usize).collect::<Vec<_>>(),
                device,
            )?
            .to_dtype(dtype)?; 

            weights.insert(name.to_string(), tensor);
        }
    }

    Ok(weights)
}

fn get_weight(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    weights
        .get(name)
        .cloned()
        .with_context(|| format!("weight not found: {}", name))
}

fn load_rmsnorm(weights: &HashMap<String, Tensor>, name: &str, eps: f64) -> Result<RMSNorm> {
    let weight = get_weight(weights, name)?;
    Ok(RMSNorm::new(weight, eps))
}

fn load_linear(weights: &HashMap<String, Tensor>, name: &str) -> Result<Linear> {
    let weight = get_weight(weights, name)?;
    Ok(Linear::new(weight, None))
}

fn load_block(
    weights: &HashMap<String, Tensor>,
    layer_idx: usize,
    config: &LlamaConfig,
    device: &Device,
) -> Result<LlamaBlock> {
    let prefix = format!("model.layers.{}", layer_idx);

    let attn_norm = load_rmsnorm(
        weights,
        &format!("{}.input_layernorm.weight", prefix),
        config.rms_norm_eps,
    )?;
    let ffn_norm = load_rmsnorm(
        weights,
        &format!("{}.post_attention_layernorm.weight", prefix),
        config.rms_norm_eps,
    )?;

    let q_proj = load_linear(weights, &format!("{}.self_attn.q_proj.weight", prefix))?;
    let k_proj = load_linear(weights, &format!("{}.self_attn.k_proj.weight", prefix))?;
    let v_proj = load_linear(weights, &format!("{}.self_attn.v_proj.weight", prefix))?;
    let o_proj = load_linear(weights, &format!("{}.self_attn.o_proj.weight", prefix))?;

    let rope = RoPE::new(config, device)?;
    let attn = GroupedQueryAttention::new(q_proj, k_proj, v_proj, o_proj, rope, config);

    let gate_proj = load_linear(weights, &format!("{}.mlp.gate_proj.weight", prefix))?;
    let up_proj = load_linear(weights, &format!("{}.mlp.up_proj.weight", prefix))?;
    let down_proj = load_linear(weights, &format!("{}.mlp.down_proj.weight", prefix))?;
    let ffn = SwiGLU::new(gate_proj, up_proj, down_proj);

    Ok(LlamaBlock::new(attn_norm, attn, ffn_norm, ffn))
}

pub fn load_model(model_dir: &Path, config: &LlamaConfig, device: &Device) -> Result<Llama> {
    let dtype = match device {
        Device::Cpu => DType::F32,
        Device::Cuda(_) => DType::BF16,
        Device::Metal(_) => DType::F32,
    };

    let weights = load_weights(model_dir, device, dtype)?;

    let embed_weight = get_weight(&weights, "model.embed_tokens.weight")?;
    let embedding = Embedding::new(embed_weight, config.hidden_size);

    let blocks: Vec<LlamaBlock> = (0..config.n_layers)
        .map(|i| {
            load_block(&weights, i, config, device)
        })
        .collect::<Result<Vec<_>>>()?;

    let norm = load_rmsnorm(&weights, "model.norm.weight", config.rms_norm_eps)?;
    let lm_head = load_linear(&weights, "lm_head.weight")?;

    Ok(Llama::new(embedding, blocks, norm, lm_head, config.clone()))
}
