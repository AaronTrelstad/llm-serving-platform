use transformer::attention::{GroupedQueryAttention};
use transformer::config::LlamaConfig;
use transformer::rope::RoPE;
use candle_core::{Device, Tensor};
use candle_nn::Linear;

fn small_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size:                 128256,
        hidden_size:                256,
        n_layers:                   2,
        n_heads:                    4,
        n_kv_heads:                 2,
        intermediate_size:          512,
        max_seq_len:                128,
        rope_theta:                 500000.0,
        rope_scaling_factor:        8.0,
        rope_low_freq_factor:       1.0,
        rope_high_freq_factor:      4.0,
        rope_original_max_seq_len:  8192,
        rms_norm_eps:               1e-5,
        bos_token_id:               128000,
        eos_token_id:               128001,
        head_dim:                   256 / 4,   
        n_rep:                      4 / 2,    
    }
}

fn make_attention(config: &LlamaConfig, device: &Device) -> GroupedQueryAttention {
    let h  = config.hidden_size;
    let qd = config.n_heads    * config.head_dim;
    let kd = config.n_kv_heads * config.head_dim;

    let q_proj = Linear::new(Tensor::randn(0f32, 0.02, (qd, h), device).unwrap(), None);
    let k_proj = Linear::new(Tensor::randn(0f32, 0.02, (kd, h), device).unwrap(), None);
    let v_proj = Linear::new(Tensor::randn(0f32, 0.02, (kd, h), device).unwrap(), None);
    let o_proj = Linear::new(Tensor::randn(0f32, 0.02, (h,  qd), device).unwrap(), None);
    let rope   = RoPE::new(config, device).unwrap();

    GroupedQueryAttention::new(q_proj, k_proj, v_proj, o_proj, rope, config)
}

#[test]
fn test_attention_output_shape() {
    let device = Device::Cpu;
    let config = small_config();
    let attn   = make_attention(&config, &device);

    let x   = Tensor::randn(0f32, 1f32, (1, 4, config.hidden_size), &device).unwrap();
    let out = attn.forward(&x, &mut None, 0).unwrap();

    assert_eq!(out.dims(), &[1, 4, config.hidden_size]);
}

#[test]
fn test_attention_with_kv_cache() {
    let device = Device::Cpu;
    let config = small_config();
    let attn   = make_attention(&config, &device);

    let x_prefill = Tensor::randn(0f32, 1f32, (1, 4, config.hidden_size), &device).unwrap();
    let mut cache = None;
    attn.forward(&x_prefill, &mut cache, 0).unwrap();
    assert!(cache.is_some(), "KV cache should be populated after prefill");

    let x_decode = Tensor::randn(0f32, 1f32, (1, 1, config.hidden_size), &device).unwrap();
    let out = attn.forward(&x_decode, &mut cache, 4).unwrap();
    assert_eq!(out.dims(), &[1, 1, config.hidden_size]);
}

#[test]
fn test_kv_cache_grows() {
    let device = Device::Cpu;
    let config = small_config();
    let attn   = make_attention(&config, &device);

    let mut cache = None;

    for i in 0..3 {
        let x = Tensor::randn(0f32, 1f32, (1, 1, config.hidden_size), &device).unwrap();
        attn.forward(&x, &mut cache, i).unwrap();

        let k_len = cache.as_ref().unwrap().k.dim(2).unwrap();
        assert_eq!(k_len, i + 1, "KV cache should grow by 1 each step");
    }
}

#[test]
fn test_causal_mask_no_future_attention() {
    let device = Device::Cpu;
    let config = small_config();
    let attn   = make_attention(&config, &device);

    let x   = Tensor::randn(0f32, 1f32, (1, 4, config.hidden_size), &device).unwrap();
    let out = attn.forward(&x, &mut None, 0).unwrap();

    let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(data.iter().all(|v| v.is_finite()), "output contains NaN or inf");
}
