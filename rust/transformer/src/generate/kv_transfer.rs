use crate::attention::KVCache;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[derive(Debug)]
pub struct SerializedKV {
    pub layer_idx: usize,
    pub k_data: Vec<u8>,
    pub v_data: Vec<u8>,
    pub k_shape: Vec<usize>,
    pub v_shape: Vec<usize>,
    pub dtype: String,
}

pub fn serialize_kv_caches(caches: &[Option<KVCache>]) -> Result<Vec<SerializedKV>> {
    caches
        .iter()
        .enumerate()
        .filter_map(|(i, cache)| cache.as_ref().map(|c| (i, c)))
        .map(|(layer_idx, cache)| {
            let k_data = cache
                .k
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let v_data = cache
                .v
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let k_bytes = bytemuck::cast_slice(&k_data).to_vec();
            let v_bytes = bytemuck::cast_slice(&v_data).to_vec();

            Ok(SerializedKV {
                layer_idx,
                k_data: k_bytes,
                v_data: v_bytes,
                k_shape: cache.k.dims().to_vec(),
                v_shape: cache.v.dims().to_vec(),
                dtype: "f32".to_string(),
            })
        })
        .collect()
}

pub fn deserialize_kv_caches(
    serialized: Vec<SerializedKV>,
    n_layers: usize,
    device: &Device,
) -> Result<Vec<Option<KVCache>>> {
    let mut caches: Vec<Option<KVCache>> = vec![None; n_layers];

    for s in serialized {
        let k_f32: &[f32] = bytemuck::cast_slice(&s.k_data);
        let v_f32: &[f32] = bytemuck::cast_slice(&s.v_data);

        let k = Tensor::from_slice(k_f32, s.k_shape.as_slice(), device)?;
        let v = Tensor::from_slice(v_f32, s.v_shape.as_slice(), device)?;

        caches[s.layer_idx] = Some(KVCache { k, v });
    }

    Ok(caches)
}
