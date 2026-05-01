use transformer::ffn::SwiGLU;
use candle_core::{Device, Tensor};
use candle_nn::Linear;

#[test]
fn test_swiglu_shape_preserved() {
    let device    = Device::Cpu;
    let hidden    = 512usize;
    let inter     = 1024usize;

    let gate_w = Tensor::randn(0f32, 1f32, (inter, hidden), &device).unwrap();
    let up_w   = Tensor::randn(0f32, 1f32, (inter, hidden), &device).unwrap();
    let down_w = Tensor::randn(0f32, 1f32, (hidden, inter), &device).unwrap();

    let gate_proj = Linear::new(gate_w, None);
    let up_proj   = Linear::new(up_w,   None);
    let down_proj = Linear::new(down_w, None);

    let ffn = SwiGLU::new(gate_proj, up_proj, down_proj);

    let x   = Tensor::randn(0f32, 1f32, (1, 3, hidden), &device).unwrap();
    let out = ffn.forward(&x).unwrap();

    assert_eq!(out.dims(), &[1, 3, hidden]);
}
