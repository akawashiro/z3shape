# z3shape

Infer tensor shapes in ONNX using Z3.

## Usage
```
# cargo test -- --nocapture automatically download squeezenet1.1-7.onnx from GitHub.
> cargo run squeezenet1.1-7.onnx
...
squeezenet0_conv5_fwd_shape: [1, 64, 55, 55]              
squeezenet0_conv4_fwd_shape: [1, 16, 55, 55]    
squeezenet0_conv3_fwd_shape: [1, 64, 55, 55]
squeezenet0_conv2_fwd_shape: [1, 64, 55, 55]  
squeezenet0_conv1_fwd_shape: [1, 16, 55, 55] 
squeezenet0_conv0_fwd_shape: [1, 64, 111, 111]
Check: squeezenet1.1-7.onnx_shape_inference_result.smtlib2
```

## Run tests
```
cargo test -- --nocapture
```

## TODO
- Constant propagation for Reshape
- Improve coverage
    - Tiny YOLOv2
    - Bidirectional Attention Flow
- Parse argv
