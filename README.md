# z3shape

Infer tensor shapes in ONNX using Z3.

## Run tests
```
cargo test -- --nocapture
```

## TODO
- Constant propagation for Reshape
- Improve coverage
    - Tiny YOLOv2
    - ArcFace
    - Bidirectional Attention Flow
- Download ONNX files automatically
- Parse the result Z3
