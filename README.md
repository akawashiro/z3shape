# z3shape

Infer tensor shapes in [ONNX](https://github.com/onnx/onnx/blob/main/docs/Operators.md) using [z3](https://github.com/Z3Prover/z3). 

`z3shape` generates a SMT-LIB2 format text file from an ONNX file. The generated file contains all constraints on tensor shapes in the given ONNX and we can get shapes of all tensors just solving constraints using z3.

## Requirements
Please install [z3](https://github.com/Z3Prover/z3).

## Usage
`cargo run <ONNX file>` gives you shape informations. You can try it by copy-and-paste following commands.

```
> git clone https://github.com/akawashiro/z3shape.git
> cd z3shape
> cargo test -- --nocapture
> cargo run squeezenet1.1-7.onnx
...
squeezenet0_conv5_fwd_shape: [1, 64, 55, 55]              
squeezenet0_conv4_fwd_shape: [1, 16, 55, 55]    
squeezenet0_conv3_fwd_shape: [1, 64, 55, 55]
squeezenet0_conv2_fwd_shape: [1, 64, 55, 55]  
squeezenet0_conv1_fwd_shape: [1, 16, 55, 55] 
squeezenet0_conv0_fwd_shape: [1, 64, 111, 111]
Check: squeezenet1.1-7.onnx_shape_inference_result.smtlib2
> tail -n 5 squeezenet1.1-7.onnx_shape_inference.smtlib2
(assert (= (head (tail shape_squeezenet0_relu25_fwd)) (head (tail shape_squeezenet0_pool3_fwd))))
(assert (= (+ (div (- (+ (+ (head (tail (tail shape_squeezenet0_relu25_fwd))) 0) 0) 13) 13) 1) (head (tail (tail shape_squeezenet0_pool3_fwd)))))
(assert (= (+ (div (- (+ (+ (head (tail (tail (tail shape_squeezenet0_relu25_fwd)))) 0) 0) 13) 13) 1) (head (tail (tail (tail shape_squeezenet0_pool3_fwd))))))
(check-sat)
(get-model)
```

## Run tests
```
cargo test -- --nocapture
```

## TODO
- Constant propagation
- Parse argv
