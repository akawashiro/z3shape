extern crate protobuf;
extern crate reqwest;

use protobuf::{CodedInputStream, Message};
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

mod onnx;
use onnx::ModelProto;
mod z3;
use crate::z3::*;

fn shape_name(n: &str) -> String {
    String::from("shape_") + n
}

fn gen_constraints(model: &onnx::ModelProto) -> (HashSet<Z3Exp>, Vec<Z3Exp>) {
    let mut decares = HashSet::new();
    let mut conditions = Vec::new();

    for inout in model.graph.input.iter().chain(model.graph.output.iter()) {
        // dbg!(&inout);
        let name = shape_name(inout.name.as_ref().unwrap());
        decares.insert(Z3Exp::DecareConst(
            name.clone(),
            Z3Type::List(Box::new(Z3Type::Int)),
        ));

        let mut shape = Vec::new();
        if let onnx::type_proto::Value::TensorType(t) = inout.type_.clone().unwrap().value.unwrap()
        {
            for d in t.shape.dim.iter() {
                if let onnx::tensor_shape_proto::dimension::Value::DimValue(i) =
                    d.value.as_ref().unwrap()
                {
                    shape.push(*i);
                } else if let onnx::tensor_shape_proto::dimension::Value::DimParam(_) =
                    d.value.as_ref().unwrap()
                {
                    // TODO: Symbolic value
                    shape.push(0);
                }
            }
        }

        let mut name_e = Z3Exp::Variable(name);
        for s in shape.iter() {
            let eq = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                Box::new(Z3Exp::Head(Box::new(name_e.clone()))),
                Box::new(Z3Exp::Int(*s)),
            )));
            name_e = Z3Exp::Tail(Box::new(name_e));
            conditions.push(eq);
        }
    }

    for init in model.graph.initializer.iter() {
        if let Some(name) = &init.name {
            let name = shape_name(name);
            decares.insert(Z3Exp::DecareConst(
                name.clone(),
                Z3Type::List(Box::new(Z3Type::Int)),
            ));

            let mut name_e = Z3Exp::Variable(name);
            for s in init.dims.iter() {
                let eq = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Head(Box::new(name_e.clone()))),
                    Box::new(Z3Exp::Int(*s)),
                )));
                name_e = Z3Exp::Tail(Box::new(name_e));
                conditions.push(eq);
            }
        }
    }

    fn get_attribute<'a>(node: &'a onnx::NodeProto, att: &str) -> Option<&'a onnx::AttributeProto> {
        for a in node.attribute.iter() {
            if a.name == Some(att.to_string()) {
                return Some(a);
            }
        }
        None
    }

    for node in model.graph.node.iter() {
        if let Some(op_type) = &node.op_type {
            if node.name == Some(String::from("Gather_100")) {
                break;
            }

            if op_type == "Reshape" || op_type == "Slice" {
                // TODO (akawashiro): We need constant propagation.
            } else if op_type == "Shape" {
                // TODO (akawashiro): We need len(list) in Z3.
            } else if op_type == "Resize" {
            } else if op_type == "Constant" || op_type == "Gather" || op_type == "Unsqueeze" {
            } else if op_type == "Gemm" {
                assert!(node.input.len() == 2 || node.input.len() == 3);
                assert_eq!(node.output.len(), 1);

                let trans_a = get_attribute(node, "transA").map_or(0, |a| a.i.map_or(0, |x| x));
                let trans_b = get_attribute(node, "transB").map_or(0, |a| a.i.map_or(0, |x| x));

                let mat_a = Z3Exp::Variable(shape_name(&node.input[0]));
                let mat_b = Z3Exp::Variable(shape_name(&node.input[1]));
                let mat_y = Z3Exp::Variable(shape_name(&node.output[0]));
                decares.insert(dims_dec(shape_name(&node.input[0])));
                decares.insert(dims_dec(shape_name(&node.input[1])));
                decares.insert(dims_dec(shape_name(&node.output[0])));

                let mut dim_m_a = first(mat_a.clone());
                let mut dim_k_a = second(mat_a);
                if trans_a == 1 {
                    std::mem::swap(&mut dim_m_a, &mut dim_k_a);
                }
                let mut dim_n_b = first(mat_b.clone());
                let mut dim_k_b = second(mat_b);
                if trans_b == 1 {
                    std::mem::swap(&mut dim_n_b, &mut dim_k_b);
                }
                let dim_m_y = first(mat_y.clone());
                let dim_n_y = second(mat_y);

                conditions.push(ass_eq(dim_m_a, dim_m_y));
                conditions.push(ass_eq(dim_k_a, dim_k_b));
                conditions.push(ass_eq(dim_n_b, dim_n_y));
            } else if op_type == "MaxPool" || op_type == "AveragePool" {
                assert_eq!(node.input.len(), 1);
                assert_eq!(node.output.len(), 1);

                let kernel_shape_att = get_attribute(node, "kernel_shape").unwrap();
                let kernel_shape = &kernel_shape_att.ints;
                assert_eq!(kernel_shape.len(), 2);

                let default_pads = vec![0, 0, 0, 0];
                let pads = get_attribute(node, "pads").map_or(&default_pads, |a| &a.ints);
                assert_eq!(pads.len(), 4);

                let strides_att = get_attribute(node, "strides").unwrap();
                let strides = &strides_att.ints;
                assert_eq!(strides.len(), 2);

                decares.insert(dims_dec(shape_name(&node.input[0])));
                decares.insert(dims_dec(shape_name(&node.output[0])));

                let in_image = Z3Exp::Variable(shape_name(&node.input[0]));
                let out_image = Z3Exp::Variable(shape_name(&node.output[0]));

                let in_batch = head(in_image.clone());
                let out_batch = head(out_image.clone());
                conditions.push(ass_eq(in_batch, out_batch));

                let in_ch = head(tail(in_image.clone()));
                let out_ch = head(tail(out_image.clone()));
                conditions.push(ass_eq(in_ch, out_ch));

                let k_h = kernel_shape[0];
                let k_w = kernel_shape[1];
                let in_h = plus(
                    plus(head(tail(tail(in_image.clone()))), int(pads[0])),
                    int(pads[2]),
                );
                let in_w = plus(
                    plus(head(tail(tail(tail(in_image)))), int(pads[1])),
                    int(pads[3]),
                );
                let out_h = head(tail(tail(out_image.clone())));
                let out_w = head(tail(tail(tail(out_image))));

                let dilation = 1;

                conditions.push(ass_eq(
                    plus(
                        div(sub(in_h, int((k_h - 1) * dilation + 1)), int(strides[0])),
                        int(1),
                    ),
                    out_h,
                ));
                conditions.push(ass_eq(
                    plus(
                        div(sub(in_w, int((k_w - 1) * dilation + 1)), int(strides[1])),
                        int(1),
                    ),
                    out_w,
                ));
            } else if op_type == "Conv" {
                assert!(node.input.len() == 2 || node.input.len() == 3, "{:?}", node);
                assert_eq!(node.output.len(), 1, "{:?}", node);

                let dilations_att = get_attribute(node, "dilations").unwrap();
                let dilations = &dilations_att.ints;
                assert_eq!(dilations.len(), 2);

                let group_att = get_attribute(node, "group").unwrap();
                let group = group_att.i.unwrap();

                let kernel_shape_att = get_attribute(node, "kernel_shape").unwrap();
                let kernel_shape = &kernel_shape_att.ints;
                assert_eq!(kernel_shape.len(), 2);

                let default_pads = vec![0, 0, 0, 0];
                let pads = get_attribute(node, "pads").map_or(&default_pads, |a| &a.ints);
                assert_eq!(pads.len(), 4);

                let strides_att = get_attribute(node, "strides").unwrap();
                let strides = &strides_att.ints;
                assert_eq!(strides.len(), 2);

                decares.insert(dims_dec(shape_name(&node.input[0])));
                decares.insert(dims_dec(shape_name(&node.input[1])));
                decares.insert(dims_dec(shape_name(&node.output[0])));

                let in_image = Z3Exp::Variable(shape_name(&node.input[0]));
                let weight = Z3Exp::Variable(shape_name(&node.input[1]));
                let out_image = Z3Exp::Variable(shape_name(&node.output[0]));

                let in_batch = head(in_image.clone());
                let out_batch = head(out_image.clone());
                conditions.push(ass_eq(in_batch, out_batch));

                let in_ch_eq = ass_eq(
                    head(tail(in_image.clone())),
                    mul(int(group), head(tail(weight.clone()))),
                );
                conditions.push(in_ch_eq);

                let out_ch_eq1 = ass_eq(head(weight.clone()), head(tail(out_image.clone())));
                conditions.push(out_ch_eq1);
                if node.input.len() == 3 {
                    decares.insert(dims_dec(shape_name(&node.input[2])));
                    let bias = Z3Exp::Variable(shape_name(&node.input[2]));
                    let out_ch_eq2 = ass_eq(head(weight), head(bias));
                    conditions.push(out_ch_eq2);
                }

                let k_h = (kernel_shape[0] - 1) * dilations[0] + 1;
                let k_w = (kernel_shape[1] - 1) * dilations[1] + 1;
                let in_h = plus(
                    plus(head(tail(tail(in_image.clone()))), int(pads[0])),
                    int(pads[2]),
                );
                let in_w = plus(
                    plus(head(tail(tail(tail(in_image)))), int(pads[1])),
                    int(pads[3]),
                );
                let out_h = head(tail(tail(out_image.clone())));
                let out_w = head(tail(tail(tail(out_image))));

                conditions.push(ass_eq(div(sub(in_h, int(k_h - 1)), int(strides[0])), out_h));
                conditions.push(ass_eq(div(sub(in_w, int(k_w - 1)), int(strides[1])), out_w));
            } else if op_type == "GlobalAveragePool" {
                assert_eq!(node.input.len(), 1);
                assert_eq!(node.input.len(), node.output.len());

                let i = shape_name(&node.input[0]);
                let o = shape_name(&node.output[0]);
                decares.insert(dims_dec(i.clone()));
                decares.insert(dims_dec(o.clone()));
                // conditions.push(ass_eq(first(Z3Exp::Variable(i.clone())), first(Z3Exp::Variable(o.clone()))));
                // conditions.push(ass_eq(second(Z3Exp::Variable(i.clone())), second(Z3Exp::Variable(o))));
                // conditions.push(ass_eq(third(Z3Exp::Variable(i.clone())), int(1)));
                // conditions.push(ass_eq(forth(Z3Exp::Variable(i)), int(1)));
            } else if op_type == "Relu"
                || op_type == "Dropout"
                || op_type == "Clip"
                || op_type == "LeakyRelu"
                || op_type == "Cast"
            {
                assert_eq!(node.input.len(), 1);
                assert_eq!(node.input.len(), node.output.len());

                let i = shape_name(&node.input[0]);
                let o = shape_name(&node.output[0]);
                decares.insert(dims_dec(i.clone()));
                decares.insert(dims_dec(o.clone()));
                conditions.push(Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Variable(i)),
                    Box::new(Z3Exp::Variable(o)),
                ))));
            } else if op_type == "Add" || op_type == "Mul" || op_type == "Div" {
                assert_eq!(node.input.len(), 2);
                assert_eq!(node.output.len(), 1);

                let i1 = shape_name(&node.input[0]);
                let i2 = shape_name(&node.input[1]);
                let o = shape_name(&node.output[0]);
                decares.insert(dims_dec(i1.clone()));
                decares.insert(dims_dec(i2.clone()));
                decares.insert(dims_dec(o.clone()));

                conditions.push(Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Variable(i1.clone())),
                    Box::new(Z3Exp::Variable(i2)),
                ))));
                conditions.push(Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Variable(i1)),
                    Box::new(Z3Exp::Variable(o)),
                ))));
            } else if op_type == "Concat" {
                assert!(1 <= node.input.len(), "{:?}", node);
                assert_eq!(node.output.len(), 1);
                assert_eq!(node.attribute.len(), 1);
                let axis = get_attribute(node, "axis").unwrap().i.unwrap();

                let mut inputs = Vec::new();
                let mut in_exps = Vec::new();
                for i in node.input.iter() {
                    inputs.push(shape_name(i));
                    decares.insert(dims_dec(shape_name(i)));
                    in_exps.push(Z3Exp::Variable(shape_name(i)));
                }

                let o = shape_name(&node.output[0]);
                decares.insert(Z3Exp::DecareConst(
                    o.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));
                let mut oexp = Z3Exp::Variable(o);

                for _i in 0..axis {
                    let oh = first(oexp.clone());
                    for i in 0..in_exps.len() {
                        let h = first(in_exps[i].clone());
                        conditions.push(ass_eq(oh, h));
                        in_exps[i] = tail(in_exps[i]);
                    }

                    oexp = tail(oexp);
                }

                let mut in_concat = int(0);
                for i in in_exps.iter() {
                    in_concat = plus(in_concat, *i);
                }
                conditions.push(ass_eq(in_concat, head(oexp.clone())));

                for i in in_exps.iter() {
                    conditions.push(ass_eq(tail(*i), tail(oexp)));
                }
            } else if op_type == "BatchNormalization" {
                assert_eq!(node.input.len(), 5);
                assert_eq!(node.output.len(), 1);

                let x = shape_name(&node.input[0]);
                let scale = shape_name(&node.input[1]);
                let b = shape_name(&node.input[2]);
                let mean = shape_name(&node.input[3]);
                let var = shape_name(&node.input[4]);
                let o = shape_name(&node.output[0]);

                decares.insert(dims_dec(x.clone()));
                decares.insert(dims_dec(scale.clone()));
                decares.insert(dims_dec(b.clone()));
                decares.insert(dims_dec(mean.clone()));
                decares.insert(dims_dec(var.clone()));
                decares.insert(dims_dec(o.clone()));

                let x_exp = Z3Exp::Variable(x);
                let scale_exp = Z3Exp::Variable(scale);
                let b_exp = Z3Exp::Variable(b);
                let mean_exp = Z3Exp::Variable(mean);
                let var_exp = Z3Exp::Variable(var);
                let o_exp = Z3Exp::Variable(o);

                conditions.push(ass_eq(x_exp.clone(), o_exp.clone()));
                conditions.push(ass_eq(second(x_exp.clone()), first(scale_exp)));
                conditions.push(ass_eq(second(x_exp.clone()), first(b_exp)));
                conditions.push(ass_eq(second(x_exp.clone()), first(mean_exp)));
                conditions.push(ass_eq(second(x_exp), first(var_exp)));
            } else if op_type == "Flatten" {
                // TODO (akawashiro): We need fold(mul, shape).
            } else {
                unreachable!("Unknown op {:?}", node);
            }
        }
    }

    (decares, conditions)
}

fn shape_infer(onnx_path: &Path) -> Option<Z3Result> {
    let file = File::open(onnx_path).expect("fail to open file");
    let mut buffered_reader = BufReader::new(file);
    let mut cis = CodedInputStream::from_buf_read(&mut buffered_reader);

    let mut model = ModelProto::new();
    model.merge_from(&mut cis).expect("fail to merge");

    let (decares, conditions) = gen_constraints(&model);

    let smt_filename = onnx_path.to_str().unwrap().to_owned() + "_shape_inference.smtlib2";
    let mut smt_file = File::create(smt_filename.clone()).unwrap();
    let mut contents = String::from("");
    for d in decares.iter() {
        contents += &format!("{:}\n", d);
    }
    for c in conditions.iter() {
        contents += &format!("{:}\n", c);
    }
    contents += &format!("{:}\n", Z3Exp::CheckSat);
    contents += &format!("{:}\n", Z3Exp::GetModel);
    smt_file.write_all(contents.as_bytes()).unwrap();

    let output = Command::new("z3")
        .arg("-smt2")
        .arg(smt_filename)
        .output()
        .unwrap_or_else(|e| panic!("failed to execute process: {}", e));

    let result = String::from_utf8_lossy(&output.stdout);
    if let Ok((remain, parsed)) = parse_z3_result(&result) {
        assert_eq!(remain, "");

        for (k, v) in parsed.shapes.iter() {
            println!("{:}: {:?}", k, v);
        }
        let result_filename =
            onnx_path.to_str().unwrap().to_owned() + "_shape_inference_result.smtlib2";
        let mut result_file = File::create(&result_filename).unwrap();
        result_file.write_all(result.as_bytes()).unwrap();
        println!("Check: {:}", result_filename);

        Some(parsed)
    } else {
        println!("Failed to parse the result {:?}", parse_z3_result(&result));
        None
    }
}

struct Testcase<'a> {
    file: &'a Path,
    url: &'a str,
    ass: Vec<(&'a str, Vec<i64>)>,
}

fn shape_partial_eq(s1: &Vec<i64>, s2: &Vec<i64>) -> bool {
    for (d1, d2) in s1.iter().zip(s2.iter()) {
        if d1 != d2 {
            return false;
        }
    }
    true
}

#[test]
fn e2e_test() {
    let mut testcases = Vec::new();
    testcases.push(Testcase{
        file: Path::new("MaskRCNN-10.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx",
        ass:vec![]
    });
    testcases.push(Testcase{
        file: Path::new("tinyyolov2-7.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx",
        ass:vec![]
    });
    testcases.push(Testcase{
        file: Path::new("resnet50-v1-7.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx",
        ass:vec![]
    });
    testcases.push(Testcase{
        file: Path::new("squeezenet1.1-7.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx", 
        ass:vec![
            ("shape_squeezenet0_conv24_fwd", vec![1, 256, 13, 13]),
            ("shape_squeezenet0_relu24_fwd", vec![1, 256, 13, 13]),
            ("shape_squeezenet0_concat7", vec![1, 512, 13, 13]),
            ("shape_squeezenet0_dropout0_fwd", vec![1, 512, 13, 13]),
            ("shape_squeezenet0_conv25_fwd", vec![1, 1000, 13, 13]), 
        ]});
    testcases.push(Testcase{
        file: Path::new("mobilenetv2-7.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        ass:vec![
            ("shape_477", vec![0,32,112,112]),
            ("shape_474", vec![0,32,112,112]),
            ("shape_317", vec![0,32,112,112])
        ]});

    for t in testcases.iter() {
        let retry_z3 = 20;

        let d1: Vec<i64> = Vec::new();
        let d2: Vec<i64> = Vec::new();
        let mut failed_shapes = vec![(d1, d2, "dummy")];

        // Hmm... The output of Z3 is not decidable. We need some retries to get the answer.
        for _i in 0..retry_z3 {
            if !t.file.exists() {
                println!("Download {} from github", t.file.to_string_lossy());
                let responce = reqwest::blocking::get(&*t.url)
                    .expect(&(String::from("Failed to download from ") + t.url));
                let contents = responce.bytes().expect("No contents in response");
                let mut out = File::create(t.file).expect("failed to create file");
                out.write_all(&contents)
                    .expect("Failed to write contents to the file");
            }
            if let Some(result) = shape_infer(t.file) {
                let mut f: Vec<(Vec<i64>, Vec<i64>, &str)> = Vec::new();
                for (k, s1) in t.ass.iter() {
                    let s2 = result.shapes.get(&String::from(*k)).expect(k);
                    if s1 != s2 {
                        f.push((s1.clone(), s2.clone(), *k));
                    }
                }
                failed_shapes = f.clone();
                if failed_shapes.len() == 0 {
                    break;
                }
            } else {
                unreachable!();
            }
        }

        assert_eq!(failed_shapes.len(), 0, "{:?}", failed_shapes);
    }
}

fn main() {
    assert_eq!(std::env::args().len(), 2);
    let arg1 = std::env::args().nth(1).unwrap();
    let onnx_path = Path::new(&arg1);

    shape_infer(onnx_path);
}
