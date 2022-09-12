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

    fn get_attribute<'a>(node: &'a onnx::NodeProto, att: &str) -> Option<&'a onnx::AttributeProto>{
        for a in node.attribute.iter(){
            if a.name == Some(att.to_string()) {
                return Some(a)
            }
        }
        None
    }

    for node in model.graph.node.iter() {
        if let Some(op_type) = &node.op_type {
            if op_type == "Reshape" {
                // TODO (akawashiro): We need constant propagation.
            }else if op_type == "Shape" {
                // TODO (akawashiro): We need len(list) in Z3.
            }else if op_type == "Constant" || op_type == "Gather" || op_type == "Unsqueeze" {
            }else if op_type == "Gemm" {
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
            }else if op_type == "MaxPool" || op_type == "AveragePool" {
                assert_eq!(node.input.len(), 1);
                assert_eq!(node.output.len(), 1);

                let kernel_shape_att = get_attribute(node, "kernel_shape").unwrap();
                let kernel_shape = &kernel_shape_att.ints;
                assert_eq!(kernel_shape.len(), 2);

                let pads_att = get_attribute(node, "pads").unwrap();
                let pads = &pads_att.ints;
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
                assert_eq!(node.input.len(), 3);
                assert_eq!(node.output.len(), 1);

                let dilations_att = &node.attribute[0];
                assert_eq!(dilations_att.name, Some(String::from("dilations")));
                let dilations = &dilations_att.ints;
                assert_eq!(dilations.len(), 2);

                let group_att = &node.attribute[1];
                assert_eq!(group_att.name, Some(String::from("group")));
                let group = group_att.i.unwrap();
                // assert_eq!(group, 1);

                let kernel_shape_att = &node.attribute[2];
                assert_eq!(kernel_shape_att.name, Some(String::from("kernel_shape")));
                let kernel_shape = &kernel_shape_att.ints;
                assert_eq!(kernel_shape.len(), 2);

                let pads_att = &node.attribute[3];
                assert_eq!(pads_att.name, Some(String::from("pads")));
                let pads = &pads_att.ints;
                assert_eq!(pads.len(), 4);

                let strides_att = &node.attribute[4];
                assert_eq!(strides_att.name, Some(String::from("strides")));
                let strides = &strides_att.ints;
                assert_eq!(strides.len(), 2);

                decares.insert(dims_dec(shape_name(&node.input[0])));
                decares.insert(dims_dec(shape_name(&node.input[1])));
                decares.insert(dims_dec(shape_name(&node.input[2])));
                decares.insert(dims_dec(shape_name(&node.output[0])));

                let in_image = Z3Exp::Variable(shape_name(&node.input[0]));
                let weight = Z3Exp::Variable(shape_name(&node.input[1]));
                let bias = Z3Exp::Variable(shape_name(&node.input[2]));
                let out_image = Z3Exp::Variable(shape_name(&node.output[0]));

                let in_batch = head(in_image.clone());
                let out_batch = head(out_image.clone());
                conditions.push(ass_eq(in_batch, out_batch));

                let in_ch_eq = ass_eq(
                    head(tail(in_image.clone())),
                    mul(int(group), head(tail(weight.clone()))),
                );
                conditions.push(in_ch_eq);

                let out_ch_eq1 = ass_eq(
                    mul(int(group), head(weight.clone())),
                    head(tail(out_image.clone())),
                );
                let out_ch_eq2 = ass_eq(mul(int(group), head(weight)), head(bias));
                conditions.push(out_ch_eq1);
                conditions.push(out_ch_eq2);

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
            } else if op_type == "Relu" || op_type == "Dropout" || op_type == "Clip" {
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
            } else if op_type == "Add" {
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
                assert_eq!(node.input.len(), 2);
                assert_eq!(node.output.len(), 1);
                assert_eq!(node.attribute.len(), 1);
                let att = &node.attribute[0];
                assert_eq!(att.name, Some(String::from("axis")));
                let axis = att.i.unwrap();

                let i1 = shape_name(&node.input[0]);
                let i2 = shape_name(&node.input[0]);
                let o = shape_name(&node.output[0]);
                decares.insert(Z3Exp::DecareConst(
                    i1.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));
                decares.insert(Z3Exp::DecareConst(
                    i2.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));
                decares.insert(Z3Exp::DecareConst(
                    o.clone(),
                    Z3Type::List(Box::new(Z3Type::Int)),
                ));

                let mut i1exp = Z3Exp::Variable(i1);
                let mut i2exp = Z3Exp::Variable(i2);
                let mut oexp = Z3Exp::Variable(o);
                for _i in 0..axis {
                    let i1h = Z3Exp::Head(Box::new(i1exp.clone()));
                    let i2h = Z3Exp::Head(Box::new(i2exp.clone()));
                    let oh = Z3Exp::Head(Box::new(oexp.clone()));

                    let eq1 =
                        Z3Exp::Assert(Box::new(Z3Exp::Equal(Box::new(i1h.clone()), Box::new(i2h))));
                    let eq2 = Z3Exp::Assert(Box::new(Z3Exp::Equal(Box::new(i1h), Box::new(oh))));
                    conditions.push(eq1);
                    conditions.push(eq2);

                    i1exp = Z3Exp::Tail(Box::new(i1exp));
                    i2exp = Z3Exp::Tail(Box::new(i2exp));
                    oexp = Z3Exp::Tail(Box::new(oexp));
                }

                let eq_concat = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Head(Box::new(oexp.clone()))),
                    Box::new(Z3Exp::Plus(
                        Box::new(Z3Exp::Head(Box::new(i1exp.clone()))),
                        Box::new(Z3Exp::Head(Box::new(i2exp.clone()))),
                    )),
                )));
                conditions.push(eq_concat);

                let eq_tail_i = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Tail(Box::new(i1exp.clone()))),
                    Box::new(Z3Exp::Tail(Box::new(i2exp))),
                )));
                let eq_tail_o = Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Tail(Box::new(i1exp))),
                    Box::new(Z3Exp::Tail(Box::new(oexp))),
                )));
                conditions.push(eq_tail_i);
                conditions.push(eq_tail_o);
            } else {
                unreachable!("Unknown op {:?}", op_type);
            }
        }
    }

    (decares, conditions)
}

fn int_list_expr_to_vec(e: Z3Exp) -> Vec<i64> {
    match e {
        Z3Exp::Insert(e1, e2) => {
            if let Z3Exp::Int(i) = *e1 {
                let mut v = int_list_expr_to_vec(*e2);
                v.insert(0, i);
                v
            } else {
                unreachable!("int_list_expr_to_vec finds non integer elements in the list")
            }
        }
        Z3Exp::Nil => Vec::new(),
        _ => unreachable!("int_list_expr_to_vec takes non list input {:}", e),
    }
}

fn print_result(result: Z3Result) -> () {
    for s in result.shapes.iter() {
        if let Z3Exp::DefineFun(name, _, _, e) = s {
            println!("{:}: {:?}", name, int_list_expr_to_vec(*(*e).clone()));
        }
    }
}

fn shape_infer(onnx_path: &Path) {
    let file = File::open(onnx_path).expect("fail to open file");
    let mut buffered_reader = BufReader::new(file);
    let mut cis = CodedInputStream::from_buf_read(&mut buffered_reader);

    let mut model = ModelProto::new();
    model.merge_from(&mut cis).expect("fail to merge");
    println!("hoge");

    let (decares, conditions) = gen_constraints(&model);

    println!("hoge");
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

    dbg!(output.status.success());
    if output.status.success() {
        let result = String::from_utf8_lossy(&output.stdout);
        if let Ok((_, parsed)) = parse_z3_result(&result) {
            print_result(parsed);
        } else {
            unreachable!("Failed to parse the result {:?}", parse_z3_result(&result));
        }

        let result_filename =
            onnx_path.to_str().unwrap().to_owned() + "_shape_inference_result.smtlib2";
        let mut result_file = File::create(&result_filename).unwrap();
        result_file.write_all(result.as_bytes()).unwrap();
        println!("Check: {:}", result_filename);
    } else {
        let s = String::from_utf8_lossy(&output.stderr);
        print!("{}", s);
    }
}

#[test]
fn e2e_test() {
    let mut testcases = Vec::new();
    testcases.push((Path::new("squeezenet1.1-7.onnx"), "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"));
    testcases.push((Path::new("mobilenetv2-7.onnx"), "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"));
    // TODO
    // testcases.push((Path::new("tinyyolov2-7.onnx"), "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx"));
    // testcases.push((Path::new("bidaf-9.onnx"), "https://github.com/onnx/models/raw/main/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx"));
    for (file, url) in testcases.iter() {
        if !file.exists() {
            println!("Download {} from github", file.to_string_lossy());
            let responce = reqwest::blocking::get(*url)
                .expect(&(String::from("Failed to download from ") + url));
            let contents = responce.bytes().expect("No contents in response");
            let mut out = File::create(file).expect("failed to create file");
            out.write_all(&contents)
                .expect("Failed to write contents to the file");
        }
        shape_infer(file);
    }
}

fn main() {
    assert_eq!(std::env::args().len(), 2);
    let arg1 = std::env::args().nth(1).unwrap();
    let onnx_path = Path::new(&arg1);

    shape_infer(onnx_path);
}
