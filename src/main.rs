extern crate protobuf;
extern crate reqwest;

use protobuf::{CodedInputStream, Message};
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

mod onnx;
use onnx::ModelProto;

#[derive(PartialEq, Eq, Hash, Clone)]
enum Z3Type {
    Int,
    List(Box<Z3Type>),
}

impl fmt::Display for Z3Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Z3Type::Int => write!(f, "Int"),
            Z3Type::List(ty) => write!(f, "(List {:})", ty),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
enum Z3Exp {
    DecareConst(String, Z3Type),
    Assert(Box<Z3Exp>),
    Equal(Box<Z3Exp>, Box<Z3Exp>),
    CheckSat,
    GetModel,
    Variable(String),
    Head(Box<Z3Exp>),
    Tail(Box<Z3Exp>),
    Plus(Box<Z3Exp>, Box<Z3Exp>),
    Mul(Box<Z3Exp>, Box<Z3Exp>),
    Sub(Box<Z3Exp>, Box<Z3Exp>),
    Div(Box<Z3Exp>, Box<Z3Exp>),
    Int(i64),
}

impl fmt::Display for Z3Exp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Z3Exp::DecareConst(val, ty) => write!(f, "(declare-const {:} {:})", val, ty),
            Z3Exp::Assert(exp) => write!(f, "(assert {:})", exp),
            Z3Exp::Equal(exp1, exp2) => write!(f, "(= {:} {:})", exp1, exp2),
            Z3Exp::CheckSat => write!(f, "(check-sat)"),
            Z3Exp::GetModel => write!(f, "(get-model)"),
            Z3Exp::Variable(var) => write!(f, "{:}", var),
            Z3Exp::Head(exp) => write!(f, "(head {:})", exp),
            Z3Exp::Tail(exp) => write!(f, "(tail {:})", exp),
            Z3Exp::Plus(exp1, exp2) => write!(f, "(+ {:} {:})", exp1, exp2),
            Z3Exp::Mul(exp1, exp2) => write!(f, "(* {:} {:})", exp1, exp2),
            Z3Exp::Sub(exp1, exp2) => write!(f, "(- {:} {:})", exp1, exp2),
            Z3Exp::Div(exp1, exp2) => write!(f, "(div {:} {:})", exp1, exp2),
            Z3Exp::Int(i) => write!(f, "{:}", i),
        }
    }
}

fn dims_dec(s: String) -> Z3Exp {
    Z3Exp::DecareConst(s, Z3Type::List(Box::new(Z3Type::Int)))
}

fn ass_eq(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Assert(Box::new(Z3Exp::Equal(Box::new(e1), Box::new(e2))))
}

fn head(e: Z3Exp) -> Z3Exp {
    Z3Exp::Head(Box::new(e))
}

fn tail(e: Z3Exp) -> Z3Exp {
    Z3Exp::Tail(Box::new(e))
}

fn plus(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Plus(Box::new(e1), Box::new(e2))
}

fn mul(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Mul(Box::new(e1), Box::new(e2))
}

fn sub(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Sub(Box::new(e1), Box::new(e2))
}

fn div(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Div(Box::new(e1), Box::new(e2))
}

fn int(i: i64) -> Z3Exp {
    Z3Exp::Int(i)
}

#[test]
fn diplay_test() {
    assert_eq!("(* 10 42)", format!("{}", mul(int(10), int(42))));
}

fn gen_constraints(model: &onnx::ModelProto) -> (HashSet<Z3Exp>, Vec<Z3Exp>) {
    let mut decares = HashSet::new();
    let mut conditions = Vec::new();

    for inout in model.graph.input.iter().chain(model.graph.output.iter()) {
        let name = inout.name.as_ref().unwrap().clone() + "_shape";
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

    for node in model.graph.node.iter() {
        if let Some(op_type) = &node.op_type {
            if op_type == "Reshape" {
                // TODO (akawashiro): We need constant propagation.
            } else if op_type == "MaxPool" || op_type == "AveragePool" {
                assert_eq!(node.input.len(), 1);
                assert_eq!(node.output.len(), 1);

                let kernel_shape_att = &node.attribute[0];
                assert_eq!(kernel_shape_att.name, Some(String::from("kernel_shape")));
                let kernel_shape = &kernel_shape_att.ints;
                assert_eq!(kernel_shape.len(), 2);

                let pads_att = &node.attribute[1];
                assert_eq!(pads_att.name, Some(String::from("pads")));
                let pads = &pads_att.ints;
                assert_eq!(pads.len(), 4);

                let strides_att = &node.attribute[2];
                assert_eq!(strides_att.name, Some(String::from("strides")));
                let strides = &strides_att.ints;
                assert_eq!(strides.len(), 2);

                decares.insert(dims_dec(node.input[0].clone() + "_shape"));
                decares.insert(dims_dec(node.output[0].clone() + "_shape"));

                let in_image = Z3Exp::Variable(node.input[0].clone() + "_shape");
                let out_image = Z3Exp::Variable(node.output[0].clone() + "_shape");

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
                assert_eq!(group, 1);

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

                decares.insert(dims_dec(node.input[0].clone() + "_shape"));
                decares.insert(dims_dec(node.input[1].clone() + "_shape"));
                decares.insert(dims_dec(node.input[2].clone() + "_shape"));
                decares.insert(dims_dec(node.output[0].clone() + "_shape"));

                let in_image = Z3Exp::Variable(node.input[0].clone() + "_shape");
                let weight = Z3Exp::Variable(node.input[1].clone() + "_shape");
                let bias = Z3Exp::Variable(node.input[2].clone() + "_shape");
                let out_image = Z3Exp::Variable(node.output[0].clone() + "_shape");

                let in_batch = head(in_image.clone());
                let out_batch = head(out_image.clone());
                conditions.push(ass_eq(in_batch, out_batch));

                let in_ch_eq = ass_eq(head(tail(in_image.clone())), head(tail(weight.clone())));
                conditions.push(in_ch_eq);

                let out_ch_eq1 = ass_eq(head(weight.clone()), head(tail(out_image.clone())));
                let out_ch_eq2 = ass_eq(head(weight), head(bias));
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
            } else if op_type == "Relu" || op_type == "Dropout" {
                assert_eq!(node.input.len(), 1);
                assert_eq!(node.input.len(), node.output.len());

                let i = node.input[0].clone() + "_shape";
                let o = node.output[0].clone() + "_shape";
                decares.insert(dims_dec(i.clone()));
                decares.insert(dims_dec(o.clone()));
                conditions.push(Z3Exp::Assert(Box::new(Z3Exp::Equal(
                    Box::new(Z3Exp::Variable(i)),
                    Box::new(Z3Exp::Variable(o)),
                ))));
            } else if op_type == "Concat" {
                assert_eq!(node.input.len(), 2);
                assert_eq!(node.output.len(), 1);
                assert_eq!(node.attribute.len(), 1);
                let att = &node.attribute[0];
                assert_eq!(att.name, Some(String::from("axis")));
                let axis = att.i.unwrap();

                let i1 = node.input[0].clone() + "_shape";
                let i2 = node.input[0].clone() + "_shape";
                let o = node.output[0].clone() + "_shape";
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
            }
        }
    }

    (decares, conditions)
}

fn shape_infer(onnx_path: &Path) {
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

    if output.status.success() {
        let result = String::from_utf8_lossy(&output.stdout);

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
            out.write_all(&contents).expect("Failed to write contents to the file");
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
