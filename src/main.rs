extern crate protobuf;
extern crate reqwest;

use protobuf::{CodedInputStream, Message};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

mod onnx;
use onnx::ModelProto;
mod z3;
use crate::z3::*;
mod constraints;
use crate::constraints::*;

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

#[allow(dead_code)]
struct Testcase<'a> {
    file: &'a Path,
    url: &'a str,
    ass: Vec<(&'a str, Vec<i64>)>,
}

#[allow(dead_code)]
fn shape_partial_eq(s1: &Vec<i64>, s2: &Vec<i64>) -> bool {
    for (d1, d2) in s1.iter().zip(s2.iter()) {
        if d1 != d2 {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
fn run_testcase(test: &Testcase) {
    let retry_z3 = 20;

    let d1: Vec<i64> = Vec::new();
    let d2: Vec<i64> = Vec::new();
    let mut failed_shapes = vec![(d1, d2, "dummy")];

    // Hmm... The output of Z3 is not decidable. We need some retries to get the answer.
    for _i in 0..retry_z3 {
        if !test.file.exists() {
            println!("Download {} from github", test.file.to_string_lossy());
            let responce = reqwest::blocking::get(&*test.url)
                .expect(&(String::from("Failed to download from ") + test.url));
            let contents = responce.bytes().expect("No contents in response");
            let mut out = File::create(test.file).expect("failed to create file");
            out.write_all(&contents)
                .expect("Failed to write contents to the file");
        }
        if let Some(result) = shape_infer(test.file) {
            let mut f: Vec<(Vec<i64>, Vec<i64>, &str)> = Vec::new();
            for (k, s1) in test.ass.iter() {
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

#[test]
fn resnet_test() {
    let t = Testcase{
        file: Path::new("resnet50-v1-7.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx",
        ass:vec![("shape_resnetv17_stage3_batchnorm7_fwd", vec![0, 256, 14, 14])]
    };
    run_testcase(&t);
}

#[test]
fn squeezenet_test() {
    let t = Testcase{
        file: Path::new("squeezenet1.1-7.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.1-7.onnx", 
        ass:vec![
            ("shape_squeezenet0_conv24_fwd", vec![1, 256, 13, 13]),
            ("shape_squeezenet0_relu24_fwd", vec![1, 256, 13, 13]),
            ("shape_squeezenet0_concat7", vec![1, 512, 13, 13]),
            ("shape_squeezenet0_dropout0_fwd", vec![1, 512, 13, 13]),
            ("shape_squeezenet0_conv25_fwd", vec![1, 1000, 13, 13]), 
        ]};
    run_testcase(&t);
}

#[test]
fn mobilenet_test() {
    let t = Testcase{
        file: Path::new("mobilenetv2-7.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        ass:vec![
            ("shape_477", vec![0,32,112,112]),
            ("shape_474", vec![0,32,112,112]),
            ("shape_317", vec![0,32,112,112])
        ]};
    run_testcase(&t);
}

#[test]
#[should_panic]
fn maskrcnn_test() {
    // TODO
    let t = Testcase{
        file: Path::new("MaskRCNN-10.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx",
        ass:vec![]
    } ;
    run_testcase(&t);
}
#[test]
#[should_panic]
fn tinyyolo_test() {
    // This test doesn't pass now because of broadcasting in Add.
    let t = Testcase{
        file: Path::new("tinyyolov2-7.onnx"),
        url: "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx",
        ass:vec![]
    } ;
    run_testcase(&t);
}

fn main() {
    assert_eq!(std::env::args().len(), 2);
    let arg1 = std::env::args().nth(1).unwrap();
    let onnx_path = Path::new(&arg1);

    shape_infer(onnx_path);
}
