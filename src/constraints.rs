use std::collections::HashSet;

use crate::onnx;
use crate::z3::*;

fn shape_name(n: &str) -> String {
    String::from("shape_") + n
}

fn get_attribute<'a>(node: &'a onnx::NodeProto, att: &str) -> Option<&'a onnx::AttributeProto> {
    for a in node.attribute.iter() {
        if a.name == Some(att.to_string()) {
            return Some(a);
        }
    }
    None
}

fn append_gemm(node: &onnx::NodeProto, decares: &mut HashSet<Z3Exp>, conditions: &mut Vec<Z3Exp>) {
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
    let mut dim_k_b = first(mat_b.clone());
    let mut dim_n_b = second(mat_b);
    if trans_b == 1 {
        std::mem::swap(&mut dim_n_b, &mut dim_k_b);
    }
    let dim_m_y = first(mat_y.clone());
    let dim_n_y = second(mat_y);

    conditions.push(ass_eq(dim_m_a, dim_m_y));
    conditions.push(ass_eq(dim_k_a, dim_k_b));
    conditions.push(ass_eq(dim_n_b, dim_n_y));
}

fn append_pool(node: &onnx::NodeProto, decares: &mut HashSet<Z3Exp>, conditions: &mut Vec<Z3Exp>) {
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
}

fn append_conv(node: &onnx::NodeProto, decares: &mut HashSet<Z3Exp>, conditions: &mut Vec<Z3Exp>) {
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
}

fn append_bn(node: &onnx::NodeProto, decares: &mut HashSet<Z3Exp>, conditions: &mut Vec<Z3Exp>) {
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
}

fn append_concat(
    node: &onnx::NodeProto,
    decares: &mut HashSet<Z3Exp>,
    conditions: &mut Vec<Z3Exp>,
) {
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
            conditions.push(ass_eq(oh.clone(), h));
            in_exps[i] = tail(in_exps[i].clone());
        }

        oexp = tail(oexp);
    }

    let mut in_concat = int(0);
    for i in in_exps.iter() {
        in_concat = plus(in_concat, head((*i).clone()));
    }
    conditions.push(ass_eq(in_concat, head(oexp.clone())));

    for i in in_exps.iter() {
        conditions.push(ass_eq(tail((*i).clone()), tail(oexp.clone())));
    }
}

fn append_transpose(
    node: &onnx::NodeProto,
    decares: &mut HashSet<Z3Exp>,
    conditions: &mut Vec<Z3Exp>,
) {
    assert_eq!(node.input.len(), 1, "{:?}", node);
    assert_eq!(node.output.len(), 1, "{:?}", node);

    let data = shape_name(&node.input[0]);
    let transposed = shape_name(&node.output[0]);
    let perm = &get_attribute(node, "perm").unwrap().ints;

    decares.insert(dims_dec(data.clone()));
    decares.insert(dims_dec(transposed.clone()));

    let data_exp = Z3Exp::Variable(data);
    let transposed_exp = Z3Exp::Variable(transposed);

    for (i, p) in perm.iter().enumerate() {
        conditions.push(ass_eq(
            nth(*p, data_exp.clone()),
            nth(i.try_into().unwrap(), transposed_exp.clone()),
        ));
    }
}

pub fn gen_constraints(model: &onnx::ModelProto) -> (HashSet<Z3Exp>, Vec<Z3Exp>) {
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
                } else if let onnx::tensor_shape_proto::dimension::Value::DimParam(_) =
                    d.value.as_ref().unwrap()
                {
                    // TODO: Symbolic values are always 0.
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

    for node in model.graph.node.iter() {
        if let Some(op_type) = &node.op_type {
            if op_type == "Reshape"
                || op_type == "Slice"
                || op_type == "ConstantOfShape"
                || op_type == "NonZero"
                || op_type == "Expand"
                || op_type == "Constant"
                || op_type == "Gather"
                || op_type == "Unsqueeze"
                || op_type == "Resize"
                || op_type == "Shape"
                || op_type == "Flatten"
            {
                // TODO (akawashiro): We need constant propagation.
                // TODO (akawashiro): We need len(list) in Z3.
                // TODO (akawashiro): We need fold(mul, shape).
            } else if op_type == "Gemm" {
                append_gemm(node, &mut decares, &mut conditions);
            } else if op_type == "MaxPool" || op_type == "AveragePool" {
                append_pool(node, &mut decares, &mut conditions);
            } else if op_type == "Conv" {
                append_conv(node, &mut decares, &mut conditions);
            } else if op_type == "BatchNormalization" {
                append_bn(node, &mut decares, &mut conditions);
            } else if op_type == "Concat" {
                append_concat(node, &mut decares, &mut conditions);
            } else if op_type == "Transpose" {
                append_transpose(node, &mut decares, &mut conditions);
            } else if op_type == "GlobalAveragePool" {
                assert_eq!(node.input.len(), 1);
                assert_eq!(node.input.len(), node.output.len());

                let i = shape_name(&node.input[0]);
                let o = shape_name(&node.output[0]);
                decares.insert(dims_dec(i.clone()));
                decares.insert(dims_dec(o.clone()));
            } else if op_type == "Relu"
                || op_type == "Dropout"
                || op_type == "Clip"
                || op_type == "LeakyRelu"
                || op_type == "Cast"
                || op_type == "Sigmoid"
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
            } else {
                unreachable!("Unknown op {:?}", node);
            }
        }
    }

    (decares, conditions)
}
