use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0},
    combinator::{cut, map, map_res, recognize},
    error::{context, VerboseError},
    multi::{fold_many0, many0_count},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};
use std::fmt;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Z3Type {
    Int,
    Void,
    List(Box<Z3Type>),
}

impl fmt::Display for Z3Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Z3Type::Int => write!(f, "Int"),
            Z3Type::Void => write!(f, "()"),
            Z3Type::List(ty) => write!(f, "(List {:})", ty),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Z3Exp {
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
    Nil,
    Insert(Box<Z3Exp>, Box<Z3Exp>),
    DefineFun(String, Box<Z3Type>, Box<Z3Type>, Box<Z3Exp>),
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Z3Result {
    is_sat: bool,
    shapes: Vec<Z3Exp>,
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
            Z3Exp::Nil => write!(f, "nil"),
            Z3Exp::Insert(exp1, exp2) => write!(f, "(insert {:} {:})", exp1, exp2),
            Z3Exp::DefineFun(n, t1, t2, e) => {
                write!(f, "(define-fun {:} {:} {:} {:})", n, t1, t2, e)
            }
        }
    }
}

pub fn dims_dec(s: String) -> Z3Exp {
    Z3Exp::DecareConst(s, Z3Type::List(Box::new(Z3Type::Int)))
}

pub fn ass_eq(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Assert(Box::new(Z3Exp::Equal(Box::new(e1), Box::new(e2))))
}

pub fn head(e: Z3Exp) -> Z3Exp {
    Z3Exp::Head(Box::new(e))
}

pub fn tail(e: Z3Exp) -> Z3Exp {
    Z3Exp::Tail(Box::new(e))
}

pub fn plus(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Plus(Box::new(e1), Box::new(e2))
}

pub fn mul(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Mul(Box::new(e1), Box::new(e2))
}

pub fn sub(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Sub(Box::new(e1), Box::new(e2))
}

pub fn div(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Div(Box::new(e1), Box::new(e2))
}

pub fn int(i: i64) -> Z3Exp {
    Z3Exp::Int(i)
}

pub fn insert(e1: Z3Exp, e2: Z3Exp) -> Z3Exp {
    Z3Exp::Insert(Box::new(e1), Box::new(e2))
}

#[test]
fn diplay_test() {
    assert_eq!("(* 10 42)", format!("{}", mul(int(10), int(42))));
}

fn parse_num<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    alt((
        map_res(digit1, |digit_str: &str| digit_str.parse::<i64>().map(int)),
        map(preceded(tag("-"), digit1), |digit_str: &str| {
            int(-1 * digit_str.parse::<i64>().unwrap())
        }),
    ))(i)
}

fn identifier<'a>(input: &'a str) -> IResult<&str, &str, VerboseError<&'a str>> {
    recognize(pair(alpha1, many0_count(alt((alphanumeric1, tag("_"))))))(input)
}

fn parse_variable<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    map(identifier, |s: &str| Z3Exp::Variable(String::from(s)))(i)
}

fn parse_nil<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    map(tag("nil"), |_| Z3Exp::Nil)(i)
}

fn parse_insert<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(
                terminated(tag("insert"), multispace0),
                tuple((parse_expr, parse_expr)),
            ),
            |(e1, e2)| insert(e1, e2),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_plus<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(
                terminated(tag("+"), multispace0),
                tuple((parse_expr, parse_expr)),
            ),
            |(e1, e2)| plus(e1, e2),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_mul<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(
                terminated(tag("*"), multispace0),
                tuple((parse_expr, parse_expr)),
            ),
            |(e1, e2)| mul(e1, e2),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_div<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(
                terminated(tag("div"), multispace0),
                tuple((parse_expr, parse_expr)),
            ),
            |(e1, e2)| div(e1, e2),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_sub<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(
                terminated(tag("-"), multispace0),
                tuple((parse_expr, parse_expr)),
            ),
            |(e1, e2)| sub(e1, e2),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_equal<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(
                terminated(tag("="), multispace0),
                tuple((parse_expr, parse_expr)),
            ),
            |(e1, e2)| Z3Exp::Equal(Box::new(e1), Box::new(e2)),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_head<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(terminated(tag("head"), multispace0), parse_expr),
            |e| head(e),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_tail<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(terminated(tag("tail"), multispace0), parse_expr),
            |e| tail(e),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_assert<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(terminated(tag("assert"), multispace0), parse_expr),
            |e| Z3Exp::Assert(Box::new(e)),
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_define_fun<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(
            preceded(
                terminated(tag("define-fun"), multispace0),
                tuple((identifier, parse_type, parse_type, parse_expr)),
            ),
            |(id, t1, t2, e)| {
                Z3Exp::DefineFun(String::from(id), Box::new(t1), Box::new(t2), Box::new(e))
            },
        ),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_check_sat<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(preceded(tag("check-sat"), multispace0), |_| Z3Exp::CheckSat),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_get_model<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    delimited(
        char('('),
        map(preceded(tag("get-model"), multispace0), |_| Z3Exp::GetModel),
        context("closing paren", cut(preceded(multispace0, char(')')))),
    )(i)
}

fn parse_expr<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    preceded(
        multispace0,
        alt((
            parse_num,
            parse_nil,
            parse_insert,
            parse_head,
            parse_tail,
            parse_assert,
            parse_sub,
            parse_plus,
            parse_mul,
            parse_div,
            parse_equal,
            parse_check_sat,
            parse_get_model,
            parse_variable,
            parse_define_fun,
        )),
    )(i)
}

#[test]
fn test_parse_num() {
    assert_eq!(parse_num("42"), Ok(("", int(42))));
}

#[test]
fn test_parse_expr() {
    let mut testcases = Vec::new();
    testcases.push(int(42));
    testcases.push(Z3Exp::Nil);
    testcases.push(insert(int(1), Z3Exp::Nil));
    testcases.push(head(insert(int(1), Z3Exp::Nil)));
    testcases.push(tail(insert(int(1), Z3Exp::Nil)));
    testcases.push(Z3Exp::Assert(Box::new(int(42))));
    testcases.push(ass_eq(int(10), int(10)));
    testcases.push(plus(int(4), int(3)));
    testcases.push(mul(int(4), int(3)));
    testcases.push(sub(int(4), int(3)));
    testcases.push(div(int(4), int(3)));
    testcases.push(Z3Exp::CheckSat);
    testcases.push(Z3Exp::GetModel);
    testcases.push(Z3Exp::Variable(String::from("hoge")));
    testcases.push(Z3Exp::Variable(String::from("hoge_fuga")));
    testcases.push(Z3Exp::Variable(String::from("h123")));
    testcases.push(Z3Exp::Variable(String::from("h_123")));
    for e in testcases.iter() {
        let s = format!("{:}", e);
        assert_eq!(parse_expr(&s), Ok(("", e.clone())));
    }

    // Many spaces test
    assert_eq!(
        parse_expr("(insert   1    nil)"),
        Ok(("", insert(int(1), Z3Exp::Nil)))
    );

    assert_eq!(parse_expr("(define-fun squeezenet0_conv8_weight_shape () (List Int) (insert 128 (insert 32 (insert 1 (insert 1 nil)))))"), Ok(("", Z3Exp::DefineFun(String::from("squeezenet0_conv8_weight_shape"), Box::new(Z3Type::Void), Box::new(Z3Type::List(Box::new(Z3Type::Int))), Box::new(insert(int(128), insert(int(32), insert(int(1), insert(int(1), Z3Exp::Nil)))))))));
}

fn parse_primitive_type<'a>(i: &'a str) -> IResult<&'a str, Z3Type, VerboseError<&'a str>> {
    alt((
        map(tag("Int"), |_| Z3Type::Int),
        map(tag("()"), |_| Z3Type::Void),
    ))(i)
}

fn parse_type<'a>(i: &'a str) -> IResult<&'a str, Z3Type, VerboseError<&'a str>> {
    preceded(
        multispace0,
        alt((
            parse_primitive_type,
            delimited(
                char('('),
                map(
                    preceded(terminated(tag("List"), multispace0), parse_type),
                    |elem| Z3Type::List(Box::new(elem)),
                ),
                context("closing paren", cut(preceded(multispace0, char(')')))),
            ),
        )),
    )(i)
}

#[test]
fn test_parse_type() {
    assert_eq!(parse_type("Int"), Ok(("", Z3Type::Int)));
    assert_eq!(parse_type("()"), Ok(("", Z3Type::Void)));
    assert_eq!(
        parse_type("(List Int)"),
        Ok(("", Z3Type::List(Box::new(Z3Type::Int))))
    );
    assert_eq!(
        parse_type("(List       Int)"),
        Ok(("", Z3Type::List(Box::new(Z3Type::Int))))
    );
}

pub fn parse_z3_result<'a>(i: &'a str) -> IResult<&'a str, Z3Result, VerboseError<&'a str>> {
    preceded(
        terminated(tag("sat"), multispace0),
        delimited(
            char('('),
            map(
                fold_many0(parse_expr, Vec::new, |mut acc: Vec<Z3Exp>, item| {
                    acc.push(item);
                    acc
                }),
                |ds| Z3Result {
                    is_sat: true,
                    shapes: ds,
                },
            ),
            context("closing paren", cut(preceded(multispace0, char(')')))),
        ),
    )(i)
}

#[test]
fn test_parse_z3_result() {
    let input = r##"sat
(
  (define-fun squeezenet0_conv8_weight_shape () (List Int)
    (insert 128 (insert 32 (insert 1 nil))))
  (define-fun squeezenet0_dropout0_fwd_shape () (List Int)
    (insert 1 (insert 512 (insert 13 (insert 13 nil)))))
)"##;
    let result = parse_z3_result(input);
    if let Ok((remainder, parsed)) = result {
        assert_eq!(parsed.is_sat, true);
        assert_eq!(parsed.shapes.len(), 2);
        assert_eq!(remainder, "");
    } else {
        unreachable!("Failed to parse");
    }
}
