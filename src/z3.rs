use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, char, digit1, multispace0, multispace1, one_of},
    combinator::{cut, map, map_res, opt},
    error::{context, VerboseError},
    multi::many0,
    sequence::{delimited, preceded, terminated, tuple},
    IResult, Parser,
};
use std::fmt;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub enum Z3Type {
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

fn parse_expr<'a>(i: &'a str) -> IResult<&'a str, Z3Exp, VerboseError<&'a str>> {
    alt((parse_num,))(i)
}

#[test]
fn test_parse_num() {
    assert_eq!(parse_num("42"), Ok(("", int(42))));
}

#[test]
fn test_parse_expr() {
    assert_eq!(parse_num("42"), Ok(("", int(42))));
}


fn parse_primitive_type<'a>(i: &'a str) -> IResult<&'a str, Z3Type, VerboseError<&'a str>> {
    map(tag("Int"), |_| Z3Type::Int)(i)
}

fn parse_type<'a>(i: &'a str) -> IResult<&'a str, Z3Type, VerboseError<&'a str>> {
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
    ))(i)
}

#[test]
fn test_parse_type() {
    assert_eq!(parse_type("Int"), Ok(("", Z3Type::Int)));
    assert_eq!(
        parse_type("(List Int)"),
        Ok(("", Z3Type::List(Box::new(Z3Type::Int))))
    );
    assert_eq!(
        parse_type("(List       Int)"),
        Ok(("", Z3Type::List(Box::new(Z3Type::Int))))
    );
}
