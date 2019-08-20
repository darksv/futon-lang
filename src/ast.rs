use lexer::TokenType;
use parser::Ty;

#[derive(Debug)]
pub struct Argument {
    pub name: String,
    pub ty: Ty,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Identifier(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Prefix(TokenType, Box<Expression>),
    Infix(TokenType, Box<Expression>, Box<Expression>),
    Array(Vec<Expression>),
    Tuple(Vec<Expression>),
    Call(Box<Expression>, Vec<Expression>),
}

#[derive(Debug)]
pub enum Item {
    Let {
        name: String,
        ty: Option<Ty>,
        expr: Option<Expression>,
    },
    Assignment {
        lhs: Expression,
        operator: Option<char>,
        expr: Expression,
    },
    Function {
        name: String,
        args: Vec<Argument>,
        ty: Option<Ty>,
        body: Vec<Item>,
    },
    If {
        condition: Expression,
        arm_true: Vec<Item>,
        arm_false: Option<Vec<Item>>,
    },
    ForIn {
        name: String,
        expr: Expression,
        body: Vec<Item>,
    },
    Break,
    Yield(Box<Expression>),
    Return(Box<Expression>),
}
