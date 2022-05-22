use crate::lexer::SourceSpan;
use crate::ty::Ty;
use crate::mir::Var;

#[derive(Debug)]
pub(crate) enum Type {
    Name(String),
    Tuple(Vec<Type>),
    Pointer(Box<Type>),
    Array(usize, Box<Type>),
    Slice(Box<Type>),
    Unit,
    Function(Vec<Type>, Box<Type>),
}

#[derive(Debug)]
pub(crate) struct Argument {
    pub name: String,
    pub r#type: Type,
}

#[derive(Debug)]
pub(crate) struct Field {
    pub name: String,
    pub r#type: Type,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Negate,
    Ref,
    Deref,
}

#[derive(Debug, Clone)]
pub(crate) enum Expression {
    Identifier(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Prefix(Operator, Box<Expression>),
    Infix(Operator, Box<Expression>, Box<Expression>),
    Place(Box<Expression>, Box<Expression>),
    Array(Vec<Expression>),
    Tuple(Vec<Expression>),
    Call(Box<Expression>, Vec<Expression>),
    Range(Box<Expression>, Option<Box<Expression>>),
    Index(Box<Expression>, Box<Expression>),
    Var(Var),
}

#[derive(Debug)]
pub(crate) enum Item {
    Let {
        name: String,
        r#type: Option<Type>,
        expr: Option<Expression>,
    },
    Assignment {
        lhs: Expression,
        operator: Option<Operator>,
        expr: Expression,
    },
    Expr {
        expr: Expression,
    },
    Function {
        name: String,
        is_extern: bool,
        params: Vec<Argument>,
        ty: Type,
        body: Vec<Item>,
    },
    Struct {
        name: String,
        fields: Vec<Field>,
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
    Loop {
        body: Vec<Item>,
    },
    Break,
    Yield(Box<Expression>),
    Return(Box<Expression>),
    Block(Vec<Item>),
}
