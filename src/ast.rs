use crate::index_arena::{Handle, Many};
use crate::ir::Var;
use crate::lexer::SourceSpan;
use crate::types::TypeRef;

#[derive(Debug, Clone)]
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
    And,
    Or,
    Negate,
    Ref,
    Deref,
    As,
}

#[derive(Debug)]
pub(crate) enum Expr {
    Identifier(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Prefix(Operator, Handle<Expr>),
    Infix(Operator, Handle<Expr>, Handle<Expr>),
    Place(Handle<Expr>, Handle<Expr>),
    Array(Handle<Expr, Many>),
    Tuple(Handle<Expr, Many>),
    Call(Handle<Expr>, Handle<Expr, Many>),
    Range(Handle<Expr>, Option<Handle<Expr>>),
    Index(Handle<Expr>, Handle<Expr>),
    Cast(Handle<Expr>, Type),
    Var(Var),
    StructLiteral(Option<String>, Vec<(String, Handle<Expr>)>),
}

impl Expr {
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Expr::Identifier(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum Item {
    Let {
        name: String,
        r#type: Option<Type>,
        expr: Option<Handle<Expr>>,
    },
    Assignment {
        lhs: Handle<Expr>,
        operator: Option<Operator>,
        expr: Handle<Expr>,
    },
    Expr {
        expr: Handle<Expr>,
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
        condition: Handle<Expr>,
        arm_true: Vec<Item>,
        arm_false: Option<Vec<Item>>,
    },
    ForIn {
        name: String,
        expr: Handle<Expr>,
        body: Vec<Item>,
    },
    Loop {
        body: Vec<Item>,
    },
    Break,
    Yield(Box<Handle<Expr>>),
    Return(Box<Handle<Expr>>),
    Block(Vec<Item>),
    Assert(Box<Handle<Expr>>),
}
