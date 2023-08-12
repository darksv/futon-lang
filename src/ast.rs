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
pub(crate) enum Expression<'a> {
    Identifier(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Prefix(Operator, &'a Expression<'a>),
    Infix(Operator, &'a Expression<'a>, &'a Expression<'a>),
    Place(&'a Expression<'a>, &'a Expression<'a>),
    Array(&'a [Expression<'a>]),
    Tuple(&'a [Expression<'a>]),
    Call(&'a Expression<'a>, &'a [Expression<'a>]),
    Range(&'a Expression<'a>, Option<&'a Expression<'a>>),
    Index(&'a Expression<'a>, &'a Expression<'a>),
    Cast(&'a Expression<'a>, Type),
    Var(Var),
    StructLiteral(Option<String>, Vec<(String, &'a Expression<'a>)>),
}

impl Expression<'_> {
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Expression::Identifier(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum Item<'a> {
    Let {
        name: String,
        r#type: Option<Type>,
        expr: Option<&'a Expression<'a>>,
    },
    Assignment {
        lhs: &'a Expression<'a>,
        operator: Option<Operator>,
        expr: &'a Expression<'a>,
    },
    Expr {
        expr: &'a Expression<'a>,
    },
    Function {
        name: String,
        is_extern: bool,
        params: Vec<Argument>,
        ty: Type,
        body: Vec<Item<'a>>,
    },
    Struct {
        name: String,
        fields: Vec<Field>,
    },
    If {
        condition: &'a Expression<'a>,
        arm_true: Vec<Item<'a>>,
        arm_false: Option<Vec<Item<'a>>>,
    },
    ForIn {
        name: String,
        expr: &'a Expression<'a>,
        body: Vec<Item<'a>>,
    },
    Loop {
        body: Vec<Item<'a>>,
    },
    Break,
    Yield(Box<&'a Expression<'a>>),
    Return(Box<&'a Expression<'a>>),
    Block(Vec<Item<'a>>),
    Assert(Box<&'a Expression<'a>>),
}
