use crate::ty::Ty;

#[derive(Debug)]
pub(crate) struct Argument<'tcx> {
    pub name: String,
    pub ty: Ty<'tcx>,
}

#[derive(Debug)]
pub(crate) struct Field<'tcx> {
    pub name: String,
    pub ty: Ty<'tcx>,
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
}

#[derive(Debug)]
pub(crate) enum Item<'tcx> {
    Let {
        name: String,
        ty: Option<Ty<'tcx>>,
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
        args: Vec<Argument<'tcx>>,
        ty: Option<Ty<'tcx>>,
        body: Vec<Item<'tcx>>,
    },
    Struct {
        name: String,
        fields: Vec<Field<'tcx>>,
    },
    If {
        condition: Expression,
        arm_true: Vec<Item<'tcx>>,
        arm_false: Option<Vec<Item<'tcx>>>,
    },
    ForIn {
        name: String,
        expr: Expression,
        body: Vec<Item<'tcx>>,
    },
    Loop {
        body: Vec<Item<'tcx>>,
    },
    Break,
    Yield(Box<Expression>),
    Return(Box<Expression>),
}
