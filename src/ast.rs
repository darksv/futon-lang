use crate::lexer::TokenType;
use crate::parser::TyS;

pub(crate) type Ty<'tcx> = &'tcx TyS<'tcx>;

#[derive(Debug)]
pub struct Argument<'tcx> {
    pub name: String,
    pub ty: Ty<'tcx>,
}

#[derive(Debug)]
pub struct Field<'tcx> {
    pub name: String,
    pub ty: Ty<'tcx>,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Identifier(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Prefix(TokenType, Box<Expression>),
    Infix(TokenType, Box<Expression>, Box<Expression>),
    Place(Box<Expression>, Box<Expression>),
    Array(Vec<Expression>),
    Tuple(Vec<Expression>),
    Call(Box<Expression>, Vec<Expression>),
}

#[derive(Debug)]
pub enum Item<'tcx> {
    Let {
        name: String,
        ty: Option<Ty<'tcx>>,
        expr: Option<Expression>,
    },
    Assignment {
        lhs: Expression,
        operator: Option<char>,
        expr: Expression,
    },
    Expr {
        expr: Expression,
    },
    Function {
        name: String,
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
