use crate::ty::Ty;
use crate::mir::Var;

#[derive(Debug, Clone)]
pub(crate) struct Argument<'tcx> {
    pub name: String,
    pub ty: Ty<'tcx>,
}

#[derive(Debug, Clone)]
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
    Ref,
    Deref,
}

#[derive(Debug, Clone)]
pub(crate) enum Expr<'tcx> {
    Identifier(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Prefix(Operator, Box<TyExpr<'tcx>>),
    Infix(Operator, Box<TyExpr<'tcx>>, Box<TyExpr<'tcx>>),
    Place(Box<TyExpr<'tcx>>, Box<TyExpr<'tcx>>),
    Array(Vec<TyExpr<'tcx>>),
    Tuple(Vec<TyExpr<'tcx>>),
    Call(Box<TyExpr<'tcx>>, Vec<TyExpr<'tcx>>),
    Range(Box<TyExpr<'tcx>>, Option<Box<TyExpr<'tcx>>>),
    Index(Box<TyExpr<'tcx>>, Box<TyExpr<'tcx>>),
    Var(Var),
}

#[derive(Debug, Clone)]
pub(crate) struct TyExpr<'tcx> {
    pub(crate) ty: Ty<'tcx>,
    pub(crate) expr: Expr<'tcx>,
}

#[derive(Debug, Clone)]
pub(crate) enum Item<'tcx> {
    Let {
        name: String,
        ty: Option<Ty<'tcx>>,
        expr: Option<TyExpr<'tcx>>,
    },
    Assignment {
        lhs: TyExpr<'tcx>,
        operator: Option<Operator>,
        expr: TyExpr<'tcx>,
    },
    Expr {
        expr: TyExpr<'tcx>,
    },
    Function {
        name: String,
        is_extern: bool,
        args: Vec<Argument<'tcx>>,
        ty: Ty<'tcx>,
        body: Vec<Item<'tcx>>,
    },
    Struct {
        name: String,
        fields: Vec<Field<'tcx>>,
    },
    If {
        condition: TyExpr<'tcx>,
        arm_true: Vec<Item<'tcx>>,
        arm_false: Option<Vec<Item<'tcx>>>,
    },
    ForIn {
        name: String,
        expr: TyExpr<'tcx>,
        body: Vec<Item<'tcx>>,
    },
    Loop {
        body: Vec<Item<'tcx>>,
    },
    Break,
    Yield(Box<TyExpr<'tcx>>),
    Return(Box<TyExpr<'tcx>>),
    Block(Vec<Item<'tcx>>),
}
