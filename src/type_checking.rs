use std::borrow::Borrow;
use std::collections::HashMap;
use std::ptr::addr_of;

use crate::arena::Arena;
use crate::ast;
use crate::ir::{Bits, Var};
use crate::types::{Type, TypeRef};

fn is_coercible_to(ty: TypeRef<'_>, target: TypeRef<'_>) -> bool {
    match (ty, target) {
        (Type::Integer, Type::I32 | Type::U32) => true,
        (Type::Float, Type::F32) => true,
        _ => false
    }
}

fn is_compatible_to(ty: TypeRef<'_>, subty: TypeRef<'_>) -> bool {
    match (ty, subty) {
        (Type::Bool, Type::Bool) => true,
        (Type::U32, Type::U32) => true,
        (Type::I32, Type::I32) => true,
        (Type::F32, Type::F32) => true,
        (Type::Array(len1, ty1), Type::Array(len2, ty2)) => {
            len1 == len2 && is_compatible_to(ty1, ty2)
        }
        (Type::Array(_, ty1), Type::Slice(ty2)) => is_compatible_to(ty1, ty2),
        (Type::Slice(ty1), Type::Slice(ty2)) => is_compatible_to(ty1, ty2),
        (Type::Unit, Type::Unit) => true,
        (Type::Tuple(ty1), Type::Tuple(ty2)) => {
            ty1.len() == ty2.len()
                && ty1
                .iter()
                .zip(ty2.iter())
                .all(|(ty1, ty2)| is_compatible_to(ty1, ty2))
        }
        (Type::Function(args1, ret1), Type::Function(args2, ret2)) => {
            if args1.len() != args2.len() {
                return false;
            }

            if !is_compatible_to(ret1, ret2) {
                return false;
            }

            args1
                .iter()
                .zip(args2.iter())
                .all(|(ty1, ty2)| is_compatible_to(ty1, ty2))
        }
        (Type::Pointer(ty1), Type::Pointer(ty2)) => is_compatible_to(ty1, ty2),
        (Type::Any, _) | (_, Type::Any) => true,
        (Type::Struct { fields: lhs_fields }, Type::Struct { fields: rhs_fields }) => {
            std::iter::zip(lhs_fields, rhs_fields).all(
                |((lhs_name, lhs_ty), (rhs_name, rhs_ty))| {
                    lhs_name == rhs_name && is_compatible_to(lhs_ty, rhs_ty)
                },
            )
        }
        _ => false,
    }
}

macro_rules! intrinsics {
    ($($name:ident),*) => {
        #[derive(Copy, Clone, PartialEq, Debug)]
        #[allow(non_camel_case_types)]
        pub enum Intrinsic { $($name,)* }

        impl Intrinsic {
            pub fn to_str(&self) -> &'static str {
                 match self {
                    $(Intrinsic::$name => stringify!($name)),*
                }
            }
        }
    }
}

intrinsics! {
    debug
}

pub(crate) type ExprRef<'expr> = &'expr Expression<'expr>;

#[derive(Debug, Clone)]
pub(crate) enum Expression<'expr> {
    Identifier(String),
    Integer(Bits),
    Float(f64),
    Bool(bool),
    Infix(ast::Operator, ExprRef<'expr>, ExprRef<'expr>),
    Prefix(ast::Operator, ExprRef<'expr>),
    Index(ExprRef<'expr>, ExprRef<'expr>),
    Array(Vec<ExprRef<'expr>>),
    Call(ExprRef<'expr>, Vec<ExprRef<'expr>>),
    Tuple(Vec<ExprRef<'expr>>),
    StructLiteral(Vec<ExprRef<'expr>>),
    Range(ExprRef<'expr>, Option<ExprRef<'expr>>),
    Cast(ExprRef<'expr>),
    Field(ExprRef<'expr>, usize),
    Error,
    Var(Var),
    Intrinsic(Intrinsic),
}

#[derive(Debug, Clone)]
pub(crate) struct Argument<'tcx> {
    pub(crate) name: String,
    pub(crate) ty: TypeRef<'tcx>,
}

#[derive(Debug, Clone)]
pub(crate) enum Item<'expr, 'tcx> {
    Let {
        name: String,
        ty: TypeRef<'tcx>,
        expr: Option<ExprRef<'expr>>,
    },
    Assignment {
        lhs: ExprRef<'expr>,
        operator: Option<ast::Operator>,
        expr: ExprRef<'expr>,
    },
    Expression {
        expr: ExprRef<'expr>,
    },
    Function {
        name: String,
        is_extern: bool,
        args: Vec<Argument<'tcx>>,
        ty: TypeRef<'tcx>,
        body: Vec<Item<'expr, 'tcx>>,
    },
    If {
        condition: ExprRef<'expr>,
        arm_true: Vec<Item<'expr, 'tcx>>,
        arm_false: Option<Vec<Item<'expr, 'tcx>>>,
    },
    ForIn {
        name: String,
        expr: ExprRef<'expr>,
        body: Vec<Item<'expr, 'tcx>>,
    },
    Loop {
        body: Vec<Item<'expr, 'tcx>>,
    },
    Break,
    Yield(ExprRef<'expr>),
    Return(ExprRef<'expr>),
    Block(Vec<Item<'expr, 'tcx>>),
    Assert(ExprRef<'expr>),
}

pub(crate) struct ExprToType<'tcx> {
    map: indexmap::IndexMap<*const (), TypeRef<'tcx>>,
}

impl<'tcx> ExprToType<'tcx> {
    pub(crate) fn new() -> Self {
        Self {
            map: indexmap::IndexMap::new(),
        }
    }

    fn try_insert(&mut self, expr: ExprRef<'_>, ty: TypeRef<'tcx>) -> Option<TypeRef<'tcx>> {
        self.map.insert(addr_of!(*expr).cast(), ty)
    }

    pub(crate) fn insert(&mut self, expr: ExprRef<'_>, ty: TypeRef<'tcx>) {
        if let Some(old) = self.try_insert(expr, ty) {
            assert_eq!(old, ty, "type should not change");
        }
    }

    fn try_coerce(&mut self, expr: ExprRef<'_>, ty: TypeRef<'tcx>) -> bool {
        log::debug!("trying coercion of {:?} with type {:?} to {:?}", expr, self.of(expr), ty);
        match self.of(expr) {
            Type::Unknown => panic!("coercion failed??"),
            other if is_coercible_to(other, ty) => {
                self.try_insert(expr, ty);
                true
            }
            _ => false,
        }
    }

    pub(crate) fn try_coerce_any(&mut self, lhs: ExprRef<'_>, rhs: ExprRef<'_>) -> bool {
        if self.try_coerce(lhs, self.of(rhs)) ||
            self.try_coerce(rhs, self.of(lhs)) {
            return true;
        }

        match (self.of(lhs), self.of(rhs)) {
            (Type::Integer, Type::Integer) => {
                self.try_insert(lhs, &Type::I32);
                self.try_insert(rhs, &Type::I32);
                true
            }
            _ => false,
        }
    }

    pub(crate) fn of(&self, expr: ExprRef<'_>) -> TypeRef<'tcx> {
        match self.map.get(&addr_of!(*expr).cast()) {
            Some(t) => t,
            None => &Type::Unknown,
        }
    }
}


pub(crate) struct TypeCheckerContext<'tcx, 'expr, 'ast> {
    pub(crate) arena: &'tcx Arena<Type<'tcx>>,
    pub(crate) locals: HashMap<&'ast str, TypeRef<'tcx>>,
    pub(crate) defined_types: HashMap<&'expr str, TypeRef<'tcx>>,
    pub(crate) exprs: &'expr Arena<Expression<'expr>>,
    pub(crate) type_by_expr: ExprToType<'tcx>,
}

impl<'ast, 'tcx, 'expr> TypeCheckerContext<'tcx, 'expr, 'ast>
    where 'ast: 'expr
{
    pub(crate) fn make_expr(
        &mut self,
        ty: TypeRef<'tcx>,
        expr: Expression<'expr>,
    ) -> ExprRef<'expr> {
        let expr = self.exprs.alloc(expr);
        self.type_by_expr.insert(expr, ty);
        expr
    }

    fn deduce_expr_ty(&mut self, expr: &ast::Expression) -> ExprRef<'expr> {
        let (expr, ty) = match expr {
            ast::Expression::Bool(val) => (Expression::Bool(*val), self.arena.alloc(Type::Bool)),
            ast::Expression::Integer(val) => (
                // FIXME: type
                Expression::Integer((*val as i32).into()),
                self.arena.alloc(Type::Integer)
            ),
            ast::Expression::Float(val) => (Expression::Float(*val), self.arena.alloc(Type::Float)),
            ast::Expression::Infix(op, lhs, rhs) => {
                let lhs = self.deduce_expr_ty(lhs);
                let rhs = self.deduce_expr_ty(rhs);
                let ty = if is_compatible_to(self.type_by_expr.of(lhs), self.type_by_expr.of(rhs)) ||
                    self.type_by_expr.try_coerce_any(lhs, rhs)
                {
                    match op {
                        ast::Operator::Less
                        | ast::Operator::LessEqual
                        | ast::Operator::Greater
                        | ast::Operator::GreaterEqual
                        | ast::Operator::Equal
                        | ast::Operator::NotEqual => self.arena.alloc(Type::Bool),
                        ast::Operator::Add
                        | ast::Operator::Sub
                        | ast::Operator::Mul
                        | ast::Operator::Div => self.type_by_expr.of(lhs),
                        ast::Operator::Negate => unimplemented!(),
                        ast::Operator::Ref => unimplemented!(),
                        ast::Operator::Deref => unimplemented!(),
                        ast::Operator::As => unimplemented!(),
                    }
                } else {
                    log::debug!(
                    "mismatched types {:?} and {:?}",
                    self.type_by_expr.of(lhs),
                    self.type_by_expr.of(rhs)
                );
                    self.arena.alloc(Type::Error)
                };

                (Expression::Infix(*op, lhs, rhs), ty)
            }
            ast::Expression::Prefix(op, expr) => {
                let inner = self.deduce_expr_ty(expr);
                let ty = match op {
                    ast::Operator::Ref => self.arena.alloc(Type::Pointer(self.type_by_expr.of(inner))),
                    ast::Operator::Deref => match self.type_by_expr.of(inner) {
                        Type::Pointer(inner) => inner,
                        _ => unimplemented!(),
                    },
                    _ => self.type_by_expr.of(inner),
                };
                (Expression::Prefix(*op, inner), ty)
            }
            ast::Expression::Identifier(ident) => {
                let ty = if let Some(ty) = self.locals.get(ident.as_str()) {
                    ty
                } else {
                    log::debug!("no local {:?}", ident);
                    self.arena.alloc(Type::Error)
                };
                (Expression::Identifier(ident.to_string()), ty)
            }
            ast::Expression::Place(expr, field) => {
                let lhs = self.deduce_expr_ty(expr);

                let Some(name) = field.as_str() else {
                    unimplemented!()
                };

                let (idx, (_, ty)) = match &self.type_by_expr.of(lhs) {
                    Type::Struct { fields } => fields
                        .iter()
                        .enumerate()
                        .find(|(idx, (n, _))| n == name)
                        .unwrap(),
                    _ => unimplemented!(),
                };

                (Expression::Field(lhs, idx), *ty)
            }
            ast::Expression::Array(items) => {
                if items.is_empty() {
                    return self.make_expr(
                        self.arena.alloc(Type::Unknown),
                        Expression::Error,
                    );
                }

                let mut values = Vec::new();

                let first =
                    self.deduce_expr_ty(&items[0]);
                let item_ty = self.type_by_expr.of(first);
                values.push(first);

                for next in items.iter().skip(1) {
                    let expr = self.deduce_expr_ty(next);
                    if is_compatible_to(self.type_by_expr.of(expr), item_ty) ||
                        self.type_by_expr.try_coerce_any(expr, first) {
                        //
                    } else {
                        log::debug!("incompatible types: {:?} and {:?}", self.type_by_expr.of(expr), item_ty);
                        return self.make_expr(
                            self.arena.alloc(Type::Error),
                            Expression::Error,
                        );
                    }
                    values.push(expr);
                }

                (
                    Expression::Array(values),
                    self.arena.alloc(Type::Array(items.len(), item_ty)),
                )
            }
            ast::Expression::Call(callee, args) => match callee.as_ref() {
                ast::Expression::Identifier(ident) => {
                    let callee = match ident.as_str() {
                        "debug" => {
                            let ty = self.arena.alloc(Type::Function(vec![&Type::Any], &Type::Any));
                            self.make_expr(
                                ty,
                                Expression::Intrinsic(Intrinsic::debug),
                            )
                        }
                        other => {
                            let ty = self.locals
                                .get(other)
                                .unwrap_or_else(|| panic!("a type for {}", other));
                            self.deduce_expr_ty(callee)
                        }
                    };

                    let (args_ty, ret_ty) = match self.type_by_expr.of(callee) {
                        Type::Function(args_ty, ret_ty) => (args_ty, ret_ty),
                        _ => {
                            log::debug!("{} is not callable", ident.as_str());
                            return self.make_expr(
                                self.arena.alloc(Type::Error),
                                Expression::Error,
                            );
                        }
                    };

                    let mut values = Vec::new();

                    for (arg, expected_ty) in args.iter().zip(args_ty) {
                        let arg =
                            self.deduce_expr_ty(arg);

                        if is_compatible_to(self.type_by_expr.of(arg), expected_ty) ||
                            self.type_by_expr.try_coerce(arg, expected_ty) {
                            //
                        } else {
                            log::debug!(
                            "incompatible types {:?} and {:?}",
                            self.type_by_expr.of(arg),
                            expected_ty
                        );
                            return self.make_expr(
                                self.arena.alloc(Type::Error),
                                Expression::Error,
                            );
                        }

                        values.push(arg);
                    }

                    (Expression::Call(callee, values), *ret_ty)
                }
                expr => unimplemented!("{:?}", expr),
            },
            ast::Expression::Range(from, Some(to)) => {
                let from = self.deduce_expr_ty(from);
                let to = self.deduce_expr_ty(to);
                if is_compatible_to(self.type_by_expr.of(from), self.type_by_expr.of(to))
                    || self.type_by_expr.try_coerce_any(from, to) {
                    //
                } else {
                    log::debug!("incompatible range bounds");
                    return self.make_expr(
                        self.arena.alloc(Type::Error),
                        Expression::Error,
                    );
                }
                (Expression::Range(from, Some(to)), self.arena.alloc(Type::Range))
            }
            ast::Expression::Range(to, None) => {
                unimplemented!()
            }
            ast::Expression::Tuple(items) => {
                let mut values = Vec::new();
                let mut types = Vec::new();

                for value in items {
                    let expr = self.deduce_expr_ty(value);
                    types.push(self.type_by_expr.of(expr));
                    values.push(expr);
                }

                (Expression::Tuple(values), self.arena.alloc(Type::Tuple(types)))
            }
            ast::Expression::Index(array_expr, index_expr) => {
                let array = self.deduce_expr_ty(array_expr);
                let index = self.deduce_expr_ty(index_expr);

                let ty = match (self.type_by_expr.of(array), self.type_by_expr.of(index)) {
                    (Type::Array(_, item_ty), idx)
                    | (Type::Slice(item_ty), idx) => {
                        self.type_by_expr.try_coerce(array, &Type::I32);
                        idx
                    }
                    _ => self.arena.alloc(Type::Error),
                };

                (Expression::Index(array, index), ty)
            }
            ast::Expression::Var(_) => unreachable!(),
            ast::Expression::Cast(expr, ty) => {
                let expr = self.deduce_expr_ty(expr);
                let target_ty = self.unify(ty);
                (Expression::Cast(expr), target_ty)
            }
            ast::Expression::StructLiteral(expr, fields) => {
                let ty = match expr {
                    Some(name) => self.defined_types[name.as_str()],
                    None => &Type::Unknown,
                };

                let fields = fields
                    .iter()
                    .map(|(name, expr)| {
                        self.deduce_expr_ty(expr)
                    })
                    .collect();
                (Expression::StructLiteral(fields), ty)
            }
        };
        self.make_expr(ty, expr)
    }

    pub(crate) fn infer_types(
        &mut self,
        items: &'ast [ast::Item],
        expected_ret_ty: Option<TypeRef<'tcx>>,
    ) -> Vec<Item<'expr, 'tcx>>  {
        let mut lowered_items = Vec::new();

        for item in items.iter() {
            let item = match item {
                ast::Item::Let {
                    name,
                    r#type: expected_ty,
                    expr,
                } => {
                    if expr.is_none() {
                        log::debug!("no expression on the right hand side of the let binding");
                        continue;
                    }
                    let expr = self.deduce_expr_ty(
                        expr.as_ref().unwrap(),
                    );
                    log::debug!("deduced type {:?} for binding {}", expr, name);
                    let ty = match expected_ty {
                        Some(expected) => {
                            let target_ty = self.unify(expected);
                            let source_ty = self.type_by_expr.of(expr);
                            if is_compatible_to(target_ty, source_ty) ||
                                self.type_by_expr.try_coerce(expr, target_ty) {
                                target_ty
                            } else {
                                log::debug!("mismatched types. expected {:?}, got {:?}", target_ty, source_ty);
                                continue;
                            }
                        }
                        None => &self.type_by_expr.of(expr),
                    };
                    self.locals.insert(name, ty);

                    Item::Let {
                        name: name.clone(),
                        ty,
                        expr: Some(expr),
                    }
                }
                ast::Item::Assignment {
                    lhs,
                    operator,
                    expr,
                } => {
                    let lhs = self.deduce_expr_ty(lhs);
                    let rhs = self.deduce_expr_ty(expr);

                    if is_compatible_to(self.type_by_expr.of(lhs), self.type_by_expr.of(rhs)) ||
                        self.type_by_expr.try_coerce_any(lhs, rhs) {
                        //
                    } else {
                        log::debug!(
                        "incompatible types in assignment, got {:?} and {:?}",
                        self.type_by_expr.of(lhs),
                        self.type_by_expr.of(rhs)
                    );
                        continue;
                    }

                    Item::Assignment {
                        lhs,
                        operator: *operator,
                        expr: rhs,
                    }
                }
                ast::Item::Expr { expr } => {
                    let expr = self.deduce_expr_ty(expr);
                    Item::Expression { expr }
                }
                ast::Item::Function {
                    name,
                    params,
                    ty,
                    body,
                    ..
                } => {
                    let mut args = Vec::new();
                    for param in params {
                        let ty = self.unify(&param.r#type,);
                        log::debug!("Found arg {} of type {:?}", &param.name, ty);
                        self.locals.insert(
                            param.name.as_str(),
                            self.unify(&param.r#type),
                        );
                        args.push(ty);
                    }

                    let func_ty = Type::Function(args, self.unify(ty));
                    let func_ty = self.arena.alloc(func_ty);
                    self.locals.insert(name.as_str(), func_ty);

                    let body = self.infer_types(
                        body,
                        Some(self.unify(ty)),
                    );
                    Item::Function {
                        name: name.clone(),
                        is_extern: false,
                        args: params
                            .iter()
                            .map(|it| Argument {
                                name: it.name.clone(),
                                ty: self.unify(&it.r#type),
                            })
                            .collect(),
                        ty: self.unify(ty),
                        body,
                    }
                }
                ast::Item::Struct { name, fields } => {
                    let fields: Vec<_> = fields
                        .iter()
                        .map(|field| {
                            (
                                field.name.clone(),
                                self.unify(&field.r#type),
                            )
                        })
                        .collect();
                    self.defined_types.insert(name, self.arena.alloc(Type::Struct { fields }));
                    continue;
                }
                ast::Item::If {
                    condition,
                    arm_true,
                    arm_false,
                } => {
                    let cond = self.deduce_expr_ty(
                        condition,
                    );
                    if !is_compatible_to(self.type_by_expr.of(cond), self.arena.alloc(Type::Bool)) {
                        log::debug!("only boolean expressions are allowed in if conditions, got {:?}", self.type_by_expr.of(cond));
                        continue;
                    }
                    Item::If {
                        condition: cond,
                        arm_true: self.infer_types(
                            arm_true,
                            expected_ret_ty,
                        ),
                        arm_false: if let Some(arm_false) = arm_false {
                            Some(self.infer_types(
                                arm_false,
                                expected_ret_ty,
                            ))
                        } else {
                            None
                        },
                    }
                }
                ast::Item::ForIn { name, expr, body } => {
                    let expr = self.deduce_expr_ty(expr);
                    let is_iterable = match self.type_by_expr.of(expr) {
                        Type::Array(_, _) | Type::Slice(_) => true,
                        Type::Range => true,
                        _ => false,
                    };
                    if !is_iterable {
                        log::debug!("{:?} is not iterable", self.type_by_expr.of(expr));
                        continue;
                    }
                    self.locals.insert(name.as_str(), self.arena.alloc(Type::I32));
                    let body = self.infer_types(
                        body,
                        expected_ret_ty,
                    );
                    Item::ForIn {
                        name: name.clone(),
                        expr,
                        body,
                    }
                }
                ast::Item::Loop { body } => Item::Loop {
                    body: self.infer_types(
                        body,
                        expected_ret_ty,
                    ),
                },
                ast::Item::Return(expr) => {
                    if expected_ret_ty.is_none() {
                        panic!("return outside of a function");
                    }
                    let expr = self.deduce_expr_ty(expr);
                    if is_compatible_to(self.type_by_expr.of(expr), expected_ret_ty.unwrap()) ||
                        self.type_by_expr.try_coerce(expr, expected_ret_ty.unwrap()) {
                        //
                    } else {
                        log::debug!(
                        "function marked as returning {:?} but returned {:?}",
                        expected_ret_ty.unwrap(),
                        expr
                    );
                        continue;
                    }
                    Item::Return(expr)
                }
                ast::Item::Break => Item::Break,
                ast::Item::Yield(_) => unimplemented!(),
                ast::Item::Block(body) => {
                    self.infer_types(
                        body,
                        expected_ret_ty,
                    );
                    todo!()
                }
                ast::Item::Assert(expr) => {
                    let expr = self.deduce_expr_ty(expr);
                    Item::Assert(expr)
                }
            };

            lowered_items.push(item);
        }

        lowered_items
    }

    fn unify(&self, ty: &ast::Type) -> TypeRef<'tcx> {
        match ty {
            ast::Type::Name(name) => {
                match name.as_str() {
                    "i32" => self.arena.alloc(Type::I32),
                    "u32" => self.arena.alloc(Type::U32),
                    "f32" => self.arena.alloc(Type::F32),
                    "bool" => self.arena.alloc(Type::Bool),
                    custom if let Some(ty) = self.defined_types.get(custom) => ty,
                    custom => {
                        log::warn!("missing type for custom {:?}", custom);
                        return &Type::Unknown;
                    }
                }
            }
            ast::Type::Tuple(types) => {
                let types: Vec<_> = types.iter()
                    .map(|it| self.unify(it))
                    .collect();
                self.arena.alloc(Type::Tuple(types))
            }
            ast::Type::Pointer(ty) => self.arena.alloc(Type::Pointer(self.unify(ty))),
            ast::Type::Array(len, ty) => self.arena.alloc(Type::Array(*len, self.unify(ty))),
            ast::Type::Slice(item_ty) => self.arena.alloc(Type::Slice(self.unify(item_ty))),
            ast::Type::Unit => self.arena.alloc(Type::Unit),
            ast::Type::Function(args_ty, ret_ty) => {
                let args = args_ty.iter().map(|it| self.unify(it)).collect();
                self.arena.alloc(Type::Function(args, self.unify(ret_ty)))
            }
        }
    }
}