use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Index;
use std::ptr::addr_of;

use crate::arena::Arena;
use crate::ast;
use crate::ir::Var;
use crate::types::{Type, TypeRef};

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
            std::iter::zip(lhs_fields, rhs_fields).all(|((lhs_name, lhs_ty), (rhs_name, rhs_ty))| {
                lhs_name == rhs_name && is_compatible_to(lhs_ty, rhs_ty)
            })
        }
        _ => false,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TypedExpression<'expr, 'tcx> {
    pub(crate) ty: TypeRef<'tcx>,
    pub(crate) expr: Expression<'expr>,
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
    Integer(i64),
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
    Let { name: String, ty: TypeRef<'tcx>, expr: Option<&'expr Expression<'expr>> },
    Assignment { lhs: ExprRef<'expr>, operator: Option<ast::Operator>, expr: ExprRef<'expr> },
    Expression { expr: ExprRef<'expr> },
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

    pub(crate) fn insert(&mut self, expr: ExprRef<'_>, ty: TypeRef<'tcx>) {
        if let Some(old) = self.map.insert(addr_of!(*expr).cast(), ty) {
            assert_eq!(old, ty, "type should not change");
        }
    }

    pub(crate) fn of(&self, expr: ExprRef<'_>) -> TypeRef<'tcx> {
        match self.map.get(&addr_of!(*expr).cast()) {
            Some(t) => t,
            None => &Type::Unknown,
        }
    }
}

fn deduce_expr_ty<'tcx, 'expr>(
    expr: &ast::Expression,
    arena: &'tcx Arena<Type<'tcx>>,
    locals: &HashMap<&str, TypeRef<'tcx>>,
    defined_types: &HashMap<&str, TypeRef<'tcx>>,
    exprs: &'expr Arena<Expression<'expr>>,
    type_by_expr: &mut ExprToType<'tcx>,
) -> ExprRef<'expr> {
    let make_expr = |exprs: &'expr Arena<Expression<'expr>>,
                     type_by_expr: &mut ExprToType<'tcx>,
                     tye: TypedExpression<'expr, 'tcx>| -> &'expr Expression<'expr> {
        let expr = exprs.alloc(tye.expr);
        type_by_expr.insert(expr, tye.ty);
        expr
    };

    let tye = match expr {
        ast::Expression::Integer(val) => TypedExpression { expr: Expression::Integer(*val), ty: arena.alloc(Type::I32) },
        ast::Expression::Float(val) => TypedExpression { expr: Expression::Float(*val), ty: arena.alloc(Type::F32) },
        ast::Expression::Bool(val) => TypedExpression { expr: Expression::Bool(*val), ty: arena.alloc(Type::Bool) },
        ast::Expression::Infix(op, lhs, rhs) => {
            let lhs = deduce_expr_ty(lhs, arena, &locals, defined_types, exprs, type_by_expr);
            let rhs = deduce_expr_ty(rhs, arena, &locals, defined_types, exprs, type_by_expr);
            let ty = if !is_compatible_to(type_by_expr.of(lhs), type_by_expr.of(rhs)) {
                log::debug!("mismatched types {:?} and {:?}", type_by_expr.of(lhs), type_by_expr.of(rhs));
                arena.alloc(Type::Error)
            } else {
                match op {
                    ast::Operator::Less
                    | ast::Operator::LessEqual
                    | ast::Operator::Greater
                    | ast::Operator::GreaterEqual
                    | ast::Operator::Equal
                    | ast::Operator::NotEqual => arena.alloc(Type::Bool),
                    ast::Operator::Add
                    | ast::Operator::Sub
                    | ast::Operator::Mul
                    | ast::Operator::Div => type_by_expr.of(lhs),
                    ast::Operator::Negate => unimplemented!(),
                    ast::Operator::Ref => unimplemented!(),
                    ast::Operator::Deref => unimplemented!(),
                    ast::Operator::As => unimplemented!(),
                }
            };

            TypedExpression {
                expr: Expression::Infix(*op, lhs, rhs),
                ty,
            }
        }
        ast::Expression::Prefix(op, expr) => {
            let inner = deduce_expr_ty(expr, arena, &locals, defined_types, exprs, type_by_expr);
            let ty = match op {
                ast::Operator::Ref => arena.alloc(Type::Pointer(type_by_expr.of(inner))),
                ast::Operator::Deref => {
                    match type_by_expr.of(inner) {
                        Type::Pointer(inner) => inner,
                        _ => unimplemented!(),
                    }
                }
                _ => type_by_expr.of(inner),
            };
            TypedExpression {
                expr: Expression::Prefix(*op, inner),
                ty,
            }
        }
        ast::Expression::Identifier(ident) => {
            let ty = if let Some(ty) = locals.get(ident.as_str()) {
                ty
            } else {
                log::debug!("no local {:?}", ident);
                arena.alloc(Type::Error)
            };
            TypedExpression {
                expr: Expression::Identifier(ident.to_string()),
                ty,
            }
        }
        ast::Expression::Place(expr, field) => {
            let lhs = deduce_expr_ty(expr, arena, &locals, defined_types, exprs, type_by_expr);

            let Some(name) = field.as_str() else {
                unimplemented!()
            };

            let (idx, (_, ty)) = match &type_by_expr.of(lhs) {
                Type::Struct { fields } => fields.iter().enumerate().find(|(idx, (n, _))| n == name).unwrap(),
                _ => unimplemented!(),
            };

            TypedExpression {
                ty,
                expr: Expression::Field(lhs, idx),
            }
        }
        ast::Expression::Array(items) => {
            if items.is_empty() {
                return make_expr(exprs, type_by_expr, TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Unknown) });
            }

            let mut values = Vec::new();

            let first = deduce_expr_ty(&items[0], arena, locals, defined_types, exprs, type_by_expr);
            let item_ty = type_by_expr.of(first);
            values.push(first);

            for next in items.iter().skip(1) {
                let expr = deduce_expr_ty(next, arena, locals, defined_types, exprs, type_by_expr);
                if !is_compatible_to(&type_by_expr.of(expr), item_ty) {
                    log::debug!("incompatible types: {:?} and {:?}", expr, item_ty);
                    return make_expr(exprs, type_by_expr, TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) });
                }
                values.push(expr);
            }

            TypedExpression {
                expr: Expression::Array(values),
                ty: arena.alloc(Type::Array(items.len(), item_ty)),
            }
        }
        ast::Expression::Call(callee, args) => {
            match callee.as_ref() {
                ast::Expression::Identifier(ident) => {
                    let callee = match ident.as_str() {
                        "debug" => {
                            let ty = arena.alloc(Type::Function(vec![&Type::Any], &Type::Any));
                            make_expr(exprs, type_by_expr, TypedExpression { ty, expr: Expression::Intrinsic(Intrinsic::debug) })
                        }
                        other => {
                            let ty = locals.get(other).unwrap_or_else(|| panic!("a type for {}", other));
                            deduce_expr_ty(callee, arena, locals, defined_types, exprs, type_by_expr)
                        }
                    };

                    let (args_ty, ret_ty) = match type_by_expr.of(callee) {
                        Type::Function(args_ty, ret_ty) => (args_ty, ret_ty),
                        _ => {
                            log::debug!("{} is not callable", ident.as_str());
                            return make_expr(exprs, type_by_expr, TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) });
                        }
                    };

                    let mut values = Vec::new();

                    for (arg, expected_ty) in args.iter().zip(args_ty) {
                        let arg = deduce_expr_ty(arg, arena, locals, defined_types, exprs, type_by_expr);

                        if !is_compatible_to(type_by_expr.of(arg), expected_ty) {
                            log::debug!("incompatible types {:?} and {:?}", type_by_expr.of(arg), expected_ty);
                            return make_expr(exprs, type_by_expr, TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) });
                        }

                        values.push(arg);
                    }

                    TypedExpression {
                        expr: Expression::Call(callee, values),
                        ty: ret_ty,
                    }
                }
                expr => unimplemented!("{:?}", expr),
            }
        }
        ast::Expression::Range(from, Some(to)) => {
            let from = deduce_expr_ty(from, arena, locals, defined_types, exprs, type_by_expr);
            let to = deduce_expr_ty(to, arena, locals, defined_types, exprs, type_by_expr);
            if !is_compatible_to(type_by_expr.of(from), type_by_expr.of(to)) {
                log::debug!("incompatible range bounds");
                return make_expr(exprs, type_by_expr, TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) });
            }
            TypedExpression {
                expr: Expression::Range(
                    from,
                    Some(to),
                ),
                ty: arena.alloc(Type::Range),
            }
        }
        ast::Expression::Range(to, None) => {
            unimplemented!()
        }
        ast::Expression::Tuple(items) => {
            let mut values = Vec::new();
            let mut types = Vec::new();

            for value in items {
                let expr = deduce_expr_ty(value, arena, locals, defined_types, exprs, type_by_expr);
                types.push(type_by_expr.of(expr));
                values.push(expr);
            }

            TypedExpression { expr: Expression::Tuple(values), ty: arena.alloc(Type::Tuple(types)) }
        }
        ast::Expression::Index(arr, index_expr) => {
            let lhs = deduce_expr_ty(arr, arena, locals, defined_types, exprs, type_by_expr);
            let rhs = deduce_expr_ty(index_expr, arena, locals, defined_types, exprs, type_by_expr);

            let ty = match (type_by_expr.of(lhs), type_by_expr.of(rhs)) {
                (Type::Array(_, item_ty), Type::I32) => item_ty,
                (Type::Slice(item_ty), Type::I32) => item_ty,
                _ => arena.alloc(Type::Error),
            };

            TypedExpression {
                expr: Expression::Index(lhs, rhs),
                ty,
            }
        }
        ast::Expression::Var(_) => unreachable!(),
        ast::Expression::Cast(expr, ty) => {
            let expr = deduce_expr_ty(expr, arena, locals, defined_types, exprs, type_by_expr);
            let target_ty = unify(ty, arena, defined_types);
            TypedExpression {
                expr: Expression::Cast(expr),
                ty: target_ty,
            }
        }
        ast::Expression::StructLiteral(expr, fields) => {
            let ty = match expr {
                Some(name) => defined_types[name.as_str()],
                None => &Type::Unknown,
            };

            let fields = fields
                .iter()
                .map(|(name, expr)| deduce_expr_ty(expr, arena, locals, defined_types, exprs, type_by_expr))
                .collect();
            TypedExpression {
                expr: Expression::StructLiteral(fields),
                ty,
            }
        }
    };
    make_expr(exprs, type_by_expr, tye)
}

pub(crate) fn infer_types<'ast, 'tcx: 'ast, 'expr>(
    items: &'ast [ast::Item],
    arena: &'tcx Arena<Type<'tcx>>,
    locals: &mut HashMap<&'ast str, TypeRef<'tcx>>,
    expected_ret_ty: Option<TypeRef<'tcx>>,
    defined_types: &mut HashMap<&'ast str, TypeRef<'tcx>>,
    exprs: &'expr Arena<Expression<'expr>>,
    type_by_expr: &mut ExprToType<'tcx>,
) -> Vec<Item<'expr, 'tcx>> {
    let mut lowered_items = Vec::new();

    for item in items.iter() {
        let item = match item {
            ast::Item::Let { name, r#type: expected_ty, expr } => {
                if expr.is_none() {
                    log::debug!("no expression on the right hand side of the let binding");
                    continue;
                }
                let expr = deduce_expr_ty(expr.as_ref().unwrap(), arena, &locals, defined_types, exprs, type_by_expr);
                log::debug!("deduced type {:?} for binding {}", expr, name);
                let ty = match expected_ty {
                    Some(expected) => {
                        let ty = unify(expected, arena, defined_types);
                        if !is_compatible_to(ty, &type_by_expr.of(expr)) {
                            log::debug!("mismatched types. expected {:?}, got {:?}", ty, expr);
                            continue;
                        }
                        ty
                    }
                    None => &type_by_expr.of(expr),
                };
                locals.insert(name, ty);

                Item::Let { name: name.clone(), ty, expr: Some(expr) }
            }
            ast::Item::Assignment {
                lhs,
                operator,
                expr,
            } => {
                let lhs = deduce_expr_ty(lhs, arena, &locals, defined_types, exprs, type_by_expr);
                let rhs = deduce_expr_ty(expr, arena, &locals, defined_types, exprs, type_by_expr);

                if !is_compatible_to(type_by_expr.of(lhs), type_by_expr.of(rhs)) {
                    log::debug!("incompatible types in assignment, got {:?} and {:?}", type_by_expr.of(lhs), type_by_expr.of(rhs));
                    continue;
                }

                Item::Assignment { lhs, operator: *operator, expr: rhs }
            }
            ast::Item::Expr { expr } => {
                let expr = deduce_expr_ty(expr, arena, locals, defined_types, exprs, type_by_expr);
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
                    let ty = unify(&param.r#type, arena, defined_types);
                    log::debug!("Found arg {} of type {:?}", &param.name, ty);
                    locals.insert(param.name.as_str(), unify(&param.r#type, arena, defined_types));
                    args.push(ty);
                }

                let func_ty = Type::Function(args, unify(ty, arena, defined_types));
                let func_ty = arena.alloc(func_ty);
                locals.insert(name.as_str(), func_ty);

                let body = infer_types(body, arena, locals, Some(unify(ty, arena, defined_types)), defined_types, exprs, type_by_expr);
                Item::Function {
                    name: name.clone(),
                    is_extern: false,
                    args: params.iter().map(|it| Argument {
                        name: it.name.clone(),
                        ty: unify(&it.r#type, arena, defined_types),
                    }).collect(),
                    ty: unify(ty, arena, defined_types),
                    body,
                }
            }
            ast::Item::Struct { name, fields } => {
                let fields: Vec<_> = fields.iter().map(|field| {
                    (field.name.clone(), unify(&field.r#type, arena, defined_types))
                }).collect();
                defined_types.insert(name, arena.alloc(Type::Struct { fields }));
                continue;
            }
            ast::Item::If {
                condition,
                arm_true,
                arm_false,
            } => {
                let cond = deduce_expr_ty(condition, arena, &locals, defined_types, exprs, type_by_expr);
                if !is_compatible_to(type_by_expr.of(cond), arena.alloc(Type::Bool)) {
                    log::debug!("only boolean expressions are allowed in if conditions");
                    continue;
                }
                Item::If {
                    condition: cond,
                    arm_true: infer_types(arm_true, arena, locals, expected_ret_ty, defined_types, exprs, type_by_expr),
                    arm_false: if let Some(arm_false) = arm_false {
                        Some(infer_types(arm_false, arena, locals, expected_ret_ty, defined_types, exprs, type_by_expr))
                    } else {
                        None
                    },
                }
            }
            ast::Item::ForIn { name, expr, body } => {
                let expr = deduce_expr_ty(expr, arena, locals, defined_types, exprs, type_by_expr);
                let is_iterable = match type_by_expr.of(expr) {
                    Type::Array(_, _) | Type::Slice(_) => true,
                    Type::Range => true,
                    _ => false,
                };
                if !is_iterable {
                    log::debug!("{:?} is not iterable", type_by_expr.of(expr));
                    continue;
                }
                locals.insert(name.as_str(), arena.alloc(Type::I32));
                let body = infer_types(body, arena, locals, expected_ret_ty, defined_types, exprs, type_by_expr);
                Item::ForIn {
                    name: name.clone(),
                    expr,
                    body,
                }
            }
            ast::Item::Loop { body } => {
                Item::Loop {
                    body: infer_types(body, arena, locals, expected_ret_ty, defined_types, exprs, type_by_expr)
                }
            }
            ast::Item::Return(expr) => {
                if expected_ret_ty.is_none() {
                    panic!("return outside of a function");
                }
                let expr = deduce_expr_ty(expr, arena, locals, defined_types, exprs, type_by_expr);
                if !is_compatible_to(type_by_expr.of(expr), expected_ret_ty.unwrap()) {
                    log::debug!("function marked as returning {:?} but returned {:?}",
                        expected_ret_ty.unwrap(),
                        expr
                    );
                    continue;
                }
                Item::Return(expr)
            }
            ast::Item::Break => {
                Item::Break
            }
            ast::Item::Yield(_) => unimplemented!(),
            ast::Item::Block(body) => {
                infer_types(body, arena, locals, expected_ret_ty, defined_types, exprs, type_by_expr);
                todo!()
            }
            ast::Item::Assert(expr) => {
                let expr = deduce_expr_ty(expr, arena, locals, defined_types, exprs, type_by_expr);
                Item::Assert(expr)
            }
        };

        lowered_items.push(item);
    }

    lowered_items
}

fn unify<'tcx>(
    ty: &ast::Type,
    arena: &'tcx Arena<Type<'tcx>>,
    defined_types: &HashMap<&str, TypeRef<'tcx>>,
) -> TypeRef<'tcx> {
    match ty {
        ast::Type::Name(name) => {
            match name.as_str() {
                "i32" => arena.alloc(Type::I32),
                "u32" => arena.alloc(Type::U32),
                "f32" => arena.alloc(Type::F32),
                "bool" => arena.alloc(Type::Bool),
                custom => defined_types[custom],
            }
        }
        ast::Type::Tuple(types) => {
            let types: Vec<_> = types.iter()
                .map(|it| unify(it, arena, defined_types))
                .collect();
            arena.alloc(Type::Tuple(types))
        }
        ast::Type::Pointer(ty) => arena.alloc(Type::Pointer(unify(ty, arena, defined_types))),
        ast::Type::Array(len, ty) => arena.alloc(Type::Array(*len, unify(ty, arena, defined_types))),
        ast::Type::Slice(item_ty) => arena.alloc(Type::Slice(unify(item_ty, arena, defined_types))),
        ast::Type::Unit => arena.alloc(Type::Unit),
        ast::Type::Function(args_ty, ret_ty) => {
            let args = args_ty.iter().map(|it| unify(it, arena, defined_types)).collect();
            arena.alloc(Type::Function(args, unify(ret_ty, arena, defined_types)))
        }
    }
}