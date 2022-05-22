use std::collections::HashMap;

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
        (Type::Other(name1), Type::Other(name2)) => name1 == name2,
        (Type::Any, _) | (_, Type::Any) => true,
        _ => false,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TypedExpression<'tcx> {
    pub(crate) ty: TypeRef<'tcx>,
    pub(crate) expr: Expression<'tcx>,
}

#[derive(Debug, Clone)]
pub(crate) enum Expression<'tcx> {
    Identifier(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Infix(ast::Operator, Box<TypedExpression<'tcx>>, Box<TypedExpression<'tcx>>),
    Prefix(ast::Operator, Box<TypedExpression<'tcx>>),
    Index(Box<TypedExpression<'tcx>>, Box<TypedExpression<'tcx>>),
    Array(Vec<TypedExpression<'tcx>>),
    Call(Box<TypedExpression<'tcx>>, Vec<TypedExpression<'tcx>>),
    Tuple(Vec<TypedExpression<'tcx>>),
    Range(Box<TypedExpression<'tcx>>, Option<Box<TypedExpression<'tcx>>>),
    Cast(Box<TypedExpression<'tcx>>, TypeRef<'tcx>),
    Error,
    Var(Var),
}

#[derive(Debug, Clone)]
pub(crate) struct Argument<'tcx> {
    pub(crate) name: String,
    pub(crate) ty: TypeRef<'tcx>,
}

#[derive(Debug, Clone)]
pub(crate) enum Item<'tcx> {
    Let { name: String, ty: TypeRef<'tcx>, expr: Option<TypedExpression<'tcx>> },
    Assignment { lhs: TypedExpression<'tcx>, operator: Option<ast::Operator>, expr: TypedExpression<'tcx> },
    Expression { expr: TypedExpression<'tcx> },
    Function {
        name: String,
        is_extern: bool,
        args: Vec<Argument<'tcx>>,
        ty: TypeRef<'tcx>,
        body: Vec<Item<'tcx>>,
    },
    /*
        Struct {
            name: String,
            fields: Vec<Field>,
        },*/
    If {
        condition: TypedExpression<'tcx>,
        arm_true: Vec<Item<'tcx>>,
        arm_false: Option<Vec<Item<'tcx>>>,
    },
    ForIn {
        name: String,
        expr: TypedExpression<'tcx>,
        body: Vec<Item<'tcx>>,
    },
    Loop {
        body: Vec<Item<'tcx>>,
    },
    Break,
    Yield(Box<TypedExpression<'tcx>>),
    Return(Box<TypedExpression<'tcx>>),
    Block(Vec<Item<'tcx>>),
    Assert(Box<TypedExpression<'tcx>>),
}

fn deduce_expr_ty<'tcx>(
    expr: &ast::Expression,
    arena: &'tcx Arena<Type<'tcx>>,
    locals: &HashMap<&str, TypeRef<'tcx>>,
) -> TypedExpression<'tcx> {
    match expr {
        ast::Expression::Integer(val) => TypedExpression { expr: Expression::Integer(*val), ty: arena.alloc(Type::I32) },
        ast::Expression::Float(val) => TypedExpression { expr: Expression::Float(*val), ty: arena.alloc(Type::F32) },
        ast::Expression::Bool(val) => TypedExpression { expr: Expression::Bool(*val), ty: arena.alloc(Type::Bool) },
        ast::Expression::Infix(op, lhs, rhs) => {
            let lhs = deduce_expr_ty(lhs, arena, &locals);
            let rhs = deduce_expr_ty(rhs, arena, &locals);
            let ty = if !is_compatible_to(lhs.ty, rhs.ty) {
                log::debug!("mismatched types {:?} and {:?}", lhs.ty, rhs.ty);
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
                    | ast::Operator::Div => lhs.ty,
                    ast::Operator::Negate => unimplemented!(),
                    ast::Operator::Ref => unimplemented!(),
                    ast::Operator::Deref => unimplemented!(),
                    ast::Operator::As => unimplemented!(),
                }
            };

            TypedExpression {
                expr: Expression::Infix(*op, Box::new(lhs), Box::new(rhs)),
                ty,
            }
        }
        ast::Expression::Prefix(op, expr) => {
            let inner = deduce_expr_ty(expr, arena, &locals);
            let ty = match op {
                ast::Operator::Ref => arena.alloc(Type::Pointer(inner.ty)),
                ast::Operator::Deref => unimplemented!(),
                _ => inner.ty,
            };
            TypedExpression {
                expr: Expression::Prefix(*op, Box::new(inner)),
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
        ast::Expression::Place(expr, ty) => {
            log::debug!("unsupported place expr");
            unimplemented!()
        }
        ast::Expression::Array(items) => {
            if items.is_empty() {
                return TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Unknown) };
            }

            let mut values = Vec::new();

            let first = deduce_expr_ty(&items[0], arena, locals);
            let item_ty = first.ty;
            values.push(first);

            for next in items.iter().skip(1) {
                let expr = deduce_expr_ty(next, arena, locals);
                if !is_compatible_to(expr.ty, item_ty) {
                    log::debug!("incompatible types: {:?} and {:?}", expr.ty, item_ty);
                    return TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) };
                }
                values.push(expr);
            }

            TypedExpression {
                expr: Expression::Array(values),
                ty: arena.alloc(Type::Array(items.len(), item_ty)),
            }
        }
        ast::Expression::Call(callee, args) => {
            let callee = deduce_expr_ty(callee, arena, locals);
            match &callee.expr {
                Expression::Identifier(ident) => {
                    let callee_ty = match ident.as_str() {
                        "debug" => {
                            arena.alloc(Type::Function(vec![&Type::Any], &Type::Unit))
                        }
                        other => {
                            locals.get(other).unwrap_or_else(|| panic!("a type for {}", other))
                        }
                    };

                    let (args_ty, ret_ty) = match callee_ty {
                        Type::Function(args_ty, ret_ty) => (args_ty, ret_ty),
                        _ => {
                            log::debug!("{} is not callable", ident.as_str());
                            return TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) };
                        }
                    };

                    let mut values = Vec::new();

                    for (arg, expected_ty) in args.iter().zip(args_ty) {
                        let arg = deduce_expr_ty(arg, arena, locals);

                        if !is_compatible_to(arg.ty, expected_ty) {
                            log::debug!("incompatible types {:?} and {:?}", arg.ty, expected_ty);
                            return TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) };
                        }

                        values.push(arg);
                    }

                    TypedExpression {
                        expr: Expression::Call(Box::new(callee), values),
                        ty: ret_ty,
                    }
                }
                expr => unimplemented!("{:?}", expr),
            }
        }
        ast::Expression::Range(from, Some(to)) => {
            let from = deduce_expr_ty(from, arena, locals);
            let to = deduce_expr_ty(to, arena, locals);
            if !is_compatible_to(from.ty, to.ty) {
                log::debug!("incompatible range bounds");
                return TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) };
            }
            TypedExpression {
                expr: Expression::Range(
                    Box::new(from),
                    Some(Box::new(to)),
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
                let expr = deduce_expr_ty(value, arena, locals);
                types.push(expr.ty);
                values.push(expr);
            }

            TypedExpression { expr: Expression::Tuple(values), ty: arena.alloc(Type::Tuple(types)) }
        }
        ast::Expression::Index(arr, index_expr) => {
            let lhs = deduce_expr_ty(arr, arena, locals);
            let rhs = deduce_expr_ty(index_expr, arena, locals);

            let ty = match (lhs.ty, rhs.ty) {
                (Type::Array(_, item_ty), Type::I32) => item_ty,
                (Type::Slice(item_ty), Type::I32) => item_ty,
                _ => arena.alloc(Type::Error),
            };

            TypedExpression {
                expr: Expression::Index(Box::new(lhs), Box::new(rhs)),
                ty,
            }
        }
        ast::Expression::Var(_) => unreachable!(),
        ast::Expression::Cast(expr, ty) => {
            let expr = deduce_expr_ty(expr, arena, locals);
            let target_ty = unify(arena, ty);
            TypedExpression {
                expr: Expression::Cast(Box::new(expr), target_ty),
                ty: target_ty
            }
        }
    }
}

pub(crate) fn infer_types<'ast, 'tcx: 'ast>(
    items: &'ast [ast::Item],
    arena: &'tcx Arena<Type<'tcx>>,
    locals: &mut HashMap<&'ast str, TypeRef<'tcx>>,
    expected_ret_ty: Option<TypeRef<'tcx>>,
) -> Vec<Item<'tcx>> {
    let mut lowered_items = Vec::new();

    for item in items.iter() {
        let item = match item {
            ast::Item::Let { name, r#type: ty, expr } => {
                if expr.is_none() {
                    log::debug!("no expression on the right hand side of the let binding");
                    continue;
                }
                let expr = deduce_expr_ty(expr.as_ref().unwrap(), arena, &locals);
                log::debug!("deduced type {:?} for binding {}", expr.ty, name);
                let ty = match ty {
                    Some(ty) => {
                        let ty = unify(arena, ty);
                        if !is_compatible_to(ty, expr.ty) {
                            log::debug!("mismatched types. expected {:?}, got {:?}", ty, expr.ty);
                            continue;
                        }
                        ty
                    }
                    None => expr.ty,
                };
                locals.insert(name, ty);

                Item::Let { name: name.clone(), ty, expr: Some(expr) }
            }
            ast::Item::Assignment {
                lhs,
                operator,
                expr,
            } => {
                let lhs = deduce_expr_ty(lhs, arena, &locals);
                let rhs = deduce_expr_ty(expr, arena, &locals);

                if !is_compatible_to(lhs.ty, rhs.ty) {
                    log::debug!("incompatible types in assignment, got {:?} and {:?}", lhs.ty, rhs.ty);
                    continue;
                }

                Item::Assignment { lhs, operator: *operator, expr: rhs }
            }
            ast::Item::Expr { expr } => {
                let expr = deduce_expr_ty(expr, arena, locals);
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
                    let ty = unify(arena, &param.r#type);
                    log::debug!("Found arg {} of type {:?}", &param.name, ty);
                    locals.insert(param.name.as_str(), unify(arena, &param.r#type));
                    args.push(ty);
                }

                let func_ty = Type::Function(args, unify(arena, ty));
                let func_ty = arena.alloc(func_ty);
                locals.insert(name.as_str(), func_ty);

                let body = infer_types(body, arena, locals, Some(unify(arena, ty)));
                Item::Function {
                    name: name.clone(),
                    is_extern: false,
                    args: params.iter().map(|it| Argument { name: it.name.clone(), ty: unify(arena, &it.r#type) }).collect(),
                    ty: unify(arena, ty),
                    body,
                }
            }
            ast::Item::Struct { .. } => {
                log::info!("Skipping struct");
                continue;
            }
            ast::Item::If {
                condition,
                arm_true,
                arm_false,
            } => {
                let cond = deduce_expr_ty(condition, arena, &locals);
                if !is_compatible_to(cond.ty, arena.alloc(Type::Bool)) {
                    log::debug!("only boolean expressions are allowed in if conditions");
                    continue;
                }
                Item::If {
                    condition: cond,
                    arm_true: infer_types(arm_true, arena, locals, expected_ret_ty),
                    arm_false: if let Some(arm_false) = arm_false {
                        Some(infer_types(arm_false, arena, locals, expected_ret_ty))
                    } else {
                        None
                    },
                }
            }
            ast::Item::ForIn { name, expr, body } => {
                let expr = deduce_expr_ty(expr, arena, locals);
                let is_iterable = match expr.ty {
                    Type::Array(_, _) | Type::Slice(_) => true,
                    Type::Range => true,
                    _ => false,
                };
                if !is_iterable {
                    log::debug!("{:?} is not iterable", expr.ty);
                    continue;
                }
                locals.insert(name.as_str(), arena.alloc(Type::I32));
                let body = infer_types(body, arena, locals, expected_ret_ty);
                Item::ForIn {
                    name: name.clone(),
                    expr,
                    body,
                }
            }
            ast::Item::Loop { body } => {
                Item::Loop {
                    body: infer_types(body, arena, locals, expected_ret_ty)
                }
            }
            ast::Item::Return(expr) => {
                if expected_ret_ty.is_none() {
                    panic!("return outside of a function");
                }
                let expr = deduce_expr_ty(expr, arena, locals);
                if !is_compatible_to(expr.ty, expected_ret_ty.unwrap()) {
                    log::debug!("function marked as returning {:?} but returned {:?}",
                        expected_ret_ty.unwrap(),
                        expr.ty
                    );
                    continue;
                }
                Item::Return(Box::new(expr))
            }
            ast::Item::Break => {
                Item::Break
            }
            ast::Item::Yield(_) => unimplemented!(),
            ast::Item::Block(body) => {
                infer_types(body, arena, locals, expected_ret_ty);
                todo!()
            }
            ast::Item::Assert(expr) => {
                let expr = deduce_expr_ty(expr, arena, locals);
                Item::Assert(Box::new(expr))
            }
        };

        lowered_items.push(item);
    }

    lowered_items
}

fn unify<'tcx>(arena: &'tcx Arena<Type<'tcx>>, ty: &ast::Type) -> TypeRef<'tcx> {
    match ty {
        ast::Type::Name(name) => {
            match name.as_str() {
                "i32" => arena.alloc(Type::I32),
                "u32" => arena.alloc(Type::U32),
                "f32" => arena.alloc(Type::F32),
                "bool" => arena.alloc(Type::Bool),
                oth => unimplemented!("{:?}", oth),
            }
        }
        ast::Type::Tuple(types) => {
            let types: Vec<_> = types.iter()
                .map(|it| unify(arena, it))
                .collect();
            arena.alloc(Type::Tuple(types))
        }
        ast::Type::Pointer(ty) => arena.alloc(Type::Pointer(unify(arena, ty))),
        ast::Type::Array(len, ty) => arena.alloc(Type::Array(*len, unify(arena, ty))),
        ast::Type::Slice(item_ty) => arena.alloc(Type::Slice(unify(arena, item_ty))),
        ast::Type::Unit => arena.alloc(Type::Unit),
        ast::Type::Function(args_ty, ret_ty) => {
            let args = args_ty.iter().map(|it| unify(arena, it)).collect();
            arena.alloc(Type::Function(args, unify(arena, ret_ty)))
        }
    }
}