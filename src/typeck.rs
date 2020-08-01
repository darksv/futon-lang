use crate::arena::Arena;
use crate::ast::{Expr, Item, Operator, TyExpr};
use crate::ty::{Ty, TyS};
use std::collections::HashMap;

fn is_compatible_to(ty: Ty<'_>, subty: Ty<'_>) -> bool {
    match (ty, subty) {
        (TyS::Bool, TyS::Bool) => true,
        (TyS::U32, TyS::U32) => true,
        (TyS::I32, TyS::I32) => true,
        (TyS::F32, TyS::F32) => true,
        (TyS::Array(len1, ty1), TyS::Array(len2, ty2)) => {
            len1 == len2 && is_compatible_to(ty1, ty2)
        }
        (TyS::Array(_, ty1), TyS::Slice(ty2)) => is_compatible_to(ty1, ty2),
        (TyS::Slice(ty1), TyS::Slice(ty2)) => is_compatible_to(ty1, ty2),
        (TyS::Unit, TyS::Unit) => true,
        (TyS::Tuple(ty1), TyS::Tuple(ty2)) => {
            ty1.len() == ty2.len()
                && ty1
                .iter()
                .zip(ty2.iter())
                .all(|(ty1, ty2)| is_compatible_to(ty1, ty2))
        }
        (TyS::Function(args1, ret1), TyS::Function(args2, ret2)) => {
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
        (TyS::Pointer(ty1), TyS::Pointer(ty2)) => is_compatible_to(ty1, ty2),
        (TyS::Other(name1), TyS::Other(name2)) => name1 == name2,
        _ => false,
    }
}

fn deduce_expr_ty<'tcx>(
    expr: &mut TyExpr<'tcx>,
    arena: &'tcx Arena<TyS<'tcx>>,
    locals: &HashMap<&str, Ty<'tcx>>,
) -> Ty<'tcx> {
    let ty = match &mut expr.expr {
        Expr::Integer(_) => arena.alloc(TyS::I32),
        Expr::Float(_) => arena.alloc(TyS::F32),
        Expr::Bool(_) => arena.alloc(TyS::Bool),
        Expr::Infix(op, lhs, rhs) => {
            let lhs_ty = deduce_expr_ty(lhs, arena, &locals);
            let rhs_ty = deduce_expr_ty(rhs, arena, &locals);
            log::debug!("{:?} {:?}", lhs_ty, rhs_ty);
            if !is_compatible_to(lhs_ty, rhs_ty) {
                log::debug!("mismatched types {:?} and {:?}", lhs_ty, rhs_ty);
                return arena.alloc(TyS::Error);
            }

            match op {
                Operator::Less
                | Operator::LessEqual
                | Operator::Greater
                | Operator::GreaterEqual
                | Operator::Equal
                | Operator::NotEqual => arena.alloc(TyS::Bool),
                Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => lhs_ty,
                Operator::Negate => unimplemented!(),
                Operator::Ref => unimplemented!(),
                Operator::Deref => unimplemented!(),
            }
        }
        Expr::Prefix(op, expr) => {
            let inner = deduce_expr_ty(expr, arena, &locals);
            match op {
                Operator::Ref => arena.alloc(TyS::Pointer(inner)),
                Operator::Deref => unimplemented!(),
                _ => inner,
            }
        }
        Expr::Identifier(ident) => {
            if let Some(ty) = locals.get(ident.as_str()) {
                ty
            } else {
                log::debug!("no local {:?}", ident);
                return arena.alloc(TyS::Error);
            }
        }
        Expr::Place(expr, ty) => {
            log::debug!("unsupported place expr");
            return arena.alloc(TyS::Error);
        }
        Expr::Array(items) => {
            if items.is_empty() {
                return arena.alloc(TyS::Unknown);
            }

            let first = deduce_expr_ty(&mut items[0], arena, locals);
            for next in items.iter_mut().skip(1) {
                let ty = deduce_expr_ty(next, arena, locals);
                if !is_compatible_to(ty, first) {
                    log::debug!("incompatible types: {:?} and {:?}", ty, first);
                    return arena.alloc(TyS::Error);
                }
            }

            arena.alloc(TyS::Array(items.len(), first))
        }
        Expr::Call(callee, args) => match &mut callee.as_mut().expr {
            Expr::Identifier(ident) => {
                let callee = locals
                    .get(ident.as_str())
                    .expect(&format!("a type for {}", ident.as_str()));
                let (args_ty, ret_ty) = match callee {
                    TyS::Function(args_ty, ret_ty) => (args_ty, ret_ty),
                    _ => {
                        log::debug!("{} is not callable", ident.as_str());
                        return arena.alloc(TyS::Error);
                    },
                };
                for (arg, expected_ty) in args.iter_mut().zip(args_ty) {
                    let arg_ty = deduce_expr_ty(arg, arena, locals);

                    if !is_compatible_to(arg_ty, expected_ty) {
                        log::debug!("incompatible types {:?} and {:?}",
                            arg_ty, expected_ty);
                        return arena.alloc(TyS::Error);
                    }
                }

                ret_ty
            }
            expr => unimplemented!("{:?}", expr),
        },
        Expr::Range(from, Some(to)) => {
            let from_ty = deduce_expr_ty(from, arena, locals);
            let to_ty = deduce_expr_ty(to, arena, locals);
            if !is_compatible_to(from_ty, to_ty) {
                log::debug!("incompatible range bounds");
                return arena.alloc(TyS::Error);
            }
            arena.alloc(TyS::Range)
        }
        Expr::Range(to, None) => {
            let _to_ty = deduce_expr_ty(to, arena, locals);
            arena.alloc(TyS::Range)
        }
        Expr::Tuple(values) => {
            let types: Vec<_> = values
                .iter_mut()
                .map(|v| deduce_expr_ty(v, arena, locals))
                .collect();
            arena.alloc(TyS::Tuple(types))
        }
        Expr::Index(arr, index_expr) => {
            deduce_expr_ty(arr, arena, locals);
            deduce_expr_ty(index_expr, arena, locals);
            match arr.ty {
                TyS::Array(length, ty) => ty,
                other @ _ => unimplemented!("{:?}", &other),
            }
        }
    };
    expr.ty = ty;
    ty
}

pub(crate) fn infer_types<'ast, 'tcx: 'ast>(
    items: &'ast mut [Item<'tcx>],
    arena: &'tcx Arena<TyS<'tcx>>,
    locals: &mut HashMap<&'ast str, Ty<'tcx>>,
    expected_ret_ty: Option<Ty<'tcx>>,
) {
    for item in items.iter_mut() {
        match item {
            Item::Let { name, ty, expr } => {
                if expr.is_none() {
                    log::debug!("no expression on the right hand side of the let binding");
                    continue;
                }
                let expr = expr.as_mut().unwrap();
                let expr_ty = deduce_expr_ty(expr, arena, &locals);
                log::debug!("deduced type {:?} for binding {}", expr_ty, name);
                let ty = match ty {
                    Some(ty) => {
                        if !is_compatible_to(ty, expr_ty) {
                            log::debug!("mismatched types. expected {:?}, got {:?}", ty, expr_ty);
                            continue;
                        }
                        ty
                    }
                    None => {
                        *ty = Some(expr_ty);
                        expr_ty
                    }
                };
                locals.insert(name, ty);
            }
            Item::Assignment {
                lhs,
                operator: _,
                expr,
            } => {
                let lhs_ty = deduce_expr_ty(lhs, arena, &locals);
                let rhs_ty = deduce_expr_ty(expr, arena, &locals);

                if !is_compatible_to(lhs_ty, rhs_ty) {
                    log::debug!("incompatible types in assignment, got {:?} and {:?}", lhs_ty, rhs_ty);
                    continue;
                }
            }
            Item::Expr { expr } => {
                deduce_expr_ty(expr, arena, locals);
            }
            Item::Function {
                name,
                args,
                ty,
                body,
                ..
            } => {
                let func_ty = TyS::Function(args.iter().map(|it| it.ty).collect(), ty);
                let func_ty = arena.alloc(func_ty);
                locals.insert(name.as_str(), func_ty);
                for arg in args {
                    log::debug!("Found arg {} of type {:?}", &arg.name, arg.ty);
                    locals.insert(arg.name.as_str(), arg.ty);
                }

                infer_types(body, arena, locals, Some(*ty));
            }
            Item::Struct { .. } => {
                log::debug!("ifnore struct");
            }
            Item::If {
                condition,
                arm_true,
                arm_false,
            } => {
                let cond_ty = deduce_expr_ty(condition, arena, &locals);
                if !is_compatible_to(cond_ty, arena.alloc(TyS::Bool)) {
                    log::debug!("only boolean expressions are allowed in if conditions");
                    continue;
                }
                infer_types(arm_true, arena, locals, expected_ret_ty);
                if let Some(arm_false) = arm_false {
                    infer_types(arm_false, arena, locals, expected_ret_ty);
                }
            }
            Item::ForIn { name, expr, body } => {
                let ty = deduce_expr_ty(expr, arena, locals);
                let is_iterable = match ty {
                    TyS::Array(_, _) | TyS::Slice(_) => true,
                    TyS::Range => true,
                    _ => false,
                };
                if !is_iterable {
                    log::debug!("{:?} is not iterable", ty);
                    continue;
                }
                locals.insert(name.as_str(), arena.alloc(TyS::I32));
                infer_types(body, arena, locals, expected_ret_ty);
            }
            Item::Loop { body } => {
                infer_types(body, arena, locals, expected_ret_ty);
            }
            Item::Return(expr) => {
                if expected_ret_ty.is_none() {
                    panic!("return outside of a function");
                }
                let ty = deduce_expr_ty(expr, arena, locals);
                if !is_compatible_to(ty, expected_ret_ty.unwrap()) {
                    log::debug!(
                        "function marked as returning {:?} but returned {:?}",
                        expected_ret_ty.unwrap(),
                        ty
                    );
                    continue;
                }
            }
            Item::Break => {}
            Item::Yield(_) => unimplemented!(),
            Item::Block(body) => {
                infer_types(body, arena, locals, expected_ret_ty);
            }
        }
    }
}
