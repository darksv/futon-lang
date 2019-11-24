use std::collections::HashMap;
use crate::ast::{Item, Expression, Operator};
use crate::arena::Arena;
use crate::ty::{TyS, Ty};

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
    e: &Expression,
    arena: &'tcx Arena<TyS<'tcx>>,
    locals: &HashMap<&str, Ty<'tcx>>,
) -> Result<Ty<'tcx>, String> {
    Ok(match e {
        Expression::Integer(_) => arena.alloc(TyS::I32),
        Expression::Float(_) => arena.alloc(TyS::F32),
        Expression::Bool(_) => arena.alloc(TyS::Bool),
        Expression::Infix(op, lhs, rhs) => {
            let lhs_ty = deduce_expr_ty(lhs, arena, &locals)?;
            let rhs_ty = deduce_expr_ty(rhs, arena, &locals)?;
            log::debug!("{:?} {:?}", lhs_ty, rhs_ty);
            if !is_compatible_to(lhs_ty, rhs_ty) {
                return Err(format!("mismatched types {:?} and {:?}", lhs_ty, rhs_ty));
            }

            match op {
                | Operator::Less
                | Operator::LessEqual
                | Operator::Greater
                | Operator::GreaterEqual
                | Operator::Equal
                | Operator::NotEqual => return Ok(arena.alloc(TyS::Bool)),
                | Operator::Add
                | Operator::Sub
                | Operator::Mul
                | Operator::Div => return Ok(lhs_ty),
                | Operator::Negate => unimplemented!(),
                | Operator::Deref => unimplemented!(),
            }
        }
        Expression::Prefix(_op, expr) => {
            deduce_expr_ty(expr, arena, &locals)?
        }
        Expression::Identifier(ident) => {
            if let Some(ty) = locals.get(ident.as_str()) {
                ty
            } else {
                return Err(format!("no local {:?}", ident));
            }
        }
        Expression::Place(expr, ty) => {
            log::debug!("{:?} {:?}", expr, ty);
            return Err("unsupported".into());
        }
        Expression::Array(items) => {
            if items.is_empty() {
                panic!("cant deduce type of array items");
            }

            let first = deduce_expr_ty(&items[0], arena, locals)?;
            for next in items.iter().skip(1) {
                let ty = deduce_expr_ty(next, arena, locals)?;
                if !is_compatible_to(ty, first) {
                    return Err("incompatible".into());
                }
            }

            arena.alloc(TyS::Array(items.len(), first))
        }
        Expression::Call(callee, args) => {
            match callee.as_ref() {
                Expression::Identifier(ident) if ident == "range" => {
                    arena.alloc(TyS::Range)
                }
                Expression::Identifier(ident) => {
                    let callee = locals.get(ident.as_str()).expect(&format!("a type for {}", ident.as_str()));
                    let (args_ty, ret_ty) = match callee {
                        TyS::Function(args_ty, ret_ty) => (args_ty, ret_ty),
                        _ => return Err(format!("{} is not callable", ident.as_str()))
                    };
                    for (arg, expected_ty) in args.iter().zip(args_ty) {
                        let arg_ty = deduce_expr_ty(arg, arena, locals)?;

                        if !is_compatible_to(arg_ty, expected_ty) {
                            return Err(format!("incompatible types {:?} and {:?}", arg_ty, expected_ty));
                        }
                    }

                    ret_ty
                }
                expr => unimplemented!("{:?}", expr),
            }
        }
        Expression::Range(from, Some(to)) => {
            let from_ty = deduce_expr_ty(from, arena, locals)?;
            let to_ty = deduce_expr_ty(to, arena, locals)?;
            if !is_compatible_to(from_ty, to_ty) {
                return Err("incompatible range bounds".into());
            }
            arena.alloc(TyS::Range)
        }
        Expression::Range(to, None) => {
            let _to_ty = deduce_expr_ty(to, arena, locals)?;
            arena.alloc(TyS::Range)
        }
        Expression::Tuple(values) => {
            let types: Result<Vec<_>, _> = values
                .iter()
                .map(|v| deduce_expr_ty(v, arena, locals))
                .collect();
            arena.alloc(TyS::Tuple(types?))
        }
    })
}

pub(crate) fn infer_types<'ast, 'tcx: 'ast>(
    items: &'ast mut [Item<'tcx>],
    arena: &'tcx Arena<TyS<'tcx>>,
    locals: &mut HashMap<&'ast str, Ty<'tcx>>,
    expected_ret_ty: Option<Ty<'tcx>>,
) -> Result<(), String> {
    for item in items.iter_mut() {
        match item {
            Item::Let { name, ty, expr } => {
                if expr.is_none() {
                    return Err(format!("no expression on the right hand side of the let binding"));
                }
                let expr = expr.as_ref().unwrap();
                let expr_ty = deduce_expr_ty(expr, arena, &locals)?;
                log::debug!("deduced type {:?} for binding {}", expr_ty, name);
                let ty = match ty {
                    Some(ty) => {
                        if !is_compatible_to(ty, expr_ty) {
                            return Err(format!("mismatched types. expected {:?}, got {:?}", ty, expr_ty));
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
            Item::Assignment { lhs, operator: _, expr } => {
                let lhs_ty = deduce_expr_ty(lhs, arena, &locals)?;
                let rhs_ty = deduce_expr_ty(expr, arena, &locals)?;

                if !is_compatible_to(lhs_ty, rhs_ty) {
                    return Err(format!("incompatible types in assignment, got {:?} and {:?}", lhs_ty, rhs_ty));
                }
            }
            Item::Expr { expr } => {
                deduce_expr_ty(expr, arena, locals)?;
            }
            Item::Function { name, args, ty, body, .. } => {
                if ty.is_none() {
                    *ty = Some(arena.alloc(TyS::Unit));
                }
                let func_ty = TyS::Function(
                    args.iter().map(|it| it.ty).collect(),
                    ty.unwrap(),
                );
                let func_ty = arena.alloc(func_ty);
                locals.insert(name.as_str(), func_ty);
                for arg in args {
                    log::debug!("Found arg {} of type {:?}", &arg.name, arg.ty);
                    locals.insert(arg.name.as_str(), arg.ty);
                }

                infer_types(body, arena, locals, *ty)?;
            }
            Item::Struct { .. } => {
                log::debug!("ifnore struct");
            }
            Item::If { condition, arm_true, arm_false } => {
                let cond_ty = deduce_expr_ty(condition, arena, &locals)?;
                if !is_compatible_to(cond_ty, arena.alloc(TyS::Bool)) {
                    return Err(format!("only boolean expressions are allowed in if conditions"));
                }
                infer_types(arm_true, arena, locals, expected_ret_ty)?;
                if let Some(arm_false) = arm_false {
                    infer_types(arm_false, arena, locals, expected_ret_ty)?;
                }
            }
            Item::ForIn { name, expr, body } => {
                let ty = deduce_expr_ty(expr, arena, locals)?;
                let is_iterable = match ty {
                    TyS::Array(_, _) | TyS::Slice(_) => true,
                    TyS::Range => true,
                    _ => false,
                };
                if !is_iterable {
                    return Err(format!("{:?} is not iterable", ty));
                }
                locals.insert(name.as_str(), arena.alloc(TyS::I32));
                infer_types(body, arena, locals, expected_ret_ty)?;
            }
            Item::Loop { body } => {
                infer_types(body, arena, locals, expected_ret_ty)?;
            }
            Item::Return(expr) => {
                if expected_ret_ty.is_none() {
                    panic!("return outside of a function");
                }
                let ty = deduce_expr_ty(expr, arena, locals)?;
                if !is_compatible_to(ty, expected_ret_ty.unwrap()) {
                    return Err(format!("function marked as returning {:?} but returned {:?}", expected_ret_ty.unwrap(), ty));
                }
            }
            Item::Break => {}
            Item::Yield(_) => unimplemented!()
        }
    }
    Ok(())
}

