use std::collections::HashMap;
use crate::ast::{Item, Expression};
use crate::arena::Arena;
use crate::ty::{TyS, Ty};
use crate::lexer::TokenType;

fn are_types_compatible(a: Ty<'_>, b: Ty<'_>) -> bool {
    match (a, b) {
        (TyS::Bool, TyS::Bool) => true,
        (TyS::U32, TyS::U32) => true,
        (TyS::I32, TyS::I32) => true,
        (TyS::F32, TyS::F32) => true,
        (TyS::Array(len1, ty1), TyS::Array(len2, ty2)) => {
            len1 == len2 && are_types_compatible(ty1, ty2)
        }
        (TyS::Slice(ty1), TyS::Slice(ty2)) => are_types_compatible(ty1, ty2),
        (TyS::Unit, TyS::Unit) => true,
        (TyS::Tuple(ty1), TyS::Tuple(ty2)) => {
            ty1.len() == ty2.len()
                && ty1
                .iter()
                .zip(ty2.iter())
                .all(|(ty1, ty2)| are_types_compatible(ty1, ty2))
        }
        (TyS::Function(args1, ret1), TyS::Function(args2, ret2)) => {
            if args1.len() != args2.len() {
                return false;
            }

            if !are_types_compatible(ret1, ret2) {
                return false;
            }

            args1
                .iter()
                .zip(args2.iter())
                .all(|(ty1, ty2)| are_types_compatible(ty1, ty2))
        }
        (TyS::Pointer(ty1), TyS::Pointer(ty2)) => are_types_compatible(ty1, ty2),
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
            println!("{:?} {:?}", lhs_ty, rhs_ty);
            if !are_types_compatible(lhs_ty, rhs_ty) {
                return Err(format!("mismatched types {:?} and {:?}", lhs, rhs));
            }

            match op {
                TokenType::Punct(ch) => {
                    match ch {
                        '<' | '>' => return Ok(arena.alloc(TyS::Bool)),
                        '+' | '-' | '*' | '/' => return Ok(lhs_ty),
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
        }
        Expression::Prefix(_op, expr) => {
            deduce_expr_ty(expr, arena, &locals)?
        }
        Expression::Identifier(ident) => {
            dbg!(locals);

            if let Some(ty) = locals.get(ident.as_str()) {
                ty
            } else {
                return Err(format!("no local {:?}", ident));
            }
        }
        Expression::Place(expr, ty) => {
            dbg!(expr, ty);
            return Err("unsupported".into());
        }
        Expression::Array(items) => {
            if items.is_empty() {
                panic!("cant deduce type of array items");
            }

            let first = deduce_expr_ty(&items[0], arena, locals)?;
            for next in items.iter().skip(1) {
                let ty = deduce_expr_ty(next, arena, locals)?;
                if !are_types_compatible(ty, first) {
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
                    let callee = locals.get(ident.as_str()).expect("a type");
                    let (args_ty, ret_ty) = match callee {
                        TyS::Function(args_ty, ret_ty) => (args_ty, ret_ty),
                        _ => panic!("{} is not callable", ident.as_str())
                    };
                    for (arg, expected_ty) in args.iter().zip(args_ty) {
                        let arg_ty = deduce_expr_ty(arg, arena, locals)?;

                        if !are_types_compatible(arg_ty, expected_ty) {
                            panic!("incompatible types {:?} and {:?}", arg_ty, expected_ty);
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
            if !are_types_compatible(from_ty, to_ty) {
                return Err("incompatible range bounds".into());
            }
            arena.alloc(TyS::Range)
        }
        Expression::Range(to, None) => {
            let _to_ty = deduce_expr_ty(to, arena, locals)?;
            arena.alloc(TyS::Range)
        }
        e => {
            unimplemented!("{:?}", e);
        }
    })
}

pub(crate) fn infer_types<'ast, 'tcx: 'ast>(
    items: &'ast mut [Item<'tcx>],
    arena: &'tcx Arena<TyS<'tcx>>,
    locals: &mut HashMap<&'ast str, Ty<'tcx>>,
) -> Result<(), String> {
    for item in items.iter_mut() {
        match item {
            Item::Let { name, ty, expr } => {
                let expr = expr.as_ref().unwrap();

                let deduced_type = deduce_expr_ty(expr, arena, &locals)?;
                println!("deduced type {:?} for binding {}", deduced_type, name);

                match ty {
                    Some(ty) => {
                        if !are_types_compatible(ty, deduced_type) {
                            return Err(format!("mismatched types. expected {:?}, got {:?}", ty, deduced_type));
                        }
                    }
                    None => {
                        *ty = Some(deduced_type);
                    }
                }
                locals.insert(name, ty.expect("has type"));
            }
            Item::Assignment { lhs, operator: _, expr } => {
                let lhs_expr = deduce_expr_ty(lhs, arena, &locals)?;
                let rhs_expr = deduce_expr_ty(expr, arena, &locals)?;

                assert!(are_types_compatible(lhs_expr, rhs_expr));
            }
            Item::Expr { expr } => {
                deduce_expr_ty(expr, arena, locals)?;
            }
            Item::Function { name, args, ty, body } => {
                let func_ty = TyS::Function(
                    args.iter().map(|it| it.ty).collect(),
                    ty.unwrap_or(arena.alloc(TyS::Unit)),
                );
                let func_ty = arena.alloc(func_ty);
                locals.insert(name.as_str(), func_ty);
                for arg in args.iter() {
                    dbg!("inserting", &arg.name, arg.ty);
                    locals.insert(arg.name.as_str(), arg.ty);
                }

                infer_types(body, arena, locals)?;
            }
            Item::Struct { .. } => {
                dbg!("ifnore struct");
            }
            Item::If { condition, arm_true: _, arm_false: _ } => {
                let cond_ty = deduce_expr_ty(condition, arena, &locals)?;
                assert!(are_types_compatible(cond_ty, arena.alloc(TyS::Bool)));
            }
            Item::ForIn { name, expr, body } => {
                let ty = deduce_expr_ty(expr, arena, locals)?;
                let is_iterable = match ty {
                    TyS::Array(_, _) | TyS::Slice(_) => true,
                    TyS::Range => true,
                    _ => false,
                };
                assert!(is_iterable);
                locals.insert(name.as_str(), arena.alloc(TyS::I32));
                infer_types(body, arena, locals)?;
            }
            Item::Loop { .. } => {
                dbg!("ifnore loop");
            }
            Item::Return(_) => {
                dbg!("ifnore return");
            }
            other => {
                unimplemented!("cant typeck {:?}", &other);
            }
        }
    }
    Ok(())
}

