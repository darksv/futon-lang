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
pub(crate) struct TypedExpression<'tcx> {
    pub(crate) ty: TypeRef<'tcx>,
    pub(crate) expr: Expression<'tcx>,
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
    StructLiteral(Vec<TypedExpression<'tcx>>),
    Range(Box<TypedExpression<'tcx>>, Option<Box<TypedExpression<'tcx>>>),
    Cast(Box<TypedExpression<'tcx>>, TypeRef<'tcx>),
    Field(Box<TypedExpression<'tcx>>, usize),
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
    defined_types: &HashMap<&str, TypeRef<'tcx>>,
) -> TypedExpression<'tcx> {
    match expr {
        ast::Expression::Integer(val) => TypedExpression { expr: Expression::Integer(*val), ty: arena.alloc(Type::I32) },
        ast::Expression::Float(val) => TypedExpression { expr: Expression::Float(*val), ty: arena.alloc(Type::F32) },
        ast::Expression::Bool(val) => TypedExpression { expr: Expression::Bool(*val), ty: arena.alloc(Type::Bool) },
        ast::Expression::Infix(op, lhs, rhs) => {
            let lhs = deduce_expr_ty(lhs, arena, &locals, defined_types);
            let rhs = deduce_expr_ty(rhs, arena, &locals, defined_types);
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
            let inner = deduce_expr_ty(expr, arena, &locals, defined_types);
            let ty = match op {
                ast::Operator::Ref => arena.alloc(Type::Pointer(inner.ty)),
                ast::Operator::Deref => {
                    match inner.ty {
                        Type::Pointer(inner) => inner,
                        _ => unimplemented!(),
                    }
                }
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
        ast::Expression::Place(expr, field) => {
            let lhs = deduce_expr_ty(expr, arena, &locals, defined_types);

            let Some(name) = field.as_str() else {
                unimplemented!()
            };

            let (idx, (_, ty)) = match &lhs.ty {
                Type::Struct { fields } => fields.iter().enumerate().find(|(idx, (n, _))| n == name).unwrap(),
                _ => unimplemented!(),
            };

            TypedExpression {
                ty,
                expr: Expression::Field(Box::new(lhs), idx),
            }
        }
        ast::Expression::Array(items) => {
            if items.is_empty() {
                return TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Unknown) };
            }

            let mut values = Vec::new();

            let first = deduce_expr_ty(&items[0], arena, locals, defined_types);
            let item_ty = first.ty;
            values.push(first);

            for next in items.iter().skip(1) {
                let expr = deduce_expr_ty(next, arena, locals, defined_types);
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
            match callee.as_ref() {
                ast::Expression::Identifier(ident) => {
                    let callee = match ident.as_str() {
                        "debug" => {
                            let ty = arena.alloc(Type::Function(vec![&Type::Any], &Type::Any));
                            TypedExpression { ty, expr: Expression::Intrinsic(Intrinsic::debug) }
                        }
                        other => {
                            let ty = locals.get(other).unwrap_or_else(|| panic!("a type for {}", other));
                            deduce_expr_ty(callee, arena, locals, defined_types)
                        }
                    };

                    let (args_ty, ret_ty) = match callee.ty {
                        Type::Function(args_ty, ret_ty) => (args_ty, ret_ty),
                        _ => {
                            log::debug!("{} is not callable", ident.as_str());
                            return TypedExpression { expr: Expression::Error, ty: arena.alloc(Type::Error) };
                        }
                    };

                    let mut values = Vec::new();

                    for (arg, expected_ty) in args.iter().zip(args_ty) {
                        let arg = deduce_expr_ty(arg, arena, locals, defined_types);

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
            let from = deduce_expr_ty(from, arena, locals, defined_types);
            let to = deduce_expr_ty(to, arena, locals, defined_types);
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
                let expr = deduce_expr_ty(value, arena, locals, defined_types);
                types.push(expr.ty);
                values.push(expr);
            }

            TypedExpression { expr: Expression::Tuple(values), ty: arena.alloc(Type::Tuple(types)) }
        }
        ast::Expression::Index(arr, index_expr) => {
            let lhs = deduce_expr_ty(arr, arena, locals, defined_types);
            let rhs = deduce_expr_ty(index_expr, arena, locals, defined_types);

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
            let expr = deduce_expr_ty(expr, arena, locals, defined_types);
            let target_ty = unify(ty, arena, defined_types);
            TypedExpression {
                expr: Expression::Cast(Box::new(expr), target_ty),
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
                .map(|(name, expr)| deduce_expr_ty(expr, arena, locals, defined_types))
                .collect();
            TypedExpression {
                expr: Expression::StructLiteral(fields),
                ty,
            }
        }
    }
}

pub(crate) fn infer_types<'ast, 'tcx: 'ast>(
    items: &'ast [ast::Item],
    arena: &'tcx Arena<Type<'tcx>>,
    locals: &mut HashMap<&'ast str, TypeRef<'tcx>>,
    expected_ret_ty: Option<TypeRef<'tcx>>,
    defined_types: &mut HashMap<&'ast str, TypeRef<'tcx>>,
) -> Vec<Item<'tcx>> {
    let mut lowered_items = Vec::new();

    for item in items.iter() {
        let item = match item {
            ast::Item::Let { name, r#type: expected_ty, expr } => {
                if expr.is_none() {
                    log::debug!("no expression on the right hand side of the let binding");
                    continue;
                }
                let expr = deduce_expr_ty(expr.as_ref().unwrap(), arena, &locals, defined_types);
                log::debug!("deduced type {:?} for binding {}", expr.ty, name);
                let ty = match expected_ty {
                    Some(expected) => {
                        let ty = unify(expected, arena, defined_types);
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
                let lhs = deduce_expr_ty(lhs, arena, &locals, defined_types);
                let rhs = deduce_expr_ty(expr, arena, &locals, defined_types);

                if !is_compatible_to(lhs.ty, rhs.ty) {
                    log::debug!("incompatible types in assignment, got {:?} and {:?}", lhs.ty, rhs.ty);
                    continue;
                }

                Item::Assignment { lhs, operator: *operator, expr: rhs }
            }
            ast::Item::Expr { expr } => {
                let expr = deduce_expr_ty(expr, arena, locals, defined_types);
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

                let body = infer_types(body, arena, locals, Some(unify(ty, arena, defined_types)), defined_types);
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
                let cond = deduce_expr_ty(condition, arena, &locals, defined_types);
                if !is_compatible_to(cond.ty, arena.alloc(Type::Bool)) {
                    log::debug!("only boolean expressions are allowed in if conditions");
                    continue;
                }
                Item::If {
                    condition: cond,
                    arm_true: infer_types(arm_true, arena, locals, expected_ret_ty, defined_types),
                    arm_false: if let Some(arm_false) = arm_false {
                        Some(infer_types(arm_false, arena, locals, expected_ret_ty, defined_types))
                    } else {
                        None
                    },
                }
            }
            ast::Item::ForIn { name, expr, body } => {
                let expr = deduce_expr_ty(expr, arena, locals, defined_types);
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
                let body = infer_types(body, arena, locals, expected_ret_ty, defined_types);
                Item::ForIn {
                    name: name.clone(),
                    expr,
                    body,
                }
            }
            ast::Item::Loop { body } => {
                Item::Loop {
                    body: infer_types(body, arena, locals, expected_ret_ty, defined_types)
                }
            }
            ast::Item::Return(expr) => {
                if expected_ret_ty.is_none() {
                    panic!("return outside of a function");
                }
                let expr = deduce_expr_ty(expr, arena, locals, defined_types);
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
                infer_types(body, arena, locals, expected_ret_ty, defined_types);
                todo!()
            }
            ast::Item::Assert(expr) => {
                let expr = deduce_expr_ty(expr, arena, locals, defined_types);
                Item::Assert(Box::new(expr))
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