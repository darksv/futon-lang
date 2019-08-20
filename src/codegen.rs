use ast::{Expression, Item};
use lexer::TokenType;
use parser::Ty;
use std::borrow::{Borrow, Cow};
use std::collections::HashMap;

struct VariableHolder {
    variables: HashMap<String, Vec<(Ty, Option<Expression>)>>,
}

impl VariableHolder {
    fn new() -> Self {
        VariableHolder {
            variables: Default::default(),
        }
    }

    fn def<T: Into<String> + Borrow<str>>(&mut self, name: T, ty: Ty, val: Option<Expression>) {
        self.variables
            .entry(name.into())
            .or_insert_with(Vec::new)
            .push((ty.clone(), val.clone()));
    }

    #[allow(unused)]
    fn undef<T: Into<String> + Borrow<str>>(&mut self, name: T) -> bool {
        if let Some(x) = self.variables.get_mut(name.borrow()) {
            x.pop().is_some()
        } else {
            false
        }
    }

    fn get<T: Into<String> + Borrow<str>>(&mut self, name: T) -> Option<&(Ty, Option<Expression>)> {
        self.variables.get(name.borrow()).and_then(|it| it.last())
    }
}

pub fn genc(items: &[Item], n: usize) {
    println!(r"#include <stdint.h>");
    println!(r"typedef struct Slice {{ void* ptr; uint64_t len; }};");

    let mut vars = VariableHolder::new();
    for item in items {
        indent(n);
        genc_item(item, n, &mut vars);
    }
}

fn genc_item(item: &Item, ind: usize, vars: &mut VariableHolder) {
    match item {
        Item::Let { name, ty, expr, .. } => {
            indent(ind);
            println!(
                "{} {} = {};",
                format_ty(ty.as_ref().unwrap()),
                name,
                format_expr(expr.as_ref().unwrap())
            );
            vars.def(name.clone(), ty.clone().unwrap(), expr.clone());
        }
        Item::Assignment {
            lhs,
            operator,
            expr,
        } => {
            indent(ind);
            if let Some(op) = operator {
                println!("{} {}= {};", format_expr(lhs), op, format_expr(expr));
            } else {
                println!("{} = {};", format_expr(lhs), format_expr(expr));
            }
        }
        Item::Function {
            name,
            args,
            ty,
            body,
            ..
        } => {
            indent(ind);
            print!("{} {}(", format_ty(ty.as_ref().unwrap()), name);
            if args.is_empty() {
                print!("void");
            } else {
                for (i, arg) in args.iter().enumerate() {
                    print!("{} {}", format_ty(&arg.ty), arg.name);
                    if i != args.len() - 1 {
                        print!(", ")
                    }

                    vars.def(arg.name.clone(), arg.ty.clone(), None);
                }
            }

            println!(") {{");
            for b in body {
                genc_item(b, ind + 1, vars);
            }
            indent(ind);
            println!("}}");
        }
        Item::If {
            condition,
            arm_true,
            arm_false,
        } => {
            indent(ind);
            println!("if ({}) {{", format_expr(condition));
            for item in arm_true {
                genc_item(item, ind + 1, vars);
            }
            if let Some(arm_false) = arm_false {
                indent(ind);
                println!("}} else {{");
                for item in arm_false {
                    genc_item(item, ind + 1, vars);
                }
            }
            indent(ind);
            println!("}}");
        }
        Item::ForIn {
            name: bound,
            expr,
            body,
        } => match expr {
            Expression::Identifier(name) => {
                let (ty, _expr) = vars.get(name.clone()).expect(&name);
                let (n, ty) = match ty {
                    Ty::Array(n, ty) => (Some(n), &**ty),
                    Ty::Slice(ty) => (None, &**ty),
                    Ty::U32 => (Some(&10), ty),
                    other => {
                        dbg!(other);
                        unimplemented!()
                    }
                };

                if let Some(n) = n {
                    indent(ind);
                    println!("for (int64_t i = 0; i < {}; ++i) {{", n);
                    indent(ind + 1);
                    println!("{} {} = {}[i];", format_ty(ty), bound, name);
                } else {
                    indent(ind);
                    println!("for (int64_t i = 0; i < {}.len; ++i) {{", name);
                    indent(ind + 1);
                    println!(
                        "{} {} = ((*{}){}.ptr)[i];",
                        format_ty(ty),
                        bound,
                        format_ty(ty),
                        name
                    );
                }

                for item in body {
                    genc_item(item, ind + 1, vars);
                }
                indent(ind);
                println!("}}");
            }
            _ => unimplemented!(),
        },
        Item::Break => {}
        Item::Yield(_) => {}
        Item::Return(expr) => {
            indent(ind);
            println!("return {};", format_expr(expr));
        }
    }
}

fn format_ty(ty: &Ty) -> Cow<'static, str> {
    match ty {
        Ty::U32 => Cow::Borrowed("uint32_t"),
        Ty::I32 => Cow::Borrowed("int32_t"),
        Ty::Array(len, ty) => Cow::Owned(format!("{}[{}]", format_ty(&**ty), len)),
        Ty::Slice(_ty) => Cow::Owned(format!("Slice")),
        Ty::Other(_) => Cow::Borrowed("/* generated */"),
        Ty::Tuple(tys) => {
            if tys.is_empty() {
                Cow::Borrowed("void")
            } else {
                Cow::Borrowed("/* generated */")
            }
        }
        Ty::Function(..) => Cow::Borrowed("void*"),
        Ty::Pointer(inner) => Cow::Owned(format!("{}*", format_ty(&**inner))),
        Ty::Bool => Cow::Borrowed("bool"),
    }
}

fn format_operator(op: &TokenType) -> Cow<'static, str> {
    match op {
        TokenType::Punct(c) => Cow::Owned(c.to_string()),
        _ => Cow::Borrowed(""),
    }
}

fn format_expr(ty: &Expression) -> Cow<str> {
    match ty {
        Expression::Identifier(x) => Cow::Borrowed(x.as_str()),
        Expression::Integer(value) => Cow::Owned(value.to_string()),
        Expression::Float(value) => Cow::Owned(value.to_string()),
        Expression::Prefix(op, expr) => {
            Cow::Owned(format!("{}{}", format_operator(op), format_expr(&**expr)))
        }
        Expression::Infix(op, lhs, rhs) => Cow::Owned(format!(
            "({} {} {})",
            format_expr(&**lhs),
            format_operator(op),
            format_expr(&**rhs)
        )),
        Expression::Array(values) => {
            let mut s = String::new();
            s.push('{');
            for (i, value) in values.iter().enumerate() {
                s.push_str(&*format_expr(value));
                if i != values.len() - 1 {
                    s.push_str(", ");
                }
            }
            s.push('}');
            Cow::Owned(s)
        }
        Expression::Call(expr, args) => {
            let mut s = String::new();
            s.push('(');
            s.push_str(&format_expr(expr));
            s.push(')');
            s.push('(');
            s.push_str(&format_expr(&args[0]));
            s.push(')');

            Cow::Owned(s)
        }
        Expression::Tuple(_) => Cow::Borrowed("/* generated */"),
        Expression::Bool(value) => Cow::Borrowed(if *value { "true" } else { "false" }),
    }
}

fn indent(n: usize) {
    print!("{:width$}", "", width = n * 2);
}
