use ast::{Expression, Item, Ty};
use lexer::TokenType;
use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use parser::TyS;


struct VariableHolder<'tcx> {
    variables: HashMap<String, Vec<(Ty<'tcx>, Option<Expression>)>>,
}

impl<'tcx> VariableHolder<'tcx> {
    fn new() -> Self {
        VariableHolder {
            variables: Default::default(),
        }
    }

    fn def<T: Into<String> + Borrow<str>>(&mut self, name: T, ty: Ty<'tcx>, val: Option<Expression>) {

        self.variables
            .entry(name.into())
            .or_insert_with(Vec::new)
            .push((ty, val.clone()));
    }

    #[allow(unused)]
    fn undef<T: Into<String> + Borrow<str>>(&mut self, name: T) -> bool {
        if let Some(x) = self.variables.get_mut(name.borrow()) {
            x.pop().is_some()
        } else {
            false
        }
    }

    fn get<T: Into<String> + Borrow<str>>(&mut self, name: T) -> Option<&(Ty<'tcx>, Option<Expression>)> {
        self.variables.get(name.borrow()).and_then(|it| it.last())
    }
}


pub(crate) struct SourceBuilder {
    buffer: String,
    level: usize,
    indented: bool,
}

impl SourceBuilder {
    pub(crate) fn new() -> Self {
        SourceBuilder {
            buffer: String::new(),
            level: 0,
            indented: false,
        }
    }

    pub(crate) fn build(self) -> String {
        self.buffer
    }

    fn write(&mut self, s: &str) {
        self.maybe_indent();
        self.buffer.push_str(s);
    }

    fn maybe_indent(&mut self) {
        if !self.indented {
            self.buffer.push_str(&format!("{:width$}", "", width = self.level * 2));
            self.indented = true;
        }
    }

    fn writeln(&mut self, s: &str) {
        self.maybe_indent();
        self.buffer.push_str(s);
        self.buffer.push('\n');
        self.indented = false;
    }

    fn shift(&mut self) {
        self.level += 1;
    }

    fn unshift(&mut self) {
        self.level -= 1;
    }
}

pub(crate) fn genc(fmt: &mut SourceBuilder, items: &[Item]) {
    fmt.writeln(r"#include <stdint.h>");
    fmt.writeln(r"typedef struct Slice { void* ptr; uint64_t len; };");

    let mut vars = VariableHolder::new();
    for item in items {
        genc_item(fmt, item, &mut vars);
    }
}

fn genc_item<'a, 'tcx: 'a>(fmt: &mut SourceBuilder, item: &'tcx Item<'tcx>, vars: &'a mut VariableHolder<'tcx>) {
    match item {
        Item::Let { name, ty, expr, .. } => {
            fmt.writeln(&format!("{} {} = {};",
                                 format_ty(ty.unwrap()),
                                 name,
                                 format_expr(expr.as_ref().unwrap())));
            vars.def(name.clone(), ty.clone().unwrap(), expr.clone());
        }
        Item::Assignment {
            lhs,
            operator,
            expr,
        } => {
            match operator {
                Some(op) => {
                    fmt.writeln(&format!("{} {}= {};", format_expr(lhs), op, format_expr(expr)));
                }
                None => {
                    fmt.writeln(&format!("{} = {};", format_expr(lhs), format_expr(expr)));
                }
            }
        }
        Item::Function {
            name,
            args,
            ty,
            body,
            ..
        } => {
            fmt.write(&format!("{} {}(", format_ty(ty.unwrap()), name));
            if args.is_empty() {
                fmt.write("void");
            } else {
                for (i, arg) in args.iter().enumerate() {
                    fmt.write(&format!("{} {}", format_ty(arg.ty), arg.name));
                    if i != args.len() - 1 {
                        fmt.write(", ");
                    }
                }

                for arg in args {
                    vars.def(arg.name.clone(), arg.ty, None);
                }
            }

            fmt.writeln(") {");
            fmt.shift();
            for b in body {
                genc_item(fmt, b, vars);
            }
            fmt.unshift();
            fmt.writeln("}");
        }
        Item::Struct {
            name,
            fields,
        } => {
            fmt.write("struct ");
            fmt.write(&name);
            fmt.writeln(" {");
            fmt.shift();
            for field in fields {
                fmt.write(&format_ty(field.ty));
                fmt.write(" ");
                fmt.write(&field.name);
                fmt.writeln(";");
            }
            fmt.unshift();
            fmt.writeln("}");
        }
        Item::If {
            condition,
            arm_true,
            arm_false,
        } => {
            fmt.writeln(&format!("if ({}) {{", format_expr(condition)));
            fmt.shift();
            for item in arm_true {
                fmt.shift();
                genc_item(fmt, item, vars);
                fmt.unshift();
            }
            if let Some(arm_false) = arm_false {
                fmt.writeln("} else {");
                for item in arm_false {
                    fmt.shift();
                    genc_item(fmt, item, vars);
                    fmt.unshift();
                }
            }
            fmt.unshift();
            fmt.writeln("}");
        }
        Item::ForIn {
            name: bound,
            expr,
            body,
        } => match expr {
            Expression::Identifier(name) => {
                let (ty, _expr) = vars.get(name.clone()).expect(&name);
                let (n, ty) = match *ty {
                    TyS::Array(n, ty) => (Some(n), ty),
                    TyS::Slice(ty) => (None, ty),
                    TyS::U32 => (Some(&10), ty),
                    other => {
                        dbg!(other);
                        unimplemented!()
                    }
                };

                if let Some(n) = n {
                    fmt.writeln(&format!("for (int64_t i = 0; i < {}; ++i) {{", n));
                } else {
                    fmt.writeln(&format!("for (int64_t i = 0; i < {}.len; ++i) {{", name));
                }

                fmt.shift();
                if n.is_some() {
                    fmt.writeln(&format!("{} {} = {}[i];", format_ty(ty), bound, name));
                } else {
                    fmt.writeln(&format!(
                        "{} {} = ((*{}){}.ptr)[i];",
                        format_ty(ty),
                        bound,
                        format_ty(ty),
                        name
                    ));
                }
                for item in body {
                    genc_item(fmt, item, vars);
                }
                fmt.unshift();
                fmt.writeln("}");
            }
            _ => unimplemented!(),
        },
        Item::Break => {}
        Item::Yield(_) => {}
        Item::Return(expr) => {
            fmt.writeln(&format!("return {};", format_expr(expr)));
        }
        Item::Expr { expr } => {
            fmt.writeln(&format!("{};", format_expr(expr)));
        }
    }
}

fn format_ty(ty: &TyS) -> Cow<'static, str> {
    match ty {
        TyS::U32 => Cow::Borrowed("uint32_t"),
        TyS::I32 => Cow::Borrowed("int32_t"),
        TyS::Array(len, ty) => Cow::Owned(format!("{}[{}]", format_ty(&**ty), len)),
        TyS::Slice(_) => Cow::Owned(format!("Slice")),
        TyS::Other(_) => Cow::Borrowed("/* generated */"),
        TyS::Tuple(_) => Cow::Borrowed("/* generated */"),
        TyS::Unit => Cow::Borrowed("void)"),
        TyS::Function(..) => Cow::Borrowed("void*"),
        TyS::Pointer(inner) => Cow::Owned(format!("{}*", format_ty(&**inner))),
        TyS::Bool => Cow::Borrowed("bool"),
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
            for (i, arg) in args.iter().enumerate() {
                s.push_str(&*format_expr(arg));
                if i != args.len() - 1 {
                    s.push_str(", ");
                }
            }
            s.push(')');

            Cow::Owned(s)
        }
        Expression::Tuple(_) => Cow::Borrowed("/* generated */"),
        Expression::Bool(value) => Cow::Borrowed(if *value { "true" } else { "false" }),
    }
}