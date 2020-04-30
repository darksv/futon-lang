use crate::ast::{Expr, Item, Operator, TyExpr};
use crate::ty::{Ty, TyS};
use std::borrow::{Borrow, Cow};
use std::collections::{HashMap, HashSet};

struct VariableHolder<'tcx> {
    variables: HashMap<String, Vec<(Ty<'tcx>, Option<TyExpr<'tcx>>)>>,
}

impl<'tcx> VariableHolder<'tcx> {
    fn new() -> Self {
        VariableHolder {
            variables: Default::default(),
        }
    }

    fn def<T: Into<String> + Borrow<str>>(
        &mut self,
        name: T,
        ty: Ty<'tcx>,
        val: Option<TyExpr<'tcx>>,
    ) {
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

    fn get<T: Into<String> + Borrow<str>>(
        &mut self,
        name: T,
    ) -> Option<&(Ty<'tcx>, Option<TyExpr<'tcx>>)> {
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
            self.buffer
                .push_str(&format!("{:width$}", "", width = self.level * 2));
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

    let mut vars = VariableHolder::new();
    let mut emitted = HashSet::new();
    for item in items {
        genc_item(fmt, item, &mut vars, &mut emitted);
    }
}

fn genc_item<'a, 'tcx: 'a>(
    fmt: &mut SourceBuilder,
    item: &'tcx Item<'tcx>,
    vars: &'a mut VariableHolder<'tcx>,
    emitted_tys: &mut HashSet<Ty<'tcx>>,
) {
    match item {
        Item::Let { name, ty, expr, .. } => {
            ensure_ty_emitted(
                fmt,
                emitted_tys,
                ty.expect(&format!("no type for {}", name)),
            );
            fmt.writeln(&format!(
                "{} {} = {};",
                format_ty(ty.unwrap()),
                name,
                format_expr(expr.as_ref().unwrap())
            ));
            vars.def(name.clone(), ty.clone().unwrap(), expr.clone());
        }
        Item::Assignment {
            lhs,
            operator,
            expr,
        } => match operator {
            Some(op) => {
                fmt.writeln(&format!(
                    "{} {}= {};",
                    format_expr(lhs),
                    format_operator(op),
                    format_expr(expr)
                ));
            }
            None => {
                fmt.writeln(&format!("{} = {};", format_expr(lhs), format_expr(expr)));
            }
        },
        Item::Function {
            is_extern: true, ..
        } => {}
        Item::Function {
            name,
            args,
            ty,
            body,
            ..
        } => {
            ensure_ty_emitted(fmt, emitted_tys, ty);
            fmt.write(&format!("{} {}(", format_ty(ty), name));
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
                    ensure_ty_emitted(fmt, emitted_tys, arg.ty);
                }
            }

            fmt.writeln(") {");
            fmt.shift();
            for b in body {
                genc_item(fmt, b, vars, emitted_tys);
            }
            fmt.unshift();
            fmt.writeln("}");
        }
        Item::Struct { .. } => {}
        Item::If {
            condition,
            arm_true,
            arm_false,
        } => {
            fmt.writeln(&format!("if ({}) {{", format_expr(condition)));
            fmt.shift();
            for item in arm_true {
                fmt.shift();
                genc_item(fmt, item, vars, emitted_tys);
                fmt.unshift();
            }
            if let Some(arm_false) = arm_false {
                fmt.writeln("} else {");
                for item in arm_false {
                    fmt.shift();
                    genc_item(fmt, item, vars, emitted_tys);
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
        } => match &expr.expr {
            Expr::Identifier(name) => {
                let (ty, _expr) = vars.get(name.clone()).expect(&name);
                let (n, ty) = match *ty {
                    TyS::Array(n, ty) => (Some(n), ty),
                    TyS::Slice(ty) => (None, ty),
                    TyS::U32 => (Some(&10), ty),
                    TyS::I32 => (Some(&10), ty),
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
                    genc_item(fmt, item, vars, emitted_tys);
                }
                fmt.unshift();
                fmt.writeln("}");
            }
            Expr::Range(to, None) => {
                fmt.writeln(&format!(
                    "for (int64_t i = 0; i < {}; ++i) {{",
                    format_expr(to)
                ));
                fmt.shift();
                for item in body {
                    genc_item(fmt, item, vars, emitted_tys);
                }
                fmt.unshift();
                fmt.writeln("}");
            }
            expr => unimplemented!("{:?}", expr),
        },
        Item::Break => {
            fmt.writeln("break;");
        }
        Item::Yield(_) => {}
        Item::Return(expr) => {
            fmt.writeln(&format!("return {};", format_expr(expr)));
        }
        Item::Expr { expr } => {
            fmt.writeln(&format!("{};", format_expr(expr)));
        }
        Item::Loop { body } => {
            fmt.writeln("for (;;) {{");

            fmt.shift();
            for item in body {
                genc_item(fmt, item, vars, emitted_tys);
            }
            fmt.unshift();
            fmt.writeln("}");
        }
    }
}

fn ensure_ty_emitted<'tcx>(fmt: &mut SourceBuilder, emitted: &mut HashSet<Ty<'tcx>>, ty: Ty<'tcx>) {
    if emitted.contains(ty) {
        return;
    }

    match ty {
        TyS::Bool => {}
        TyS::U32 => {}
        TyS::I32 => {}
        TyS::F32 => {}
        TyS::Array(_, _) => {}
        TyS::Slice(_) => {
            fmt.writeln(r"typedef struct { void* ptr; uint64_t len; }  Slice;");
        }
        TyS::Unit => {}
        TyS::Tuple(tys) => {
            fmt.writeln("typedef struct {");
            fmt.shift();
            for (idx, ty) in tys.iter().enumerate() {
                ensure_ty_emitted(fmt, emitted, ty);
                fmt.write(&format_ty(ty));
                fmt.write(" ");
                fmt.write("item");
                fmt.write(&idx.to_string());
                fmt.writeln(";");
            }
            fmt.unshift();
            fmt.writeln("} Struct123;");
        }
        TyS::Function(_, _) => {}
        TyS::Pointer(_) => {}
        TyS::Range => {}
        TyS::Other(_) => {}
        TyS::Unknown => {}
        TyS::Error => {}
    }
    emitted.insert(ty);
}

fn format_ty(ty: Ty) -> Cow<str> {
    match ty {
        TyS::U32 => Cow::Borrowed("uint32_t"),
        TyS::I32 => Cow::Borrowed("int32_t"),
        TyS::F32 => Cow::Borrowed("float"),
        TyS::Array(len, ty) => Cow::Owned(format!("{}[{}]", format_ty(ty), len)),
        TyS::Slice(_) => Cow::Owned(format!("Slice")),
        TyS::Other(name) => Cow::Borrowed(name.as_str()),
        TyS::Tuple(_) => Cow::Borrowed("/* generated */"),
        TyS::Unit => Cow::Borrowed("void)"),
        TyS::Function(..) => Cow::Borrowed("void*"),
        TyS::Pointer(inner) => Cow::Owned(format!("{}*", format_ty(inner))),
        TyS::Bool => Cow::Borrowed("bool"),
        TyS::Range => Cow::Borrowed("/* generated */"),
        TyS::Unknown => Cow::Borrowed("/* unknown */"),
        TyS::Error => Cow::Borrowed("/* error */"),
    }
}

fn format_operator(op: &Operator) -> Cow<'static, str> {
    match op {
        Operator::Add => Cow::Borrowed("+"),
        Operator::Sub => Cow::Borrowed("-"),
        Operator::Mul => Cow::Borrowed("*"),
        Operator::Div => Cow::Borrowed("/"),
        Operator::Equal => Cow::Borrowed("=="),
        Operator::NotEqual => Cow::Borrowed("!="),
        Operator::Less => Cow::Borrowed("<"),
        Operator::Greater => Cow::Borrowed(">"),
        Operator::LessEqual => Cow::Borrowed("<="),
        Operator::GreaterEqual => Cow::Borrowed(">="),
        Operator::Negate => Cow::Borrowed("-"),
        Operator::Ref => Cow::Borrowed("&"),
        Operator::Deref => Cow::Borrowed("*"),
    }
}

fn format_expr<'expr, 'tcx>(expr: &'expr TyExpr<'tcx>) -> Cow<'expr, str> {
    match &expr.expr {
        Expr::Identifier(x) => Cow::Borrowed(x.as_str()),
        Expr::Integer(value) => Cow::Owned(value.to_string()),
        Expr::Float(value) => Cow::Owned(value.to_string()),
        Expr::Prefix(op, expr) => {
            Cow::Owned(format!("{}{}", format_operator(op), format_expr(expr)))
        }
        Expr::Infix(op, lhs, rhs) => Cow::Owned(format!(
            "({} {} {})",
            format_expr(lhs),
            format_operator(op),
            format_expr(rhs)
        )),
        Expr::Array(values) => {
            let mut s = String::new();
            s.push('{');
            for (i, value) in values.iter().enumerate() {
                s.push_str(format_expr(value).as_ref());
                if i != values.len() - 1 {
                    s.push_str(", ");
                }
            }
            s.push('}');
            Cow::Owned(s)
        }
        Expr::Call(expr, args) => {
            let mut s = String::new();
            s.push('(');
            s.push_str(&format_expr(expr));
            s.push(')');
            s.push('(');
            for (i, arg) in args.iter().enumerate() {
                s.push_str(format_expr(arg).as_ref());
                if i != args.len() - 1 {
                    s.push_str(", ");
                }
            }
            s.push(')');
            Cow::Owned(s)
        }
        Expr::Tuple(_) => Cow::Borrowed("/* generated */"),
        Expr::Bool(value) => Cow::Borrowed(if *value { "true" } else { "false" }),
        place @ Expr::Place(_, _) => format_place(expr),
        Expr::Range(_, _) => Cow::Borrowed("/* range */"),
        Expr::Index(arr, index_exp) => {
            todo!()
        }
    }
}

fn format_place<'expr, 'tcx>(place: &'expr TyExpr<'tcx>) -> Cow<'expr, str> {
    match &place.expr {
        Expr::Identifier(name) => Cow::Borrowed(name),
        Expr::Place(base, path) => {
            Cow::Owned(format!("{}.{}", format_place(base), format_place(path)))
        }
        _ => unimplemented!(),
    }
}
