#![feature(if_let_guard)]
#![allow(clippy::match_like_matches_macro)]
#![allow(unused)]

extern crate core;

use indexmap::IndexMap;
use std::collections::HashMap;
use std::env;
use std::path::Path;

use lexer::{Keyword, Lexer, PunctKind, Token, TokenType};
use parser::Parser;

use crate::arena::Arena;
use crate::ast::Operator;
use crate::ir::{build_ir, dump_ir, execute_ir, validate_types, Const};
use crate::type_checking::{ExprToType, Expression, Item, TypeCheckerContext};

mod arena;
mod ast;
mod ir;
mod lexer;
mod multi_peek;
mod parser;
mod type_checking;
mod types;

fn main() {
    env_logger::init();

    if let Some(file) = env::args_os().nth(1) {
        run_test(file);
    } else {
        let mut total = 0;
        let mut successful = 0;
        for entry in std::fs::read_dir("tests").unwrap() {
            if run_test(entry.unwrap().path()) {
                successful += 1;
            }
            total += 1;
        }

        println!("{}/{}", successful, total);
    }
}

fn run_test(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    print!("Testing {} --- ", path.display());
    std::panic::catch_unwind(|| compile_file(path)).unwrap_or(false)
}

fn compile_file(path: impl AsRef<Path>) -> bool {
    let content = std::fs::read(path.as_ref()).unwrap();
    let content = String::from_utf8(content).unwrap();

    let lex = Lexer::from_source(&content);
    let arena = Arena::default();
    let mut parser = Parser::new(lex);
    match parser.parse() {
        Ok(mut items) => {
            let mut tc_ctx = TypeCheckerContext {
                arena: &arena,
                locals: HashMap::new(),
                defined_types: HashMap::new(),
                exprs: &Arena::default(),
                type_by_expr: ExprToType::new(),
            };

            let items = tc_ctx.infer_types(&mut items, None);

            let mut functions = HashMap::new();
            let mut asserts = Vec::new();
            for item in &items {
                match item {
                    Item::Function { name, .. } => {
                        let ir = build_ir(&item, &arena, &tc_ctx.exprs, &mut tc_ctx.type_by_expr)
                            .unwrap();
                        validate_types(&ir);
                        dump_ir(&ir, &mut std::io::stdout()).unwrap();
                        functions.insert(name.clone(), ir);
                    }
                    Item::Assert(expr) => {
                        asserts.push(expr);
                    }
                    _ => (),
                }
            }

            let mut success = true;
            for assert in &asserts {
                let Expression::Infix(Operator::Equal, lhs, rhs) = assert else {
                    panic!("not a comparison");
                };

                let Expression::Call(fun, args) = lhs else {
                    panic!("not a call");
                };

                let Expression::Identifier(name) = fun else {
                    panic!("not a function call");
                };

                let expected = rhs.as_const().unwrap();
                let args: Vec<_> = args.iter().map(|it| it.as_const().unwrap()).collect();
                let actual = execute_ir(&functions[name], &args, &functions);

                if expected != actual {
                    println!("Assertion failed! {:?} {:?}", expected, actual);
                    success = false;
                }
            }

            if asserts.is_empty() {
                println!("no assertions");
            } else if success {
                println!("OK");
                return true;
            }
        }
        Err(e) => println!("{:?}", e),
    }
    false
}
