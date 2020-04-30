mod arena;
mod ast;
mod codegen;
mod lexer;
mod mir;
mod multi_peek;
mod parser;
mod ty;
mod typeck;

use crate::arena::Arena;
use crate::typeck::infer_types;
use codegen::SourceBuilder;
use lexer::{Keyword, Lexer, PunctKind, Token, TokenType};
use parser::Parser;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use crate::mir::{build_mir, dump_mir, execute_mir, Const};

fn main() {
    env_logger::init();

    if let Some(file) = env::args().nth(1) {
        compile_file(file);
    } else {
        for entry in std::fs::read_dir("tests").unwrap() {
            let entry = entry.unwrap();
            println!("testing {:?}", entry.path());
            compile_file(entry.path());
        }
    }
}

fn compile_file(path: impl AsRef<Path>) {
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::BufReader;

    let file = File::open(path.as_ref()).unwrap();
    let mut content = String::new();
    BufReader::new(file).read_to_string(&mut content).unwrap();

    let lex = Lexer::from_source(&content);
    let arena = Arena::default();
    let mut parser = Parser::new(lex, &arena);
    match parser.parse() {
        Ok(ref mut k) => {
            let mut s = SourceBuilder::new();
            let mut locals = HashMap::new();
            infer_types(k, &arena, &mut locals, None);
            for item in k.iter_mut() {
                let mir = build_mir(item).unwrap();
                dump_mir(&mir, &mut std::io::stdout()).unwrap();
                println!("evaluated: {:?}", execute_mir(&mir, &[Const::U32(300)]));
            }

            // codegen::genc(&mut s, k);
            // println!("// {}", path.as_ref().to_str().unwrap());
            // println!("{}", s.build());
        }
        Err(e) => println!("{:?}", e),
    }
}
