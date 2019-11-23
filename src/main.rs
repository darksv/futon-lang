mod arena;
mod ast;
mod codegen;
mod lexer;
mod multi_peek;
mod parser;
mod ty;
mod typeck;

use lexer::{Keyword, Lexer, PunctKind, Token, TokenType};
use parser::Parser;
use codegen::SourceBuilder;
use crate::arena::Arena;
use std::path::Path;
use std::env;
use crate::typeck::infer_types;
use std::collections::HashMap;

fn main() {
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
            dbg!(&k);

            let mut s = SourceBuilder::new();
            let mut locals = HashMap::new();
            match infer_types(k, &arena, &mut locals) {
                Ok(_) => {
                    codegen::genc(&mut s, k);
                    println!("// {}", path.as_ref().to_str().unwrap());
                    println!("{}", s.build());
                }
                Err(e) => {
                    println!("typeck error: {}", e);
                }
            }

        }
        Err(e) => println!("{:?}", e),
    }
}