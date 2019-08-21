extern crate typed_arena;

mod ast;
mod codegen;
mod lexer;
mod parser;

use lexer::{Keyword, Lexer, PunctKind, Token, TokenType};
use parser::Parser;
use codegen::SourceBuilder;
use typed_arena::Arena;

fn main() {
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::BufReader;

    for entry in std::fs::read_dir("tests").unwrap() {
        let entry = entry.unwrap();

        let file = File::open(entry.path()).unwrap();
        let mut content = String::new();
        BufReader::new(file).read_to_string(&mut content).unwrap();

        let mut lex = Lexer::from_source(&content);
        let arena = Arena::default();
        let mut parser = Parser::new(&mut lex, &arena);
        match parser.parse() {
            Ok(k) => {
                let mut s = SourceBuilder::new();
                codegen::genc(&mut s, &k,);
                println!("// {}", entry.path().to_str().unwrap());
                println!("{}", s.build());
            }
            Err(e) => println!("{:?}", e),
        }
    }
}
