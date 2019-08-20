mod ast;
mod codegen;
mod lexer;
mod parser;

use lexer::{Keyword, Lexer, PunctKind, Token, TokenType};
use parser::Parser;

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
        let mut parser = Parser::new(&mut lex);
        match parser.parse() {
            Ok(k) => {
                println!("// {}", entry.path().to_str().unwrap());
                //            println!("{:#?}", k);
                codegen::genc(&k, 0);
            }
            Err(e) => println!("{:?}", e),
        }
    }
}
