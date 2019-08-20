mod lexer;
mod parser;
mod codegen;

use lexer::{Lexer, Token, TokenType, Keyword, PunctKind};
use parser::{Parser, Item, Ty};

fn main() {
    use std::io::prelude::*;
    use std::io::BufReader;
    use std::fs::File;

    for entry in std::fs::read_dir("tests").unwrap() {
        let entry = entry.unwrap();

        let file = File::open(entry.path()).unwrap();
        let mut content = String::new();
        BufReader::new(file).read_to_string(&mut content).unwrap();

        let mut lex = Lexer::from_source(&content);
        let mut parser = Parser::new(&mut lex);
        match parser.parse() {
            Ok(k) => {
//            println!("{:#?}", k);
                codegen::genc(&k, 0);
            }
            Err(e) => println!("{:?}", e),
        }
    }
}
