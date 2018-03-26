mod lexer;
mod parser;

use lexer::{Lexer, Token, TokenType, Keyword, Special};
use parser::Parser;

fn main() {
    use std::io::prelude::*;
    use std::io::BufReader;
    use std::fs::File;
    let file = File::open("test.kl").unwrap();
    let mut content = String::new();
    BufReader::new(file).read_to_string(&mut content).unwrap();

    let mut lex = Lexer::from_source(&content);
    let mut parser = Parser::new(&mut lex);
    match parser.parse() {
        Ok(k) => println!("{:#?}", k),
        Err(e) => println!("{:?}", e),
    }
}
