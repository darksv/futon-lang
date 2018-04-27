use std::fmt;
use super::{Lexer, Token, TokenType, Keyword, Special};


pub struct Parser<'a> {
    lex: &'a mut Lexer<'a>,
    peeked: Option<Token<'a>>,
}

pub enum ParseError {
    UnexpectedToken(TokenType, usize, usize, Option<TokenType>),
}

impl fmt::Debug for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &ParseError::UnexpectedToken(actual, line, column, None) => {
                write!(f, "Unexpected {:?} at {}:{}", actual, line, column)?
            }
            &ParseError::UnexpectedToken(actual, line, column, Some(expected)) => {
                write!(f, "Unexpected {:?} at {}:{}, expected {:?}", actual, line, column, expected)?
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Argument { name: String }

#[derive(Debug)]
pub enum Expression {
    Identifier(String),
    IntegralConstant(i32),
    FloatingConstant(f32),
}

#[derive(Debug)]
pub enum Item {
    Function {
        name: String,
        args: Vec<Argument>,
        body: Vec<Item>,
    },
    If {
        condition: Expression,
        arm_true: Vec<Item>,
        arm_false: Option<Vec<Item>>,
    },
    ForIn {
        name: String,
        expr: Expression,
        body: Vec<Item>,
    },
    Break,
    Yield(Box<Expression>),
    Return(Box<Expression>),
}

type ParseResult<T> = Result<T, ParseError>;

impl<'a> Parser<'a> {
    pub fn new(lex: &'a mut Lexer<'a>) -> Parser<'a> {
        Parser {
            lex,
            peeked: None,
        }
    }

    pub fn parse(&mut self) -> ParseResult<Vec<Item>> {
        let mut items = vec![];
        loop {
            let token = self.peek();
            let item = match token.get_type() {
                TokenType::Keyword(Keyword::Func) => self.parse_func(),
                TokenType::EndOfSource => break,
                token_type => unimplemented!("{:?}", token_type),
            };
            items.push(item?);
        }
        Ok(items)
    }

    fn parse_func(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::Func)?;
        let identifier = self.expect_identifier()?.as_string();
        self.expect_single('(')?;
        let mut args = vec![];
        while let Some(t) = self.match_identifier() {
            args.push(Argument {
                name: t.as_string(),
            });
            if self.match_single(',').is_none() {
                break;
            }
        }
        self.expect_single(')')?;
        self.expect_single('{')?;
        let body = self.parse_stmts()?;
        self.expect_single('}')?;
        Ok(Item::Function { name: identifier, args, body })
    }

    fn parse_stmts(&mut self) -> ParseResult<Vec<Item>> {
        let mut items = vec![];
        loop {
            let token = self.peek();
            let item = match token.get_type() {
                TokenType::Keyword(Keyword::For) => self.parse_for(),
                TokenType::Keyword(Keyword::Func) => self.parse_func(),
                TokenType::Keyword(Keyword::If) => self.parse_if(),
                TokenType::Keyword(Keyword::Yield) => self.parse_yield(),
                TokenType::Keyword(Keyword::Return) => self.parse_return(),
                TokenType::Keyword(Keyword::Break) => {
                    self.advance();
                    Ok(Item::Break)
                }
                _ => break,
            };
            items.push(item?);
        }
        Ok(items)
    }

    fn parse_expr(&mut self) -> ParseResult<Expression> {
        let token = self.peek();
        let expr = match (token.get_type(), token) {
            (TokenType::IntegralNumber, token) => {
                self.advance();
                Expression::IntegralConstant(token.get_integer().unwrap())
            }
            (TokenType::FloatingNumber, token) => {
                self.advance();
                Expression::FloatingConstant(token.get_float().unwrap())
            }
            (TokenType::Identifier, token) => {
                self.advance();
                Expression::Identifier(token.as_string())
            }
            _ => unimplemented!(),
        };
        Ok(expr)
    }

    fn parse_for(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::For)?;
        let identifier = self.expect_identifier()?.as_string();
        self.expect_keyword(Keyword::In)?;
        let expr = self.parse_expr()?;
        self.expect_single('{')?;
        let items = self.parse_stmts()?;
        self.expect_single('}')?;
        Ok(Item::ForIn { name: identifier.to_owned(), expr, body: items })
    }

    fn parse_if(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::If)?;
        let condition = self.parse_expr()?;
        self.expect_single('{')?;
        let arm_true = self.parse_stmts()?;
        self.expect_single('}')?;
        let arm_false = if self.match_keyword(Keyword::Else).is_some() {
            self.expect_single('{')?;
            let false_arm = self.parse_stmts()?;
            self.expect_single('}')?;
            Some(false_arm)
        } else {
            None
        };
        Ok(Item::If { condition, arm_true, arm_false })
    }

    fn parse_yield(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::Yield)?;
        let value = self.parse_expr()?;
        Ok(Item::Yield(Box::new(value)))
    }

    fn parse_return(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::Return)?;
        let value = self.parse_expr()?;
        Ok(Item::Return(Box::new(value)))
    }

    fn expect_keyword(&mut self, keyword: Keyword) -> ParseResult<Token<'a>> {
        match self.match_keyword(keyword) {
            Some(token) => Ok(token),
            None => Err(self.expected(TokenType::Keyword(keyword))),
        }
    }

    fn expected(&mut self, token_type: TokenType) -> ParseError {
        let token = self.peek();
        ParseError::UnexpectedToken(
            TokenType::EndOfSource,
            token.line(),
            token.column(),
            Some(token_type),
        )
    }

    fn match_keyword(&mut self, keyword: Keyword) -> Option<Token<'a>> {
        match self.peek().get_type() {
            TokenType::Keyword(kw) if kw == keyword => Some(self.advance()),
            _ => None,
        }
    }

    fn expect_identifier(&mut self) -> ParseResult<Token<'a>> {
        match self.match_identifier() {
            Some(token) => Ok(token),
            None => Err(self.expected(TokenType::Identifier))
        }
    }

    fn match_identifier(&mut self) -> Option<Token<'a>> {
        match self.peek().get_type() {
            TokenType::Identifier => Some(self.advance()),
            _ => None,
        }
    }

    /// Consumes and returns next token, otherwise returns an error
    fn expect_single(&mut self, symbol: char) -> ParseResult<Token<'a>> {
        self.expect_symbol(Special::Single(symbol))
    }

    /// Consumes and returns next token only if it is a char given by argument
    fn match_single(&mut self, symbol: char) -> Option<Token<'a>> {
        self.match_symbol(Special::Single(symbol))
    }

    /// Consumes and returns next token, otherwise returns an error
    fn expect_symbol(&mut self, symbol: Special) -> ParseResult<Token<'a>> {
        match self.match_symbol(symbol) {
            Some(token) => Ok(token),
            None => Err(self.expected(TokenType::Special(symbol)))
        }
    }

    /// Consumes and returns next token only if it is a char given by argument
    fn match_symbol(&mut self, symbol: Special) -> Option<Token<'a>> {
        if self.peek().get_special() == Some(symbol) {
            Some(self.advance())
        } else {
            None
        }
    }

    /// Returns next token without consuming it
    fn peek(&mut self) -> Token<'a> {
        self.ensure_peeked();
        self.peeked.clone().unwrap()
    }

    /// Returns next token and consumes it
    fn advance(&mut self) -> Token<'a> {
        self.ensure_peeked();
        let token = self.peeked.take().unwrap();
        token
    }

    /// Ensures that next token (if any) is taken from lexer
    fn ensure_peeked(&mut self) {
        if self.peeked.is_none() {
            self.peeked = self.lex.next().ok();
        }
    }
}
