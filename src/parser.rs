use std::fmt;
use super::{Lexer, Token, TokenType, Keyword, PunctKind};


pub struct Parser<'a> {
    peek: MultiPeek<'a>,
}

struct MultiPeek<'a> {
    peeked: [Option<Token<'a>>; 2],
    index: usize,
    length: usize,
    lex: &'a mut Lexer<'a>,
}

impl<'a> MultiPeek<'a> {
    fn new(lex: &'a mut Lexer<'a>) -> Self {
        MultiPeek {
            peeked: [None, None],
            index: 0,
            length: 0,
            lex
        }
    }

    /// Returns next item without consuming it
    pub fn peek(&mut self, offset: usize) -> &Token<'a> {
        self.ensure_peeked(offset);
        self.peeked[(self.index + offset) % self.peeked.len()].as_ref().unwrap()
    }

    /// Returns next item and removes it from queue
    pub fn advance(&mut self) -> Token<'a> {
        self.ensure_peeked(0);
        let item = self.peeked[self.index].take().unwrap();
        self.length -= 1;
        self.index = (self.index + 1) % self.peeked.len();
        item
    }

    /// Ensures that next token (if any) is taken from lexer
    fn ensure_peeked(&mut self, offset: usize) {
        if self.length <= offset {
            self.peeked[(self.index + self.length) % self.peeked.len()] = self.lex.next().ok();
            self.length += 1;
        }
    }
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
pub struct Argument {
    name: String,
    ty: String,
}

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
        ty: Option<String>,
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
            peek: MultiPeek::new(lex),
        }
    }

    pub fn parse(&mut self) -> ParseResult<Vec<Item>> {
        let mut items = vec![];
        loop {
            let token = self.peek(0);
            let item = match token.get_type() {
                TokenType::Keyword(Keyword::Func) => self.parse_func(),
                TokenType::EndOfSource => break,
                token_type => unimplemented!("{:?}", token_type),
            };
            items.push(item?);
        }
        Ok(items)
    }

    fn parse_stmts(&mut self) -> ParseResult<Vec<Item>> {
        let mut items = vec![];
        loop {
            let token = self.peek(0);
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
        let token = self.peek(0);
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

    fn parse_func(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::Func)?;
        let identifier = self.expect_identifier()?.as_string();
        self.expect_one('(')?;
        let mut args = vec![];
        while let Some(t) = self.match_identifier() {
            self.expect_one(':')?;
            let ty = self.expect_identifier()?;
            args.push(Argument {
                name: t.as_string(),
                ty: ty.as_string()
            });
            if !self.match_one(',') {
                break;
            }
        }
        self.expect_one(')')?;

        let ty = if self.match_many(&['-', '>']) {
            Some(self.expect_identifier()?.as_string())
        } else {
            None
        };

        self.expect_one('{')?;
        let body = self.parse_stmts()?;
        self.expect_one('}')?;
        Ok(Item::Function { name: identifier, args, body, ty })
    }

    fn parse_for(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::For)?;
        let identifier = self.expect_identifier()?.as_string();
        self.expect_keyword(Keyword::In)?;
        let expr = self.parse_expr()?;
        self.expect_one('{')?;
        let items = self.parse_stmts()?;
        self.expect_one('}')?;
        Ok(Item::ForIn { name: identifier.to_owned(), expr, body: items })
    }

    fn parse_if(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::If)?;
        let condition = self.parse_expr()?;
        self.expect_one('{')?;
        let arm_true = self.parse_stmts()?;
        self.expect_one('}')?;
        let arm_false = if self.match_keyword(Keyword::Else).is_some() {
            self.expect_one('{')?;
            let false_arm = self.parse_stmts()?;
            self.expect_one('}')?;
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
        let token = self.peek(0);
        ParseError::UnexpectedToken(
            TokenType::EndOfSource,
            token.line(),
            token.column(),
            Some(token_type),
        )
    }

    fn match_keyword(&mut self, keyword: Keyword) -> Option<Token<'a>> {
        match self.peek(0).get_type() {
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
        match self.peek(0).get_type() {
            TokenType::Identifier => Some(self.advance()),
            _ => None,
        }
    }

    /// Consumes and returns next token, otherwise returns an error
    fn expect_one(&mut self, ch: char) -> ParseResult<()> {
        match self.expect_punct(PunctKind::Single, ch) {
            Ok(_) => Ok(()),
            Err(_) =>  self.expect_punct(PunctKind::Joint, ch)
        }
    }

    /// Consumes next token and returns true only if it is a char given by argument
    fn match_one(&mut self, ch: char) -> bool {
        self.match_punct(PunctKind::Single, ch) || self.match_punct(PunctKind::Joint, ch)
    }

    #[allow(dead_code)]
    #[inline]
    fn expect_many(&mut self, chars: &[char]) -> ParseResult<()> {
        for (idx, &expected) in chars.iter().enumerate() {
            match self.peek(0).get_punct() {
                Some((actual, kind)) if actual == expected => {
                    if idx == chars.len() - 1 || kind == PunctKind::Joint {
                        self.advance();
                    }
                },
                _ => {
                    return Err(self.expected(TokenType::Punct(expected)));
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn match_many(&mut self, chars: &[char]) -> bool {
        for (index, &expected) in chars.iter().enumerate() {
            if let Some((actual, kind)) = self.peek(index).get_punct() {
                if actual != expected {
                    return false;
                }

                let is_last = index == chars.len() - 1;
                match kind {
                    PunctKind::Single if !is_last => return false,
                    PunctKind::Single | PunctKind::Joint => continue,
                }
            }
        }

        for _ in chars {
            self.advance();
        }

        return true;
    }

    /// Consumes and returns next token, otherwise returns an error
    fn expect_punct(&mut self, kind: PunctKind, ch: char) -> ParseResult<()> {
        if self.match_punct(kind, ch) {
            Ok(())
        } else {
            Err(self.expected(TokenType::Punct(ch)))
        }
    }

    /// Consumes and returns next token only if it is a char given by argument
    fn match_punct(&mut self, kind: PunctKind, ch: char) -> bool {
        if self.peek(0).get_punct() == Some((ch, kind)) {
            self.advance();
            return true;
        }
        false
    }

    /// Returns next token without consuming it
    fn peek(&mut self, offset: usize) -> Token<'a> {
        self.peek.peek(offset).clone()
    }

    /// Returns next token and consumes it
    fn advance(&mut self) -> Token<'a> {
        self.peek.advance()
    }
}
