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
            lex,
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
    Expr,
    Other,
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
            &ParseError::Other => {
                write!(f, "other")?
            }
            &ParseError::Expr => {
                write!(f, "expr")?
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Argument {
    pub name: String,
    pub ty: Ty,
}

#[derive(Debug)]
#[derive(Clone)]
pub enum Expression {
    Identifier(String),
    IntegralConstant(i32),
    FloatingConstant(f32),
    BoolConstant(bool),
    Prefix(TokenType, Box<Expression>),
    Infix(TokenType, Box<Expression>, Box<Expression>),
    Array(Vec<Expression>),
    Tuple(Vec<Expression>),
    Call(Box<Expression>, Vec<Expression>),
}

#[derive(Debug)]
pub enum Item {
    Let {
        name: String,
        ty: Option<Ty>,
        expr: Option<Expression>,
    },
    Assignment {
        lhs: Expression,
        operator: Option<char>,
        expr: Expression,
    },
    Function {
        name: String,
        args: Vec<Argument>,
        ty: Option<Ty>,
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

#[derive(Debug)]
#[derive(Clone)]
pub enum Ty {
    Bool,
    U32,
    I32,
    Array(usize, Box<Ty>),
    Slice(Box<Ty>),
    Tuple(Vec<Ty>),
    Function(Vec<Ty>, Box<Ty>),
    Ptr(Box<Ty>),
    UserDefined(String),
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
                TokenType::Keyword(Keyword::Fn) => self.parse_fn(),
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
                TokenType::Keyword(Keyword::Let) => self.parse_let(),
                TokenType::Keyword(Keyword::For) => self.parse_for(),
                TokenType::Keyword(Keyword::Fn) => self.parse_fn(),
                TokenType::Keyword(Keyword::If) => self.parse_if(),
                TokenType::Keyword(Keyword::Yield) => self.parse_yield(),
                TokenType::Keyword(Keyword::Return) => self.parse_return(),
                TokenType::Keyword(Keyword::Break) => {
                    self.advance();
                    Ok(Item::Break)
                }
                TokenType::Punct('*') => {
                    let lhs = self.parse_expr_opt(0)?.expect("xd");
                    let item = if let Some(item) = self.parse_assign_opt(lhs)? {
                        item
                    } else {
                        break;
                    };
                    Ok(item)
                }
                TokenType::Identifier => {
                    let lhs = self.parse_expr_opt(0)?.unwrap_or_else(|| Expression::Identifier(token.as_string()));
                    let item = if let Some(item) = self.parse_assign_opt(lhs)? {
                        item
                    } else {
                        break;
                    };
                    Ok(item)
                }
                _ => break,
            };
            items.push(item?);
        }
        Ok(items)
    }

    fn parse_assign_opt(&mut self, lhs: Expression) -> ParseResult<Option<Item>> {
        let item = if self.match_many(&['+', '=']) {
            Item::Assignment {
                lhs,
                operator: Some('+'),
                expr: self.parse_expr(0)?,
            }
        } else if self.match_one('=') {
            Item::Assignment {
                lhs,
                operator: None,
                expr: self.parse_expr(0)?,
            }
        } else {
            self.expect_one(';')?;
            return Ok(None);
        };
        self.expect_one(';')?;
        Ok(Some(item))
    }

    fn parse_expr(&mut self, precedence: isize) -> ParseResult<Expression> {
        Ok(self.parse_expr_opt(precedence)?.expect("a"))
    }

    fn parse_expr_opt(&mut self, precedence: isize) -> ParseResult<Option<Expression>> {
        let token = self.peek(0);
        let lhs = match token.get_type() {
            TokenType::Punct('-')
            | TokenType::Punct('*') => {
                self.advance();
                let operand = self.parse_expr(10)?;
                Expression::Prefix(token.get_type(), Box::new(operand))
            }
            TokenType::Identifier => {
                self.advance();
                Expression::Identifier(token.as_string())
            }
            TokenType::IntegralNumber => {
                self.advance();
                Expression::IntegralConstant(token.get_integer().unwrap())
            }
            TokenType::FloatingNumber => {
                self.advance();
                Expression::FloatingConstant(token.get_float().unwrap())
            }
            TokenType::Keyword(Keyword::True) => {
                self.advance();
                Expression::BoolConstant(true)
            }
            TokenType::Keyword(Keyword::False) => {
                self.advance();
                Expression::BoolConstant(false)
            }
            TokenType::Punct('(') => {
                self.advance();
                let mut expr = None;
                let mut values = vec![];
                loop {
                    let value = self.parse_expr(0)?;
                    if let Some(expr) = expr.take() {
                        values = vec![expr, value];
                    } else {
                        expr = Some(value);
                    }

                    if !self.match_one(',') {
                        break;
                    }
                }
                self.expect_one(')')?;

                if let Some(expr) = expr {
                    expr
                } else {
                    Expression::Tuple(values)
                }
            }
            TokenType::Punct('[') => {
                self.advance();
                let mut values = vec![];
                loop {
                    if self.match_one(']') {
                        break;
                    }
                    values.push(self.parse_expr(0)?);
                    self.match_one(',');
                }
                Expression::Array(values)
            }
            other => {
                return Ok(None);
            }
        };

        let mut expr = lhs;
        loop {
            let token = self.peek(0);

            let new_precedence = Self::get_precedence(&token);
            if new_precedence < precedence || new_precedence == precedence && Self::is_left_associative(&token) {
                break;
            }

            expr = match token.get_type() {
                TokenType::Punct('+')
                | TokenType::Punct('-')
                | TokenType::Punct('*')
                | TokenType::Punct('/') => {
                    if self.peek(1).get_type() == TokenType::Punct('=') {
                        break;
                    }

                    let op = self.advance();
                    let rhs = self.parse_expr(new_precedence)?;
                    Expression::Infix(op.get_type(), Box::new(expr), Box::new(rhs))
                }
                TokenType::Punct('<')
                | TokenType::Punct('>') => {
                    let op = self.advance();
                    let rhs = self.parse_expr(new_precedence)?;
                    Expression::Infix(op.get_type(), Box::new(expr), Box::new(rhs))
                }
                TokenType::Punct('(') => {
                    self.advance();
                    let arg = self.parse_expr(0)?;
                    self.expect_one(')')?;
                    Expression::Call(Box::new(expr), vec![arg])
                }
                other => break,
            };
        }

        Ok(Some(expr))
    }

    fn get_precedence(token: &Token) -> isize {
        match token.get_type() {
            TokenType::Punct('+') => 1,
            TokenType::Punct('-') => 1,
            TokenType::Punct('*') => 2,
            TokenType::Punct('/') => 2,
            _ => 0,
        }
    }

    fn is_left_associative(token: &Token) -> bool {
        match token.get_type() {
            TokenType::Punct('-') => true,
            _ => false,
        }
    }

    fn parse_fn(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::Fn)?;
        let identifier = self.expect_identifier()?.as_string();
        self.expect_one('(')?;
        let mut args = vec![];
        while let Some(t) = self.match_identifier() {
            self.expect_one(':')?;
            let ty = self.parse_ty()?;
            args.push(Argument {
                name: t.as_string(),
                ty,
            });
            if !self.match_one(',') {
                break;
            }
        }
        self.expect_one(')')?;

        let ty = if self.match_many(&['-', '>']) {
            Some(self.parse_ty()?)
        } else {
            None
        };

        self.expect_one('{')?;
        let body = self.parse_stmts()?;
        self.expect_one('}')?;
        Ok(Item::Function { name: identifier, args, body, ty })
    }

    fn parse_let(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::Let)?;
        let identifier = self.expect_identifier()?.as_string();
        let ty = if self.match_one(':') {
            Some(self.parse_ty()?)
        } else {
            None
        };

        let expr = if self.match_one('=') {
            Some(self.parse_expr(0)?)
        } else {
            None
        };

        self.expect_one(';')?;

        Ok(Item::Let {
            name: identifier,
            ty,
            expr,
        })
    }

    fn parse_ty(&mut self) -> ParseResult<Ty> {
        let token = self.advance();
        match token.get_type() {
            TokenType::Punct('[') => {
                let token = self.peek(0);
                let length = match token.get_type() {
                    TokenType::IntegralNumber => {
                        let length = token.get_integer().unwrap() as usize;
                        self.advance();
                        Some(length)
                    }
                    _ => None
                };
                self.expect_one(']')?;
                let ty = self.parse_ty()?;
                if let Some(length) = length {
                    Ok(Ty::Array(length, Box::new(ty)))
                } else {
                    Ok(Ty::Slice(Box::new(ty)))
                }
            }
            TokenType::Punct('*') => {
                Ok(Ty::Ptr(Box::new(self.parse_ty()?)))
            }
            TokenType::Identifier => {
                Ok(match token.as_slice() {
                    "bool" => Ty::Bool,
                    "i32" => Ty::I32,
                    "u32" => Ty::U32,
                    ud => Ty::UserDefined(ud.to_string()),
                })
            }
            TokenType::Keyword(Keyword::Fn) => {
                self.expect_one('(')?;
                let args = self.parse_tuple()?;

                let ret = if self.match_many(&['-', '>']) {
                    self.parse_ty()?
                } else {
                    Ty::Tuple(vec![])
                };

                Ok(Ty::Function(args, Box::new(ret)))
            }
            TokenType::Punct('(') => {
                let types = self.parse_tuple()?;
                Ok(Ty::Tuple(types))
            }
            _ => unimplemented!()
        }
    }

    fn parse_tuple(&mut self) -> ParseResult<Vec<Ty>> {
        let mut args = vec![];
        loop {
            if self.match_one(')') {
                break;
            }
            args.push(self.parse_ty()?);
            if !self.match_one(',') {
                self.expect_one(')')?;
                break;
            }
        }
        Ok(args)
    }

    fn parse_for(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::For)?;
        let identifier = self.expect_identifier()?.as_string();
        self.expect_keyword(Keyword::In)?;
        let expr = self.parse_expr(0)?;
        self.expect_one('{')?;
        let items = self.parse_stmts()?;
        self.expect_one('}')?;
        Ok(Item::ForIn { name: identifier.to_owned(), expr, body: items })
    }

    fn parse_if(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::If)?;
        let condition = self.parse_expr(0)?;
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
        let value = self.parse_expr(0)?;
        Ok(Item::Yield(Box::new(value)))
    }

    fn parse_return(&mut self) -> ParseResult<Item> {
        self.expect_keyword(Keyword::Return)?;
        let value = self.parse_expr(0)?;
        self.expect_one(';')?;
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
            token.get_type(),
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
            Err(_) => self.expect_punct(PunctKind::Joint, ch)
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
                }
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
