use std::fmt;
use std::str::CharIndices;

/// Macro used to generate Keyword enum based on collection of string slices
macro_rules! keywords {
    ($($val:expr => $name:ident),*) => {
        #[derive(Copy, Clone, PartialEq)]
        pub enum Keyword { $($name,)* }

        impl fmt::Debug for Keyword {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self {
                    $(Keyword::$name => write!(f, $val)?),*
                }
                Ok(())
            }
        }

        const KEYWORDS: [(&'static str, Keyword); count_idents!($($name),*)] = [$(($val, Keyword::$name)),*];
    }
}

/// Macro used to count idents separated by commas
macro_rules! count_idents {
    () => {0};
    ($ident:ident $(,$rest:ident)*) => {1 + count_idents!($($rest),*)};
}

keywords! {
    "let" => Let,
    "if" => If,
    "else" => Else,
    "extern" => Extern,
    "for" => For,
    "while" => While,
    "loop" => Loop,
    "fn" => Fn,
    "struct" => Struct,
    "yield" => Yield,
    "range" => Range,
    "return" => Return,
    "break" => Break,
    "true" => True,
    "false" => False,
    "in" => In
}

/// Kind of punctuation mark
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PunctKind {
    Single,
    #[allow(dead_code)]
    Joint,
}

/// Storage for values associated in a single token
#[derive(Clone, Debug, PartialEq)]
pub enum TokenValue {
    None,
    Punct(char, PunctKind),
    Identifier,
    IntegralNumber(i64),
    FloatingNumber(f64),
    String(String),
    Keyword(Keyword),
}

/// Type of the token
#[derive(Copy, Clone, PartialEq)]
pub enum TokenType {
    Punct(char),
    Identifier,
    IntegralNumber,
    FloatingNumber,
    String,
    Keyword(Keyword),
    EndOfSource,
}

impl fmt::Debug for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TokenType::Punct(ch) => write!(f, "`{:?}`", ch)?,
            TokenType::Identifier => write!(f, "identifier")?,
            TokenType::IntegralNumber => write!(f, "integral literal")?,
            TokenType::FloatingNumber => write!(f, "floating literal")?,
            TokenType::String => write!(f, "string")?,
            TokenType::Keyword(keyword) => write!(f, "`{:?} keyword`", keyword)?,
            TokenType::EndOfSource => write!(f, "end of source")?,
        }
        Ok(())
    }
}

/// Lexical unit produced by lexical analysis of source code
#[derive(Clone, PartialEq)]
pub struct Token<'a> {
    /// Value stored in the token
    value: TokenValue,
    /// Slice of the raw source with raw representation of the token
    span: SourceSpan<'a>,
}

/// Single lexical unit of the source, eg. identifier, literals, etc
impl<'a> Token<'a> {
    /// Returns raw slice of the input that represents the token in the source
    pub fn get_span(&self) -> SourceSpan<'a> {
        self.span
    }

    /// Returns line number at which exists the first char of the token
    pub fn line(&self) -> usize {
        self.span.line
    }

    /// Returns column number at which exists the first char of the token
    pub fn column(&self) -> usize {
        self.span.column
    }

    /// Returns type of the token
    pub fn get_type(&self) -> TokenType {
        match self.value {
            TokenValue::Punct(ch, _) => TokenType::Punct(ch),
            TokenValue::Identifier => TokenType::Identifier,
            TokenValue::Keyword(kw) => TokenType::Keyword(kw),
            TokenValue::IntegralNumber(_) => TokenType::IntegralNumber,
            TokenValue::FloatingNumber(_) => TokenType::FloatingNumber,
            TokenValue::None => TokenType::EndOfSource,
            TokenValue::String(_) => TokenType::String,
        }
    }

    /// Returns the char that is representing the token when it is a punctuation mark
    pub fn get_punct(&self) -> Option<(char, PunctKind)> {
        match self.value {
            TokenValue::Punct(ch, kind) => Some((ch, kind)),
            _ => None,
        }
    }

    /// Returns the integral number when token is a integer literal
    pub fn get_integer(&self) -> Option<i64> {
        match self.value {
            TokenValue::IntegralNumber(val) => Some(val),
            _ => None,
        }
    }

    /// Returns the float number when token is a float literal
    pub fn get_float(&self) -> Option<f64> {
        match self.value {
            TokenValue::FloatingNumber(val) => Some(val),
            _ => None,
        }
    }

    /// Returns a raw slice over the meaningful string value of the token
    pub fn as_slice(&'a self) -> &'a str {
        match self.value {
            TokenValue::String(ref s) => &s[..],
            _ => self.get_span().as_slice(),
        }
    }

    /// Returns an owned string over the meaningful string value of the token
    pub fn as_string(&self) -> String {
        self.as_slice().to_owned()
    }
}

impl<'a> fmt::Debug for Token<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} at {}:{}", self.value, self.line(), self.column())?;
        Ok(())
    }
}

/// Structure being context for lexical analysis
pub struct Lexer<'a> {
    /// Source to parse
    source: &'a str,
    /// Line number starting from 1 at which starts current token
    line: usize,
    /// Column number starting from 1 at which starts current token
    column: usize,
    /// Current position in the source (in bytes, not codepoints)
    position: usize,
    /// Iterator over the chars of the source
    iter: CharIndices<'a>,
    /// Recently peeked character with its position in the source (in bytes, not codepoints)
    peeked: Option<(usize, char)>,
    /// Size of the tab in number of spaces
    tab_size: u8,
}

impl<'a> Lexer<'a> {
    /// Returns lexer created from given source
    pub fn from_source(input: &'a str) -> Lexer<'a> {
        Lexer {
            source: input,
            line: 1,
            column: 1,
            position: 0,
            iter: input.char_indices(),
            peeked: None,
            tab_size: 4,
        }
    }
}

/// Custom slice that holds lexeme in the source
#[derive(Copy, Clone)]
pub struct SourceSpan<'a> {
    /// Index of the first character of the token in the source
    start: usize,
    /// Length of token (in bytes)
    length: usize,
    /// Source that span refers to
    source: &'a str,
    /// Line number the span starts at
    line: usize,
    /// Column number the span starts at
    column: usize,
}

impl<'a> SourceSpan<'a> {
    /// Returns plain slice of the source
    pub fn as_slice(&self) -> &'a str {
        if self.length > 0 {
            &self.source[self.start..self.start + self.length]
        } else {
            ""
        }
    }

    /// Returns span created from a str
    #[allow(unused)]
    pub fn from_str(str: &'a str, line: usize, column: usize) -> SourceSpan<'a> {
        SourceSpan {
            source: str,
            start: 0,
            length: str.bytes().len(),
            line,
            column,
        }
    }

    /// Checks whether current span directly adjoins with the other
    #[allow(dead_code)]
    pub fn adjoins_with(&self, other: &SourceSpan<'a>) -> bool {
        self.source.as_ptr() == other.source.as_ptr() && other.start - self.start == self.length
    }
}

impl<'a> PartialEq for SourceSpan<'a> {
    fn eq(&self, other: &SourceSpan<'a>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a> fmt::Debug for SourceSpan<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:?}",
            &self.source[self.start..self.start + self.length]
        )?;
        Ok(())
    }
}

/// Error returned by lexer
#[derive(Debug, PartialEq)]
pub enum LexerError {
    UnexpectedEndOfSource(usize, usize),
}

pub type LexerResult<T> = Result<T, LexerError>;

/// Handle for a new source span
struct SourceSpanHandle {
    start_position: usize,
    line: usize,
    column: usize,
}

impl SourceSpanHandle {
    /// Returns span handle for current lexer
    fn new(lexer: &Lexer) -> Self {
        let (line, column) = (lexer.line, lexer.column);
        Self {
            start_position: lexer.position,
            line,
            column,
        }
    }

    /// Returns new span created from handle, ending at current lexer position
    fn get_span<'a>(self, lexer: &Lexer<'a>) -> SourceSpan<'a> {
        let end_position = lexer.position;
        SourceSpan {
            start: self.start_position,
            length: end_position - self.start_position,
            source: lexer.source,
            line: self.line,
            column: self.column,
        }
    }
}

impl<'a> Lexer<'a> {
    /// Returns next token from the source
    pub fn next(&mut self) -> LexerResult<Token<'a>> {
        self.skip_space();
        let token = match self.peek(0) {
            Some(ch) if self.can_start_identifier(ch) => self.match_keyword_or_identifier()?,
            Some(ch) if ch.is_digit(10) => self.match_number()?,
            Some('"') => self.match_string()?,
            Some(ch) => self.match_punct(ch)?,
            None => self.match_end_of_source()?,
        };
        Ok(token)
    }

    /// Checks whether given character can be a starting character of the identifier
    #[inline]
    fn can_start_identifier(&self, ch: char) -> bool {
        match ch {
            _ if ch.is_alphabetic() => true,
            '_' => true,
            _ => false,
        }
    }

    /// Checks whether given character can be in the identifier
    #[inline]
    fn can_be_in_identifier(&self, ch: char) -> bool {
        match ch {
            _ if ch.is_alphanumeric() => true,
            '_' => true,
            _ => false,
        }
    }

    /// Skips all whitespaces
    fn skip_space(&mut self) {
        while let Some(ch) = self.peek(0) {
            if ch.is_whitespace() {
                self.advance().unwrap();
            } else {
                break;
            }
        }
    }

    /// Returns current token when it is a keyword or an identifier
    fn match_keyword_or_identifier(&mut self) -> LexerResult<Token<'a>> {
        let handle = self.begin_span();
        while let Some(ch) = self.peek(0) {
            if self.can_be_in_identifier(ch) {
                self.advance().unwrap();
            } else {
                break;
            }
        }
        let span = handle.get_span(self);
        let kind = match self.get_keyword(span.as_slice()) {
            Some(keyword) => TokenValue::Keyword(keyword),
            None => TokenValue::Identifier,
        };
        Ok(Token { value: kind, span })
    }

    /// Returns current token when it is a string literal
    fn match_string(&mut self) -> LexerResult<Token<'a>> {
        let handle = self.begin_span();
        // '"'
        self.advance().unwrap();
        let mut string = String::new();
        loop {
            match self.peek(0) {
                Some(ch) => match ch {
                    '"' => {
                        self.advance().unwrap();
                        break;
                    }
                    ch => {
                        string.push(ch);
                        self.advance().unwrap();
                    }
                },
                None => return Err(LexerError::UnexpectedEndOfSource(self.line, self.column)),
            }
        }
        Ok(Token {
            value: TokenValue::String(string),
            span: handle.get_span(self),
        })
    }

    /// Returns current token when it is built of a single punctuation mark
    fn match_punct(&mut self, first: char) -> LexerResult<Token<'a>> {
        let handle = self.begin_span();
        let span = {
            self.advance().unwrap();
            handle.get_span(self)
        };

        let is_joint = self
            .peek(0)
            .map(|c| c.is_ascii_punctuation())
            .unwrap_or(false);
        let kind = if is_joint {
            PunctKind::Joint
        } else {
            PunctKind::Single
        };

        Ok(Token {
            value: TokenValue::Punct(first, kind),
            span,
        })
    }

    fn match_end_of_source(&mut self) -> LexerResult<Token<'a>> {
        Ok(Token {
            value: TokenValue::None,
            span: self.begin_span().get_span(self),
        })
    }

    /// Returns current token when it is a number
    fn match_number(&mut self) -> LexerResult<Token<'a>> {
        let handle = self.begin_span();
        let mut is_floating = false;
        self.advance_while_digits();
        if let Some('.') = self.peek(0) {
            self.advance().unwrap();
            self.advance_while_digits();
            is_floating = true;
        }
        let has_exponent = match self.peek(0) {
            Some('e') | Some('E') => {
                self.advance().unwrap();
                true
            }
            _ => false,
        };
        if has_exponent {
            match self.peek(0) {
                Some('+') | Some('-') => {
                    self.advance().unwrap();
                }
                _ => (),
            }
            self.advance_while_digits();
            is_floating = true;
        };

        let span = handle.get_span(self);
        let value = if is_floating {
            let parsed = span.as_slice().parse().expect("valid float");
            TokenValue::FloatingNumber(parsed)
        } else {
            let parsed = span.as_slice().parse().expect("valid integer");
            TokenValue::IntegralNumber(parsed)
        };

        Ok(Token { value, span })
    }

    fn advance_while_digits(&mut self) {
        #[allow(clippy::while_let_loop)]
        loop {
            match self.peek(0) {
                Some('0'..='9' | '_') => self.advance().unwrap(),
                _ => break,
            };
        }
    }

    /// Returns a keyword when a given text can be one
    fn get_keyword(&mut self, text: &str) -> Option<Keyword> {
        KEYWORDS
            .iter()
            .filter(|&&(lex, _val)| lex == text)
            .map(|&(_lex, val)| val)
            .next()
    }

    /// Returns next nth character without advancing the iterator
    #[inline]
    fn peek(&mut self, nth: usize) -> Option<char> {
        if nth != 0 {
            unimplemented!();
        }
        self.ensure_peeked();
        self.peeked.map(|x| x.1)
    }

    /// Returns next nth character index without advancing the iterator
    #[inline]
    fn peek_index(&mut self, nth: usize) -> Option<usize> {
        if nth != 0 {
            unimplemented!();
        }
        self.ensure_peeked();
        self.peeked.map(|x| x.0)
    }

    /// Loads next character when there is one
    #[inline]
    fn ensure_peeked(&mut self) {
        if self.peeked.is_none() {
            self.peeked = self.iter.next();
        }
    }

    // Advances the iterator and returns consumed character
    fn advance(&mut self) -> Option<char> {
        self.peek(0)?;
        let (_idx, ch) = self.peeked.take()?;
        match ch {
            '\n' => {
                self.column = 1;
                self.line += 1;
            }
            '\t' => {
                self.column += self.tab_size as usize;
            }
            _ => {
                self.column += 1;
            }
        }
        if let Some(idx) = self.peek_index(0) {
            self.position = idx;
        } else {
            self.position = self.source.len()
        }
        Some(ch)
    }

    /// Returns a span handle for current lexer position
    fn begin_span(&mut self) -> SourceSpanHandle {
        SourceSpanHandle::new(self)
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::{Keyword, Lexer, LexerError, SourceSpan, Token, TokenType, TokenValue};

    macro_rules! assert_token_type_eq {
        ($actual:expr, $expected:expr, $line:expr, $column:expr) => {
            let token = $actual;
            let token = token.as_ref();
            assert_eq!(token.map(|t| t.get_type()), Ok($expected), "token type");
            assert_eq!(token.map(|t| t.line()), Ok($line), "line number");
            assert_eq!(token.map(|t| t.column()), Ok($column), "column number");
        };
    }

    macro_rules! assert_token_eq {
        ($actual:expr, $value:expr, $span:expr, $line:expr, $column:expr) => {
            let token = $actual;
            let token = token.as_ref();
            assert_eq!(
                token,
                Ok(&Token {
                    value: $value,
                    span: SourceSpan::from_str($span, $line, $column),
                }),
                "token value"
            );
            assert_eq!(token.map(|t| t.line()), Ok($line), "line number");
            assert_eq!(token.map(|t| t.column()), Ok($column), "column number");
        };
    }

    #[test]
    fn empty() {
        let mut lex = Lexer::from_source("");
        assert_token_type_eq!(lex.next(), TokenType::EndOfSource, 1, 1);
    }

    #[test]
    fn only_whitespace() {
        let mut lex = Lexer::from_source("  \t\n   \n");
        assert_token_type_eq!(lex.next(), TokenType::EndOfSource, 3, 1);
    }

    #[test]
    fn single_punctuation_mark() {
        let mut lex = Lexer::from_source("(");
        assert_token_type_eq!(lex.next(), TokenType::Punct('('), 1, 1);
        assert_token_eq!(lex.next(), TokenValue::None, "", 1, 2);
    }

    #[test]
    fn single_keyword() {
        let mut lex = Lexer::from_source("if");
        assert_token_eq!(lex.next(), TokenValue::Keyword(Keyword::If), "if", 1, 1);
        assert_token_eq!(lex.next(), TokenValue::None, "", 1, 3);
    }

    #[test]
    fn single_identifier() {
        let mut lex = Lexer::from_source("iff");
        assert_token_eq!(lex.next(), TokenValue::Identifier, "iff", 1, 1);
        assert_token_eq!(lex.next(), TokenValue::None, "", 1, 4);
    }

    #[test]
    fn single_identifier_surrounded_by_whitespace() {
        let mut lex = Lexer::from_source(" iff  ");
        assert_token_eq!(lex.next(), TokenValue::Identifier, "iff", 1, 2);
        assert_token_eq!(lex.next(), TokenValue::None, "", 1, 7);
    }

    #[test]
    fn single_integral_number() {
        let mut lex = Lexer::from_source("1234");
        assert_token_type_eq!(lex.next(), TokenType::IntegralNumber, 1, 1);
        assert_token_type_eq!(lex.next(), TokenType::EndOfSource, 1, 5);
    }

    #[test]
    fn single_floating_number() {
        let mut lex = Lexer::from_source("12.34");
        assert_token_type_eq!(lex.next(), TokenType::FloatingNumber, 1, 1);
        assert_token_type_eq!(lex.next(), TokenType::EndOfSource, 1, 6);
    }

    #[test]
    fn single_string() {
        let mut lex = Lexer::from_source("\"simple string\"");
        assert_token_type_eq!(lex.next(), TokenType::String, 1, 1);
        assert_token_type_eq!(lex.next(), TokenType::EndOfSource, 1, 16);
    }

    #[test]
    fn single_unterminated_string() {
        let mut lex = Lexer::from_source("\"simple");
        assert_eq!(lex.next(), Err(LexerError::UnexpectedEndOfSource(1, 8)));
    }
}
