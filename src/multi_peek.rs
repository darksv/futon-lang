pub(crate) struct MultiPeek<T, I: Iterator<Item = T>> {
    peeked: [Option<T>; 2],
    index: usize,
    length: usize,
    input: I,
}

impl<T, I: Iterator<Item = T>> MultiPeek<T, I> {
    pub(crate) fn new(input: I) -> Self {
        MultiPeek {
            peeked: [None, None],
            index: 0,
            length: 0,
            input,
        }
    }

    /// Returns next item without consuming it
    pub(crate) fn peek(&mut self, offset: usize) -> &T {
        self.ensure_peeked(offset);
        self.peeked[(self.index + offset) % self.peeked.len()]
            .as_ref()
            .unwrap()
    }

    /// Returns next item and removes it from buffer
    pub(crate) fn advance(&mut self) -> T {
        self.ensure_peeked(0);
        let item = self.peeked[self.index].take().unwrap();
        self.length -= 1;
        self.index = (self.index + 1) % self.peeked.len();
        item
    }

    /// Ensures that next item (if any) is taken from input
    fn ensure_peeked(&mut self, offset: usize) {
        if self.length <= offset {
            self.peeked[(self.index + self.length) % self.peeked.len()] = self.input.next();
            self.length += 1;
        }
    }
}
