use core::fmt;
use std::convert::TryInto;
use std::fmt::Formatter;
use std::marker::PhantomData;

#[derive(Copy, Clone)]
pub struct Single;

#[derive(Copy, Clone)]
pub struct Many(u32);

#[derive(Debug)]
pub struct IndexArena<T> {
    values: Vec<T>,
}

impl<T> Default for IndexArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> IndexArena<T> {
    fn new() -> Self {
        Self {
            values: vec![],
        }
    }
    pub(crate) fn alloc(&mut self, value: T) -> Handle<T, Single> {
        let idx = self.values.len().try_into().expect("too many items??");
        self.values.push(value);
        Handle(idx, PhantomData, Single)
    }

    pub(crate) fn alloc_many(&mut self, values: impl ExactSizeIterator<Item=T>) -> Handle<T, Many> {
        let idx = self.values.len().try_into().expect("too many items??");
        let count = values.len().try_into().expect("wtf");
        self.values.extend(values);
        Handle(idx, PhantomData, Many(count))
    }

    #[track_caller]
    pub(crate) fn resolve(&self, handle: Handle<T, Single>) -> &T {
        self.values.get(handle.0 as usize).unwrap()
    }
}

pub struct Handle<T, Meta = Single>(u32, PhantomData<T>, Meta);

impl<T, Meta: Copy + Clone> Clone for Handle<T, Meta> {
    fn clone(&self) -> Self {
        Self(self.0, self.1, self.2)
    }
}

impl<T, Meta: Copy> Copy for Handle<T, Meta> {}


impl<T> fmt::Debug for Handle<T, Single> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle(#{})", self.0)
    }
}

impl<T> fmt::Debug for Handle<T, Many> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle(#{}/{})", self.0, self.2.0)
    }
}

pub struct IntoIter<T> {
    base: u32,
    left: u32,
    _type: PhantomData<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = Handle<T, Single>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left == 0 {
            return None;
        }

        let next = Handle(self.base, PhantomData, Single);
        self.base += 1;
        self.left -= 1;

        Some(next)
    }
}

impl<T> IntoIterator for Handle<T, Many> {
    type Item = Handle<T, Single>;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { base: self.0, left: self.2.0, _type: PhantomData }
    }
}

impl<T> Handle<T, Many> {
    pub(crate) fn len(&self) -> usize {
        self.0 as _
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use crate::index_arena::IndexArena;

    #[test]
    fn foo() {
        let mut arena = IndexArena::new();
        let a = arena.alloc(12);
        let b = arena.alloc(32);
        let c = arena.alloc_many([21, 37].into_iter());
        let mut it = c.into_iter();
        let c1 = it.next().unwrap();
        let c2 = it.next().unwrap();
        assert!(it.next().is_none());

        assert_eq!(arena.resolve(a), &12);
        assert_eq!(arena.resolve(b), &32);
        assert_eq!(arena.resolve(c1), &21);
        assert_eq!(arena.resolve(c2), &37);
    }
}


