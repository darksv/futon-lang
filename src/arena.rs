use std::cell::RefCell;

const BLOCK_SIZE: usize = 4096;

pub struct Arena<T> {
    blocks: RefCell<Blocks<T>>,
}

struct Blocks<T> {
    current: Vec<T>,
    rest: Vec<Vec<T>>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena {
            blocks: RefCell::new(Blocks {
                current: Vec::with_capacity(BLOCK_SIZE),
                rest: vec![],
            }),
        }
    }
}

impl<T> Arena<T> {
    pub fn alloc(&self, value: T) -> &T {
        let mut blocks = self.blocks.borrow_mut();
        if blocks.current.len() == BLOCK_SIZE {
            let block = std::mem::replace(&mut blocks.current, Vec::with_capacity(BLOCK_SIZE));
            blocks.rest.push(block);
        }

        let offset = blocks.current.len();
        blocks.current.push(value);
        unsafe {
            // It is safe because:
            // 1. item behind this pointer is initialized
            // 2. there does not exist any shared reference to this item
            // 3. the backing storage is never reallocated
            &*blocks.current.as_ptr().add(offset)
        }
    }

    pub fn find(&self, other: &T) -> Option<&T>
    where
        T: Eq,
    {
        let blocks = self.blocks.borrow();
        blocks
            .current
            .iter()
            .chain(blocks.rest.iter().flatten())
            .find(|it| *it == other)
            .map(|it| unsafe {
                // It is safe because:
                // 1. there does not exist any shared reference to this item
                // 2. the backing storage is never reallocated
                &*(it as *const _)
            })
    }
}
