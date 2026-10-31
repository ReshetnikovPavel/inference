const CACHE_LINE: usize = 64;

pub struct CacheAlignedVec<T> {
    data: Vec<T>,
    offset: usize,
}

impl<T: Clone + Default> CacheAlignedVec<T> {
    #[inline]
    pub fn new() -> Self {
        let mut data = Vec::with_capacity(CACHE_LINE);
        let base = data.as_ptr() as usize;
        let offset = (CACHE_LINE - base % CACHE_LINE) % CACHE_LINE;
        data.extend_from_slice(&vec![T::default(); offset]);
        Self { data, offset }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.data.extend_from_slice(other);
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data[self.offset..self.len()]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len() - self.offset
    }
}
