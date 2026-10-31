const CACHE_LINE: usize = 64;

pub struct CacheAlignedVecI8 {
    data: Vec<i8>,
    offset: usize,
}

impl CacheAlignedVecI8 {
    #[inline]
    pub fn new() -> Self {
        let mut data = Vec::with_capacity(CACHE_LINE);
        let base = data.as_ptr() as usize;
        let offset = (CACHE_LINE - base % CACHE_LINE) % CACHE_LINE;
        data.extend_from_slice(&vec![0; offset]);
        Self { data, offset }
    }

    #[inline]
    pub fn extend_from_slice(&mut self, other: &[i8]) {
        self.data.extend_from_slice(other);
    }

    #[inline]
    pub fn data(&self) -> &[i8] {
        &self.data[self.offset..self.len()]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len() - self.offset
    }
}
