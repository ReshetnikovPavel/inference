pub struct Mat<T> {
    data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Mat<T> {
    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}
