use std::ops::{Deref, DerefMut};

const CACHE_LINE: usize = 64;

#[derive(Debug)]
pub struct CacheAlignedVec<T> {
    data: std::ptr::NonNull<T>,
    length: usize,
    capacity: usize,
}

fn layout<T>(capacity: usize) -> std::alloc::Layout {
    assert!(
        size_of::<T>() != 0,
        "CacheAlignedVec does not support zero-sized types"
    );
    std::alloc::Layout::array::<T>(capacity)
        .expect("Capacity overflow")
        .align_to(CACHE_LINE)
        .expect("Invalid layout alignment")
}

impl<T> CacheAlignedVec<T> {
    pub fn new() -> Self {
        let capacity = 1;
        let layout = layout::<T>(capacity);
        let data = unsafe { std::alloc::alloc(layout) };
        if data.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let data = unsafe { std::ptr::NonNull::new_unchecked(data as *mut T) };

        Self {
            data,
            length: 0,
            capacity,
        }
    }

    fn reserve(&mut self, additional: usize) {
        let required = self
            .length
            .checked_add(additional)
            .expect("capacity overflow");
        if required <= self.capacity {
            return;
        }

        let new_capacity = required
            .checked_next_power_of_two()
            .expect("capacity overflow");
        let old_layout = layout::<T>(self.capacity);
        let new_layout = layout::<T>(new_capacity);
        let data = unsafe {
            std::alloc::realloc(
                self.data.as_ptr() as *mut u8,
                old_layout,
                new_layout.size(),
            )
        };
        if data.is_null() {
            std::alloc::handle_alloc_error(new_layout);
        }

        self.data = unsafe { std::ptr::NonNull::new_unchecked(data as *mut T) };
        self.capacity = new_capacity;
    }

    pub fn push(&mut self, item: T) {
        self.reserve(1);
        unsafe {
            self.data.as_ptr().add(self.length).write(item);
        }
        self.length += 1;
    }

    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        if other.is_empty() {
            return;
        }

        self.reserve(other.len());
        for item in other {
            unsafe {
                self.data.as_ptr().add(self.length).write(item.clone());
            }
            self.length += 1;
        }
    }
}

impl<T> Default for CacheAlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for CacheAlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                for i in 0..self.length {
                    self.data.as_ptr().add(i).drop_in_place();
                }
                let layout = layout::<T>(self.capacity);
                std::alloc::dealloc(self.data.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Deref for CacheAlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.length) }
    }
}

impl<T> DerefMut for CacheAlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.length) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    fn assert_cache_aligned<T>(v: &CacheAlignedVec<T>) {
        assert_eq!(
            v.as_ptr() as usize % CACHE_LINE,
            0,
            "data pointer is not {CACHE_LINE}-byte aligned"
        );
    }

    #[test]
    fn new_is_empty_and_aligned() {
        let v = CacheAlignedVec::<i32>::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert_cache_aligned(&v);
    }

    #[test]
    fn default_is_empty_and_aligned() {
        let v = CacheAlignedVec::<i32>::default();
        assert!(v.is_empty());
        assert_cache_aligned(&v);
    }

    #[test]
    fn push_preserves_order_across_growth() {
        let mut v = CacheAlignedVec::new();
        for i in 0..128_i32 {
            v.push(i);
            assert_eq!(v.len(), i as usize + 1);
            assert_cache_aligned(&v);
        }
        let expected: Vec<i32> = (0..128).collect();
        assert_eq!(&v[..], &expected[..]);
    }

    #[test]
    fn push_non_copy_values_survive_realloc() {
        let mut v = CacheAlignedVec::new();
        for i in 0..64 {
            v.push(format!("item-{i}"));
            assert_cache_aligned(&v);
        }
        for (i, value) in v.iter().enumerate() {
            assert_eq!(value, &format!("item-{i}"));
        }
    }

    #[test]
    fn extend_appends_and_ignores_empty() {
        let mut v = CacheAlignedVec::new();
        v.extend_from_slice(&[1, 2, 3]);
        v.extend_from_slice(&[]);
        v.extend_from_slice(&[4, 5, 6, 7, 8]);
        assert_eq!(&v[..], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_cache_aligned(&v);
    }

    #[test]
    fn extend_crosses_capacity_boundaries() {
        let mut v = CacheAlignedVec::new();
        for chunk in [vec![1], vec![2, 3], vec![4, 5, 6, 7], vec![8; 16]] {
            v.extend_from_slice(&chunk);
            assert_cache_aligned(&v);
        }
        let expected: Vec<i32> = std::iter::once(1)
            .chain([2, 3])
            .chain([4, 5, 6, 7])
            .chain([8; 16])
            .collect();
        assert_eq!(&v[..], &expected[..]);
    }

    #[test]
    fn deref_mut_mutates_in_place() {
        let mut v = CacheAlignedVec::new();
        v.extend_from_slice(&[3, 1, 2]);
        v.sort_unstable();
        assert_eq!(&v[..], &[1, 2, 3]);
        for (i, value) in v.iter_mut().enumerate() {
            *value = i as i32;
        }
        assert_eq!(&v[..], &[0, 1, 2]);
        assert_cache_aligned(&v);
    }

    struct DropCounter(Rc<Cell<usize>>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn drops_each_pushed_element_once() {
        let drops = Rc::new(Cell::new(0));
        {
            let mut v = CacheAlignedVec::new();
            for _ in 0..100 {
                v.push(DropCounter(Rc::clone(&drops)));
            }
            assert_eq!(v.len(), 100);
            assert_eq!(drops.get(), 0);
        }
        assert_eq!(drops.get(), 100);
    }

    struct CloneCounter(Rc<Cell<usize>>);

    impl Clone for CloneCounter {
        fn clone(&self) -> Self {
            self.0.set(self.0.get() + 1);
            Self(Rc::clone(&self.0))
        }
    }

    #[test]
    fn extend_clones_each_element_once() {
        let clones = Rc::new(Cell::new(0));
        let source = [
            CloneCounter(Rc::clone(&clones)),
            CloneCounter(Rc::clone(&clones)),
            CloneCounter(Rc::clone(&clones)),
        ];
        let mut v = CacheAlignedVec::new();

        v.extend_from_slice(&source);

        assert_eq!(v.len(), source.len());
        assert_eq!(clones.get(), source.len());
        assert_cache_aligned(&v);
    }

    #[test]
    fn extend_clones_non_copy_values() {
        let source = [String::from("alpha"), String::from("beta")];
        let mut v = CacheAlignedVec::new();

        v.extend_from_slice(&source);
        assert_eq!(&v[..], &source[..]);

        v[0].push_str("-copy");
        assert_eq!(source[0], "alpha");
        assert_eq!(v[0], "alpha-copy");
    }

    #[test]
    #[should_panic(expected = "zero-sized types")]
    fn rejects_zero_sized_types() {
        let _ = CacheAlignedVec::<()>::new();
    }
}
