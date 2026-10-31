use std::ops::{Deref, DerefMut};

const CACHE_LINE: usize = 64;

#[derive(Debug)]
pub struct CacheAlignedVec<T> {
    data: std::ptr::NonNull<T>,
    length: usize,
    capacity: usize,
}

fn layout<T>(capacity: usize) -> std::alloc::Layout {
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

    pub fn push(&mut self, item: T) {
        if self.length >= self.capacity {
            let layout = layout::<T>(self.capacity);
            self.capacity *= 2;
            let data = unsafe {
                std::alloc::realloc(
                    self.data.as_ptr() as *mut u8,
                    layout,
                    self.capacity * size_of::<T>(),
                )
            };
            if data.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            self.data = unsafe { std::ptr::NonNull::new_unchecked(data as *mut T) };
        }
        unsafe {
            self.data.as_ptr().add(self.length).write(item);
        }
        self.length += 1;
    }

    pub fn extend_from_slice(&mut self, other: &[T]) {
        if other.is_empty() {
            return;
        }
        if self.length + other.len() >= self.capacity {
            let layout = layout::<T>(self.capacity);
            self.capacity = (self.length + other.len()).next_power_of_two();
            let data = unsafe {
                std::alloc::realloc(
                    self.data.as_ptr() as *mut u8,
                    layout,
                    self.capacity * size_of::<T>(),
                )
            };
            if data.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            self.data = unsafe { std::ptr::NonNull::new_unchecked(data as *mut T) };
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                other.as_ptr(),
                self.data.as_ptr().add(self.length),
                other.len(),
            )
        };
        self.length += other.len();
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
