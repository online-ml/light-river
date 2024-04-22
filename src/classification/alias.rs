use num::{Float, FromPrimitive};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

pub trait FType:
    Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign + std::fmt::Debug
{
}
impl<T> FType for T where
    T: Float + FromPrimitive + AddAssign + SubAssign + MulAssign + DivAssign + std::fmt::Debug
{
}
