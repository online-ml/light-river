use ndarray::ScalarOperand;
use num::{Float, FromPrimitive};
use std::fmt;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

pub trait FType:
    Float
    + FromPrimitive
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + fmt::Debug
    + fmt::Display
    + ScalarOperand
{
}

impl<T> FType for T where
    T: Float
        + FromPrimitive
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + fmt::Debug
        + fmt::Display
        + ScalarOperand
{
}
