
use num_traits::NumCast;
use serde::{Serialize, Deserialize};
use std::ops::{Mul, Div, Add, Sub, Neg};
use float_cmp::{Ulps, ApproxEq};
use FullFloat;
use vector::Vec2;

/// A type for representing an angle, without needing to remember if it is
/// denominated in Radians, Degrees, or otherwise.  Angles are NOT automatically
/// normalized -- often you want to know if something spun around twice.
// internally stored as radians
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(Serialize, Deserialize)]
pub struct Angle<F>(F);

impl<F: FullFloat> Angle<F>
{
    /// Create an angle from radians
    #[inline]
    pub fn new_radians(radians: F) -> Angle<F>
    {
        Angle::<F>::from_radians(radians)
    }

    /// Create an angle from radians
    pub fn from_radians(radians: F) -> Angle<F>
    {
        Angle(radians)
    }

    /// Get the value of the angle as radians
    pub fn as_radians(&self) -> F {
        self.0
    }

    /// Create an angle from degrees
    #[inline]
    pub fn new_degrees(degrees: F) -> Angle<F>
    {
        Angle::<F>::from_degrees(degrees)
    }

    /// Create an angle from degrees
    pub fn from_degrees(degrees: F) -> Angle<F>
    {
        let one_eighty: F = NumCast::from(180.0_f32).unwrap();
        Angle(F::PI() * degrees / one_eighty)
    }

    /// Get the value of the angle as degrees
    pub fn as_degrees(&self) -> F {
        let one_eighty: F = NumCast::from(180.0_f32).unwrap();
        self.0 * one_eighty / F::PI()
    }

    /// Create an angle from cycles (1 cycle is a full circle)
    #[inline]
    pub fn new_cycles(cycles: F) -> Angle<F>
    {
        Angle::<F>::from_cycles(cycles)
    }

    /// Create an angle from cycles (1 cycle is a full circle)
    pub fn from_cycles(cycles: F) -> Angle<F>
    {
        let two: F = NumCast::from(2.0_f32).unwrap();
        Angle(two * F::PI() * cycles)
    }

    /// Get the value of the angle as number of cycles (full circles)
    pub fn as_cycles(&self) -> F {
        let two: F = NumCast::from(2.0_f32).unwrap();
        self.0 / (two * F::PI())
    }

    /// Get the angle that a given vector points in relative to the x-axis
    /// and going counterclockwise.
    /// This ranges from -PI to PI, and all 4 quadrants are properly handled.
    pub fn of_vector(vec: &Vec2<F>) -> Angle<F>
    {
        Angle(vec.y.atan2(vec.x))
    }

    /// Normalize to within the range of 0 to 2*PI
    pub fn normalize(&mut self) {
        let two: F = NumCast::from(2.0_f32).unwrap();
        let twopi = two * F::PI();
        let zero: F = NumCast::from(0.0_f32).unwrap();
        // Remainder within -twopi ... +twopi
        self.0 = self.0 % twopi;
        if self.0 < zero { self.0 += twopi; }
    }
}

impl<F: FullFloat> Mul<F> for Angle<F>
{
    type Output = Angle<F>;

    fn mul(self, rhs: F) -> Angle<F> {
        Angle(self.0 * rhs)
    }
}

impl<F: FullFloat> Div<F> for Angle<F>
{
    type Output = Angle<F>;

    fn div(self, rhs: F) -> Angle<F> {
        Angle(self.0 / rhs)
    }
}

impl<F: FullFloat> Add<Angle<F>> for Angle<F>
{
    type Output = Angle<F>;

    fn add(self, rhs: Angle<F>) -> Angle<F> {
        Angle(self.0 + rhs.0)
    }
}

impl<F: FullFloat> Sub<Angle<F>> for Angle<F>
{
    type Output = Angle<F>;

    fn sub(self, rhs: Angle<F>) -> Angle<F> {
        Angle(self.0 - rhs.0)
    }
}

impl<F: FullFloat> Neg for Angle<F> {
    type Output = Angle<F>;

    fn neg(self) -> Angle<F> {
        Angle(-self.0)
    }
}

impl<F: FullFloat> ApproxEq for Angle<F> {
    type Flt = F;

    fn approx_eq(&self, other: &Self,
                 epsilon: <F as ApproxEq>::Flt,
                 ulps: <<F as ApproxEq>::Flt as Ulps>::U) -> bool
    {
        self.0.approx_eq(&other.0, epsilon, ulps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;
    use std::f32::EPSILON;
    use vector::Vec2;

    #[test]
    fn test_radians() {
        let f: f32 = 1.234;
        let a = Angle::from_radians(f);
        assert_eq!(a.as_radians(), f);
    }

    #[test]
    fn test_degrees() {
        let f: f32 = 1.234;
        let a = Angle::from_degrees(f);
        assert_eq!(a.as_degrees(), f);
    }

    #[test]
    fn test_cycles() {
        let f: f32 = 1.234;
        let a = Angle::from_cycles(f);
        assert_eq!(a.as_cycles(), f);
    }

    #[test]
    fn test_relations() {
        let h1 = Angle::from_radians(PI);
        let h2 = Angle::from_degrees(180.0);
        let h3 = Angle::from_cycles(0.5);
        assert!(h1.approx_eq(&h2, 2.0 * EPSILON, 2));
        assert!(h1.approx_eq(&h3, 2.0 * EPSILON, 2));
        assert!(h2.approx_eq(&h3, 2.0 * EPSILON, 2));
    }

    #[test]
    fn test_vector_angle() {
        let q1 = Vec2::new(1.0, 1.0);
        let q2 = Vec2::new(-1.0, 1.0);
        let q3 = Vec2::new(-1.0, -1.0);
        let q4 = Vec2::new(1.0, -1.0);

        assert!(Angle::of_vector(&q1).approx_eq(
            &Angle::from_cycles(1.0/8.0), 2.0 * EPSILON, 2));
        assert!(Angle::of_vector(&q2).approx_eq(
            &Angle::from_cycles(3.0/8.0), 2.0 * EPSILON, 2));
        assert!(Angle::of_vector(&q3).approx_eq(
            &Angle::from_cycles(-3.0/8.0), 2.0 * EPSILON, 2));
        assert!(Angle::of_vector(&q4).approx_eq(
            &Angle::from_cycles(-1.0/8.0), 2.0 * EPSILON, 2));
    }

    #[test]
    fn test_normalize() {
        let mut a1 = Angle::from_degrees(370.0_f32);
        a1.normalize();
        assert!(a1.as_degrees().approx_eq(&10.0_f32, 2.0 * EPSILON, 2));

        let mut a1 = Angle::from_degrees(-370.0_f32);
        a1.normalize();
        assert!(a1.as_degrees().approx_eq(&350.0_f32, 2.0 * EPSILON, 2));
    }
}
