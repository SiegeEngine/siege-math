
use num_traits::NumCast;
use std::ops::{Mul, Div, Add, Sub, Neg};
use float_cmp::{Ulps, ApproxEqUlps};
use FullFloat;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Angle<F>(F);

impl<F: FullFloat> Angle<F>
{
    #[inline]
    pub fn new_radians(radians: F) -> Angle<F>
    {
        Angle::<F>::from_radians(radians)
    }

    pub fn from_radians(radians: F) -> Angle<F>
    {
        Angle(radians)
    }

    pub fn as_radians(&self) -> F {
        self.0
    }

    #[inline]
    pub fn new_degrees(degrees: F) -> Angle<F>
    {
        Angle::<F>::from_degrees(degrees)
    }

    pub fn from_degrees(degrees: F) -> Angle<F>
    {
        let one_eighty: F = NumCast::from(180.0_f32).unwrap();
        Angle(F::PI() * degrees / one_eighty)
    }

    #[inline]
    pub fn new_cycles(cycles: F) -> Angle<F>
    {
        Angle::<F>::from_cycles(cycles)
    }

    pub fn from_cycles(cycles: F) -> Angle<F>
    {
        let two: F = NumCast::from(2.0_f32).unwrap();
        Angle(two * F::PI() * cycles)
    }

    pub fn as_degrees(&self) -> F {
        let one_eighty: F = NumCast::from(180.0_f32).unwrap();
        self.0 * one_eighty / F::PI()
    }

    pub fn as_cycles(&self) -> F {
        let two: F = NumCast::from(2.0_f32).unwrap();
        self.0 / (two * F::PI())
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

impl<F: FullFloat> ApproxEqUlps for Angle<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.0.approx_eq_ulps(&other.0, ulps)
    }
}
