
use std::ops::{Mul, Div, Add, Sub};
use float_cmp::{Ulps, ApproxEqUlps};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Angle<F>(F);

impl<F: Copy> Angle<F> {
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
}

impl Angle<f32> {
    #[inline]
    pub fn new_degrees(degrees: f32) -> Angle<f32>
    {
        Angle::<f32>::from_degrees(degrees)
    }

    pub fn from_degrees(degrees: f32) -> Angle<f32>
    {
        Angle(::std::f32::consts::PI * degrees / 180.0)
    }

    #[inline]
    pub fn new_cycles(cycles: f32) -> Angle<f32>
    {
        Angle::<f32>::from_cycles(cycles)
    }

    pub fn from_cycles(cycles: f32) -> Angle<f32>
    {
        Angle(2.0 * ::std::f32::consts::PI * cycles)
    }

    pub fn as_degrees(&self) -> f32 {
        self.0 * 180.0 / ::std::f32::consts::PI
    }

    pub fn as_cycles(&self) -> f32 {
        self.0 / (2.0 * ::std::f32::consts::PI)
    }
}

impl Angle<f64> {
    #[inline]
    pub fn new_degrees(degrees: f64) -> Angle<f64>
    {
        Angle::<f64>::from_degrees(degrees)
    }

    pub fn from_degrees(degrees: f64) -> Angle<f64>
    {
        Angle(::std::f64::consts::PI * degrees / 180.0)
    }

    #[inline]
    pub fn new_cycles(cycles: f64) -> Angle<f64>
    {
        Angle::<f64>::from_cycles(cycles)
    }

    pub fn from_cycles(cycles: f64) -> Angle<f64>
    {
        Angle(2.0 * ::std::f64::consts::PI * cycles)
    }

    pub fn as_degrees(&self) -> f64 {
        self.0 * 180.0 / ::std::f64::consts::PI
    }

    pub fn as_cycles(&self) -> f64 {
        self.0 / (2.0 * ::std::f64::consts::PI)
    }
}

impl<F: Mul<F,Output=F>> Mul<F> for Angle<F> {
    type Output = Angle<F>;

    fn mul(self, rhs: F) -> Angle<F> {
        Angle(self.0 * rhs)
    }
}

impl<F: Div<F,Output=F>> Div<F> for Angle<F> {
    type Output = Angle<F>;

    fn div(self, rhs: F) -> Angle<F> {
        Angle(self.0 / rhs)
    }
}

impl<F: Add<F,Output=F>> Add<Angle<F>> for Angle<F> {
    type Output = Angle<F>;

    fn add(self, rhs: Angle<F>) -> Angle<F> {
        Angle(self.0 + rhs.0)
    }
}

impl<F: Sub<F,Output=F>> Sub<Angle<F>> for Angle<F> {
    type Output = Angle<F>;

    fn sub(self, rhs: Angle<F>) -> Angle<F> {
        Angle(self.0 - rhs.0)
    }
}

impl<F: Ulps + ApproxEqUlps<Flt=F>> ApproxEqUlps for Angle<F> {
    type Flt = F;

    fn approx_eq_ulps(&self, other: &Self, ulps: <<F as ApproxEqUlps>::Flt as Ulps>::U) -> bool {
        self.0.approx_eq_ulps(&other.0, ulps)
    }
}
