
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub struct Angle<F>(F);

impl Angle<f32> {
    pub fn new_radians(radians: f32) -> Angle<f32>
    {
        Angle(radians)
    }

    pub fn new_degrees(degrees: f32) -> Angle<f32>
    {
        Angle(::std::f32::consts::PI * degrees / 180.0)
    }

    pub fn new_cycles(cycles: f32) -> Angle<f32>
    {
        Angle(2.0 * ::std::f32::consts::PI * cycles)
    }

    pub fn as_radians(&self) -> f32 {
        self.0
    }

    pub fn as_degrees(&self) -> f32 {
        self.0 * 180.0 / ::std::f32::consts::PI
    }

    pub fn as_cycles(&self) -> f32 {
        self.0 / (2.0 * ::std::f32::consts::PI)
    }
}

impl Angle<f64> {
    pub fn new_radians(radians: f64) -> Angle<f64>
    {
        Angle(radians)
    }

    pub fn new_degrees(degrees: f64) -> Angle<f64>
    {
        Angle(::std::f64::consts::PI * degrees / 180.0)
    }

    pub fn new_cycles(cycles: f64) -> Angle<f64>
    {
        Angle(2.0 * ::std::f64::consts::PI * cycles)
    }

    pub fn as_radians(&self) -> f64 {
        self.0
    }

    pub fn as_degrees(&self) -> f64 {
        self.0 * 180.0 / ::std::f64::consts::PI
    }

    pub fn as_cycles(&self) -> f64 {
        self.0 / (2.0 * ::std::f64::consts::PI)
    }
}
