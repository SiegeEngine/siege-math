# siege-math

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Documentation is available at https://docs.rs/siege-math

**siege-math** is a component within the Siege Engine MMO game engine.

The Siege Engine is an MMO game engine on the Vulkan API written in the Rust language.

siege-math provides primitives for *Angle*, *Vector* (including *Direction* and *Point*
variants), *Matrix*, *Quaternion*, and *Position* types defined over any floating
point type.

siege-math was developed in response to several other math crates (cgmath and nalgebra)
going in directions slightly adverse to our usage. As a math library is actually a
rather small thing, we were not too fussed about creating a new one.


## Work TBD

SIMD is not yet available in stable rust. RFC 2366 (https://github.com/rust-lang/rfcs/pull/2366)
is the latest in the chain of work to make that happen. Once it (or something like it) lands
in stable rust, we can refactor to make use of SIMD for a significant performance increase.
In the meantime, there is a crate "simd" and another "fake-simd" that might be useful
for getting started.