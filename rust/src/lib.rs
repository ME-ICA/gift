//! complex-gift: complex-valued fMRI ICA (Rust port of GIFT's complex ICA pipeline).
//!
//! GPL v3 - derives from the Adali-lab (MLSP/UMBC) complex ICA algorithms.

pub mod complex_io;
pub mod estimators;
pub mod group;
pub mod nf_table;
pub mod phase_correct;
pub mod phase_mask;
pub mod pipeline;
pub mod whiten;
