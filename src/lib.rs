//! Stakpak Agent — an autonomous DevOps agent for the terminal.
//!
//! The crate is split so the binary is a thin shell over reusable pieces:
//!
//! - [`agent`] drives the model/tool loop and owns the system prompt.
//! - [`tools`] is the tool surface the model is given, plus path sandboxing.
//! - [`redact`] swaps credentials for placeholders on the way to the model and
//!   back again on the way into a tool.
//! - [`backup`] makes every file write reversible.
//! - [`session`] persists transcripts and checkpoints.

pub mod agent;
pub mod backup;
pub mod config;
pub mod redact;
pub mod session;
pub mod tools;
pub mod ui;
