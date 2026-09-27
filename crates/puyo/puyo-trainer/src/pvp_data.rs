//! Versioned PvP data, deliberately incompatible with solo datasets.
use puyo_core::pvp::{MatchResult, PlayerId};
use puyo_core::pvp_encoding::*;
use serde::{Deserialize, Serialize};
use std::io::{Error, ErrorKind};
const MAGIC: &[u8; 8] = b"PUYOPVP1";
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PvpDataMetadata {
    pub format_version: u32,
    pub model_kind: String,
    pub board_channels: usize,
    pub rows: usize,
    pub cols: usize,
    pub context_size: usize,
    pub num_actions: usize,
}
impl Default for PvpDataMetadata {
    fn default() -> Self {
        Self {
            format_version: 1,
            model_kind: "pvp".into(),
            board_channels: BOARD_CHANNELS,
            rows: ROWS,
            cols: COLS,
            context_size: CONTEXT_SIZE,
            num_actions: NUM_ACTIONS,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PvpAlphaZeroSample {
    pub board_data: Vec<f32>,
    pub context_data: Vec<f32>,
    pub improved_policy: Vec<f32>,
    pub value_target: f32,
    /// Diagnostics/validation only. Never passed into the network.
    pub perspective: u8,
    pub legal_mask: Vec<bool>,
}
impl PvpAlphaZeroSample {
    pub fn validate(&self) -> Result<(), String> {
        if self.board_data.len() != BOARD_SIZE
            || self.context_data.len() != CONTEXT_SIZE
            || self.improved_policy.len() != NUM_ACTIONS
            || self.legal_mask.len() != NUM_ACTIONS
            || self.perspective > 1
        {
            return Err("invalid PvP sample dimensions/perspective".into());
        }
        if self
            .board_data
            .iter()
            .chain(&self.context_data)
            .any(|v| !v.is_finite())
            || ![-1., 0., 1.].contains(&self.value_target)
        {
            return Err("invalid PvP sample values".into());
        }
        let sum: f32 = self.improved_policy.iter().sum();
        if (sum - 1.).abs() > 1e-4
            || self
                .improved_policy
                .iter()
                .enumerate()
                .any(|(i, p)| !p.is_finite() || *p < 0. || (!self.legal_mask[i] && *p != 0.))
        {
            return Err("invalid PvP policy distribution".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PvpAlphaZeroDataset {
    pub metadata: PvpDataMetadata,
    pub samples: Vec<PvpAlphaZeroSample>,
}
impl PvpAlphaZeroDataset {
    pub fn validate(&self) -> Result<(), String> {
        if self.metadata != PvpDataMetadata::default() {
            return Err("incompatible PvP dataset metadata".into());
        }
        for s in &self.samples {
            s.validate()?;
        }
        Ok(())
    }
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.validate()
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        let mut bytes = MAGIC.to_vec();
        bytes.extend(bincode::serialize(self).map_err(|e| Error::new(ErrorKind::InvalidData, e))?);
        std::fs::write(path, bytes)
    }
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if !bytes.starts_with(MAGIC) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "incompatible model kind/dataset: expected pvp",
            ));
        }
        let data: Self =
            bincode::deserialize(&bytes[8..]).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        data.validate()
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        Ok(data)
    }
}
#[derive(Debug, Clone)]
pub struct MoveRecord {
    pub board_data: Vec<f32>,
    pub context_data: Vec<f32>,
    pub improved_policy: Vec<f32>,
    pub perspective: PlayerId,
    pub legal_mask: Vec<bool>,
}
/// Ongoing means truncation: no synthetic draw labels and no bootstrap targets.
pub fn finish_records(records: Vec<MoveRecord>, result: MatchResult) -> Vec<PvpAlphaZeroSample> {
    records
        .into_iter()
        .filter_map(|r| {
            result
                .outcome_for(r.perspective)
                .map(|z| PvpAlphaZeroSample {
                    board_data: r.board_data,
                    context_data: r.context_data,
                    improved_policy: r.improved_policy,
                    perspective: r.perspective.index() as u8,
                    legal_mask: r.legal_mask,
                    value_target: z as f32,
                })
        })
        .collect()
}
pub fn augment_colors(sample: &mut PvpAlphaZeroSample, perm: &[usize]) -> Result<(), String> {
    let mut sorted = perm.to_vec();
    sorted.sort_unstable();
    if sorted != [0, 1, 2, 3] {
        return Err("invalid four-color permutation".into());
    }
    sample.validate()?;
    for side in 0..2 {
        let start = side * 5 * ROWS * COLS;
        crate::data::apply_color_perm_board(
            &mut sample.board_data[start..start + 4 * ROWS * COLS],
            perm,
        );
    }
    crate::data::apply_color_perm_context(&mut sample.context_data[..48], perm);
    Ok(())
}
