//! Shared strict key/value CLI parsing for the separate PvP binaries.
use puyo_nn::pvp_model::PvpPuyoNetConfig;
use std::collections::BTreeMap;
pub struct Args(BTreeMap<String, String>);
impl Args {
    pub fn parse(allowed: &[&str]) -> Result<Self, String> {
        Self::parse_from(std::env::args().skip(1), allowed)
    }
    pub fn parse_from(
        args: impl IntoIterator<Item = String>,
        allowed: &[&str],
    ) -> Result<Self, String> {
        let mut args = args.into_iter();
        let mut values = BTreeMap::new();
        while let Some(key) = args.next() {
            if !allowed.contains(&key.as_str()) {
                return Err(format!(
                    "unknown option {key}; supported: {}",
                    allowed.join(" ")
                ));
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {key}"))?;
            if value.starts_with("--") {
                return Err(format!("missing value for {key}"));
            }
            if values.insert(key.clone(), value).is_some() {
                return Err(format!("duplicate option {key}"));
            }
        }
        Ok(Self(values))
    }
    pub fn get(&self, key: &str, default: &str) -> String {
        self.0.get(key).cloned().unwrap_or_else(|| default.into())
    }
    pub fn contains(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub fn temperature_schedule(
        &self,
    ) -> Result<crate::pvp_self_play::TemperatureSchedule, String> {
        use crate::pvp_self_play::TemperatureSchedule;
        if self.contains("--temperature-drop-piece") && self.contains("--temperature-drop-tick") {
            return Err(
                "--temperature-drop-piece and --temperature-drop-tick are mutually exclusive"
                    .into(),
            );
        }
        if self.contains("--temperature-drop-tick") {
            eprintln!(
                "WARNING: --temperature-drop-tick is deprecated; use --temperature-drop-piece"
            );
            Ok(TemperatureSchedule::LegacyTick(
                self.number("--temperature-drop-tick", "100")?,
            ))
        } else {
            Ok(TemperatureSchedule::PieceIndex(
                self.number("--temperature-drop-piece", "20")?,
            ))
        }
    }
    pub fn data_paths(&self) -> Result<Vec<String>, String> {
        if self.contains("--data") && self.contains("--data-paths") {
            return Err("--data and --data-paths are mutually exclusive".into());
        }
        let paths = if self.contains("--data-paths") {
            self.get("--data-paths", "")
                .split(',')
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>()
        } else {
            vec![self
                .get("--data", "data/pvp/pvp_alphazero_iter_001.bin")
                .trim()
                .to_string()]
        };
        let mut seen = std::collections::BTreeSet::new();
        for path in &paths {
            if path.is_empty() || !seen.insert(path.clone()) {
                return Err("empty or duplicate replay dataset path".into());
            }
        }
        Ok(paths)
    }
    pub fn number<T: std::str::FromStr>(&self, key: &str, default: &str) -> Result<T, String> {
        self.get(key, default)
            .parse()
            .map_err(|_| format!("invalid value for {key}"))
    }
    pub fn net_config(&self) -> Result<PvpPuyoNetConfig, String> {
        Ok(PvpPuyoNetConfig::new()
            .with_residual_channels(self.number("--residual-channels", "64")?)
            .with_num_residual_blocks(self.number("--residual-blocks", "6")?))
    }
}
