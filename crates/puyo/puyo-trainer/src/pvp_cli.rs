//! Shared strict key/value CLI parsing for the separate PvP binaries.
use puyo_nn::pvp_model::PvpPuyoNetConfig;
use std::collections::BTreeMap;
pub struct Args(BTreeMap<String, String>);
impl Args {
    pub fn parse(allowed: &[&str]) -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
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
