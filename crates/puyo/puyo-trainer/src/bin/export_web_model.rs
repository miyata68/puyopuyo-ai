use std::error::Error;
use std::path::Path;

use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use burn::record::{
    BinBytesRecorder,
    BinFileRecorder,
    FullPrecisionSettings,
    Recorder,
};

use puyo_core::config::GameConfig;
use puyo_nn::model::{PuyoNet, PuyoNetConfig};

use serde::Deserialize;

type InferBackend = NdArray;

#[derive(Debug, Deserialize)]
struct ModelMetadata {
    game_config: GameConfig,
    residual_channels: usize,
    num_residual_blocks: usize,
    policy_conv_channels: usize,
    value_conv_channels: usize,
    value_hidden: usize,
    film_hidden: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    // BinFileRecorder::load_file() には拡張子なしのパスを渡す。
    let model_base_path = "artifacts/puyo_model";
    let metadata_path = "artifacts/puyo_model.config.json";

    let output_path = "web/public/models/puyo_model.bin";

    println!("Loading metadata from {metadata_path}...");

    let metadata_json = std::fs::read_to_string(metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;

    let gc = metadata.game_config.clone();

    println!(
        "Game config: cols={}, rows={}, num_colors={}",
        gc.cols, gc.rows, gc.num_colors
    );

    println!(
        "Net config: residual_channels={}, num_blocks={}, \
         policy_conv_channels={}, value_conv_channels={}, \
         value_hidden={}, film_hidden={}",
        metadata.residual_channels,
        metadata.num_residual_blocks,
        metadata.policy_conv_channels,
        metadata.value_conv_channels,
        metadata.value_hidden,
        metadata.film_hidden,
    );

    let net_config = PuyoNetConfig::new()
        .with_residual_channels(metadata.residual_channels)
        .with_num_residual_blocks(metadata.num_residual_blocks)
        .with_policy_conv_channels(metadata.policy_conv_channels)
        .with_value_conv_channels(metadata.value_conv_channels)
        .with_value_hidden(metadata.value_hidden)
        .with_film_hidden(metadata.film_hidden)
        .with_game_config(gc);

    let device = <InferBackend as Backend>::Device::default();

    println!("Loading trained model using NdArray backend...");

    let model: PuyoNet<InferBackend> = net_config
        .init::<InferBackend>(&device)
        .load_file(
            model_base_path,
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )?;

    println!("Model loaded successfully.");

    println!("Serializing with BinBytesRecorder...");

    let recorder =
        BinBytesRecorder::<FullPrecisionSettings>::default();

    let record = model.into_record();

    let bytes =
        <BinBytesRecorder<FullPrecisionSettings> as Recorder<InferBackend>>::record(
            &recorder,
            record,
            (),
        )?;

    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(output_path, &bytes)?;

    println!(
        "Web model exported successfully: {} ({} bytes)",
        output_path,
        bytes.len()
    );

    Ok(())
}