//! Batched inference server for GPU-accelerated self-play.
//!
//! Architecture:
//! - One GPU thread owns the model and runs batched forward passes.
//! - N game threads send inference requests and block until results arrive.
//! - Requests are batched together for efficient GPU utilization.

use std::sync::mpsc;
use std::sync::{Condvar, Mutex};

use burn::prelude::*;

use crate::mcts::InferenceProvider;
use crate::model::GameModel;

/// A one-shot channel for sending a single response back to a requester.
struct OneshotSender<T> {
    inner: std::sync::Arc<OneshotInner<T>>,
}

struct OneshotReceiver<T> {
    inner: std::sync::Arc<OneshotInner<T>>,
}

struct OneshotInner<T> {
    data: Mutex<Option<T>>,
    ready: Condvar,
}

fn oneshot_channel<T>() -> (OneshotSender<T>, OneshotReceiver<T>) {
    let inner = std::sync::Arc::new(OneshotInner {
        data: Mutex::new(None),
        ready: Condvar::new(),
    });
    (
        OneshotSender {
            inner: inner.clone(),
        },
        OneshotReceiver { inner },
    )
}

impl<T> OneshotSender<T> {
    fn send(self, value: T) {
        let mut guard = self.inner.data.lock().expect("oneshot mutex poisoned");
        *guard = Some(value);
        self.inner.ready.notify_one();
    }
}

impl<T> OneshotReceiver<T> {
    fn recv(self) -> T {
        let mut guard = self.inner.data.lock().expect("oneshot mutex poisoned");
        loop {
            if let Some(val) = guard.take() {
                return val;
            }
            guard = self
                .inner
                .ready
                .wait(guard)
                .expect("oneshot condvar wait failed");
        }
    }
}

struct InferenceResponse {
    logits: Vec<f32>,
    value: f32,
}

struct InferenceRequest {
    board_data: Vec<f32>,
    context_data: Vec<f32>,
    response_tx: OneshotSender<InferenceResponse>,
}

/// Client handle for sending inference requests to the GPU server.
/// Cloneable — each game thread gets its own client.
pub struct InferenceClient {
    request_tx: mpsc::Sender<InferenceRequest>,
}

impl Clone for InferenceClient {
    fn clone(&self) -> Self {
        Self {
            request_tx: self.request_tx.clone(),
        }
    }
}

impl InferenceProvider for InferenceClient {
    fn infer(&self, board_data: &[f32], context_data: &[f32]) -> (Vec<f32>, f32) {
        let (response_tx, response_rx) = oneshot_channel();
        let request = InferenceRequest {
            board_data: board_data.to_vec(),
            context_data: context_data.to_vec(),
            response_tx,
        };
        self.request_tx
            .send(request)
            .expect("Inference server has shut down");
        let response = response_rx.recv();
        (response.logits, response.value)
    }

    fn infer_batch(&self, batch: &[(Vec<f32>, Vec<f32>)]) -> Vec<(Vec<f32>, f32)> {
        if batch.is_empty() {
            return vec![];
        }
        // Send all requests first, then collect all responses.
        // This allows the GPU server to batch them into a single forward pass.
        let receivers: Vec<_> = batch
            .iter()
            .map(|(board_data, context_data)| {
                let (response_tx, response_rx) = oneshot_channel();
                let request = InferenceRequest {
                    board_data: board_data.clone(),
                    context_data: context_data.clone(),
                    response_tx,
                };
                self.request_tx
                    .send(request)
                    .expect("Inference server has shut down");
                response_rx
            })
            .collect();

        receivers
            .into_iter()
            .map(|rx| {
                let response = rx.recv();
                (response.logits, response.value)
            })
            .collect()
    }
}

/// Start the batched inference server on a dedicated thread.
///
/// Returns an `InferenceClient` that can be cloned and distributed to game threads.
/// The server thread runs until all clients (senders) are dropped.
pub fn start_inference_server<B: Backend + 'static, M: GameModel<B>>(
    model: M,
    device: B::Device,
    max_batch_size: usize,
) -> InferenceClient
where
    B::Device: Send,
{
    let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>();

    std::thread::spawn(move || {
        inference_server_loop(&model, &device, &request_rx, max_batch_size);
    });

    InferenceClient { request_tx }
}

fn inference_server_loop<B: Backend, M: GameModel<B>>(
    model: &M,
    device: &B::Device,
    request_rx: &mpsc::Receiver<InferenceRequest>,
    max_batch_size: usize,
) {
    let (channels, rows, cols) = model.board_shape();
    let context_size = model.context_size();
    let num_actions = model.num_actions();

    loop {
        // Block until at least one request arrives
        let first = match request_rx.recv() {
            Ok(req) => req,
            Err(_) => return, // All senders dropped, shut down
        };

        // Greedily collect more requests (up to max_batch_size)
        let mut batch = vec![first];
        while batch.len() < max_batch_size {
            match request_rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        let batch_size = batch.len();

        // Build batched tensors
        let board_flat: Vec<f32> = batch
            .iter()
            .flat_map(|r| r.board_data.iter().copied())
            .collect();
        let context_flat: Vec<f32> = batch
            .iter()
            .flat_map(|r| r.context_data.iter().copied())
            .collect();

        let board_tensor = Tensor::<B, 1>::from_floats(board_flat.as_slice(), device)
            .reshape([batch_size, channels, rows, cols]);
        let context_tensor = Tensor::<B, 1>::from_floats(context_flat.as_slice(), device)
            .reshape([batch_size, context_size]);

        // Batched forward pass
        let (logits_batch, value_batch) = model.forward(board_tensor, context_tensor);

        // Extract results
        let logits_data = logits_batch
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to extract logits");
        let value_data = value_batch
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to extract values");

        // Distribute results back to requesters
        for (i, request) in batch.into_iter().enumerate() {
            let start = i * num_actions;
            let end = start + num_actions;
            let logits = logits_data[start..end].to_vec();
            let v_raw = if i < value_data.len() {
                value_data[i]
            } else {
                0.0
            };
            let value = model.postprocess_value(v_raw);

            request
                .response_tx
                .send(InferenceResponse { logits, value });
        }
    }
}
