//! Messages endpoint handler for Anthropic-compatible API.
//!
//! This module implements the HTTP handler for the `/anthropic/v1/messages` endpoint,
//! which provides Anthropic Messages API compatibility. It handles request validation,
//! parameter parsing, inference execution, and response formatting for both streaming
//! and non-streaming requests.

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use axum::{Extension, debug_handler};

use crate::endpoints::inference::{InferenceOutput, Params, inference};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, SwappableAppStateData};
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::error::{AnthropicCompatibleError, AnthropicStructuredJson};
use super::types::messages::{AnthropicMessagesParams, AnthropicMessagesResponse};
use super::types::streaming::prepare_serialized_anthropic_events;

/// A handler for the Anthropic-compatible messages endpoint.
#[debug_handler(state = SwappableAppStateData)]
pub async fn messages_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        cache_manager,
        deferred_tasks,
        rate_limiting_manager,
        primary_datastore,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    AnthropicStructuredJson(anthropic_params): AnthropicStructuredJson<AnthropicMessagesParams>,
) -> Result<Response<Body>, AnthropicCompatibleError> {
    let include_original_response = anthropic_params.tensorzero_include_original_response;
    let include_raw_response = anthropic_params.tensorzero_include_raw_response;

    if include_original_response {
        tracing::warn!(
            "The `tensorzero::include_original_response` parameter is deprecated. Use `tensorzero::include_raw_response` instead."
        );
    }

    let params = Params::try_from_anthropic(anthropic_params)?;

    // The prefix for the response's `model` field depends on the inference target
    let response_model_prefix = match (&params.function_name, &params.model_name) {
        (Some(function_name), None) => Ok::<String, Error>(format!(
            "tensorzero::function_name::{function_name}::variant_name::",
        )),
        (None, Some(_model_name)) => Ok("tensorzero::model_name::".to_string()),
        (Some(_), Some(_)) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "Only one of `function_name` or `model_name` can be provided".to_string(),
        }
        .into()),
        (None, None) => Err(ErrorDetails::InvalidInferenceTarget {
            message: "Either `function_name` or `model_name` must be provided".to_string(),
        }
        .into()),
    }?;

    let inference_result = Box::pin(inference(
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        cache_manager,
        deferred_tasks,
        rate_limiting_manager,
        primary_datastore,
        params,
        api_key_ext,
    ))
    .await;

    let output = match inference_result {
        Ok(data) => data.output,
        Err(e) => {
            // Return Anthropic-format error
            return Err(AnthropicCompatibleError(e));
        }
    };

    match output {
        InferenceOutput::NonStreaming(response) => {
            let anthropic_response = AnthropicMessagesResponse::from((
                response,
                response_model_prefix,
                include_original_response,
                include_raw_response,
            ));
            Ok(Json(anthropic_response).into_response())
        }
        InferenceOutput::Streaming(stream) => {
            let anthropic_stream = prepare_serialized_anthropic_events(
                stream,
                response_model_prefix,
                include_raw_response,
            );
            Ok(Sse::new(anthropic_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}
