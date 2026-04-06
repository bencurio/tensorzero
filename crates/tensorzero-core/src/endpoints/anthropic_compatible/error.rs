//! Anthropic-compatible error handling.
//!
//! This module provides error types and extractors that format responses according to Anthropic's API specification.
//! Anthropic errors use the format: `{"type": "error", "error": {"type": "...", "message": "..."}}`.

use std::fmt;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use serde_json::json;
use tracing::instrument;

use crate::error::Error;
use crate::utils::gateway::deserialize_json_request;

/// A wrapper around `Error` that implements `IntoResponse` with Anthropic-compatible error format.
///
/// Anthropic returns errors as `{"type": "error", "error": {"type": "...", "message": "..."}}`.
#[derive(Debug)]
pub struct AnthropicCompatibleError(pub Error);

impl From<Error> for AnthropicCompatibleError {
    fn from(error: Error) -> Self {
        AnthropicCompatibleError(error)
    }
}

impl fmt::Display for AnthropicCompatibleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl IntoResponse for AnthropicCompatibleError {
    fn into_response(self) -> Response {
        let status = self.0.status_code();
        let error_type = anthropic_error_type(status);
        let message = self.0.to_string();
        let body = json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            }
        });
        let mut response = (status, Json(body)).into_response();
        response.extensions_mut().insert(self.0);
        response
    }
}

/// Map HTTP status codes to Anthropic error types.
fn anthropic_error_type(status: axum::http::StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        500 => "api_error",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

/// A JSON extractor for Anthropic-compatible endpoints that returns errors in Anthropic format.
///
/// This is similar to `StructuredJson` but uses `AnthropicCompatibleError` as its rejection type
/// so that JSON parsing errors are returned in Anthropic's error format.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicStructuredJson<T>(pub T);

impl<S, T> FromRequest<S> for AnthropicStructuredJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
    T: Send + Sync + DeserializeOwned,
{
    type Rejection = AnthropicCompatibleError;

    #[instrument(
        skip_all,
        level = "trace",
        name = "AnthropicStructuredJson::from_request"
    )]
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        deserialize_json_request(req, state)
            .await
            .map(AnthropicStructuredJson)
            .map_err(AnthropicCompatibleError)
    }
}
