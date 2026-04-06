//! Anthropic-compatible API endpoints.
//!
//! This module provides compatibility with Anthropic's Messages API format, supporting
//! the messages endpoint. It handles routing, request/response conversion, and provides
//! the main entry points for Anthropic-compatible requests.

pub mod error;
pub mod messages;
pub mod types;

use messages::messages_handler;

use axum::routing::post;

use crate::endpoints::RouteHandlers;

/// Constructs (but does not register) all of our Anthropic-compatible endpoints.
pub fn build_anthropic_compatible_routes() -> RouteHandlers {
    RouteHandlers {
        routes: vec![("/anthropic/v1/messages", post(messages_handler))],
    }
}
