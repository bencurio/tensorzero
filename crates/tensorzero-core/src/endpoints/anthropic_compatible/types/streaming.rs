//! Streaming response types and logic for Anthropic-compatible API.
//!
//! This module provides types and functions for streaming messages responses
//! in Server-Sent Events (SSE) format, compatible with Anthropic's streaming API.
//! It uses Anthropic's event types: `message_start`, `content_block_start`,
//! `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`.

use axum::response::sse::Event;
use futures::Stream;
use serde::Serialize;
use serde_json::Value;
use tokio_stream::StreamExt;

use crate::endpoints::inference::{InferenceResponseChunk, InferenceStream};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{ContentBlockChunk, FinishReason};

use super::messages::{AnthropicStopReason, AnthropicUsage};

// ============================================================================
// Streaming Event Types
// ============================================================================

/// The `message_start` event payload.
#[derive(Clone, Debug, Serialize)]
pub struct MessageStartEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub message: MessageStartPayload,
}

#[derive(Clone, Debug, Serialize)]
pub struct MessageStartPayload {
    pub id: String,
    #[serde(rename = "type")]
    pub payload_type: String,
    pub role: String,
    pub content: Vec<Value>,
    pub model: String,
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// A `content_block_start` event payload.
#[derive(Clone, Debug, Serialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: usize,
    pub content_block: ContentBlockStart,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlockStart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

/// A `content_block_delta` event payload.
#[derive(Clone, Debug, Serialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: usize,
    pub delta: ContentBlockDelta,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
}

/// A `content_block_stop` event payload.
#[derive(Clone, Debug, Serialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: usize,
}

/// A `message_delta` event payload.
#[derive(Clone, Debug, Serialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub delta: MessageDelta,
    pub usage: MessageDeltaUsage,
}

#[derive(Clone, Debug, Serialize)]
pub struct MessageDelta {
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MessageDeltaUsage {
    pub output_tokens: u32,
}

/// A `message_stop` event payload.
#[derive(Clone, Debug, Serialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub event_type: String,
}

// ============================================================================
// State Tracking
// ============================================================================

/// State maintained across streaming chunks for Anthropic format.
#[derive(Debug, Default)]
struct AnthropicStreamingState {
    /// Current content block index.
    content_block_index: usize,
    /// Tracks active content block IDs and their index.
    active_blocks: std::collections::HashMap<String, usize>,
    /// Whether we have sent the first content block.
    has_started_content: bool,
    /// Total output tokens seen (from usage).
    output_tokens: u32,
}

// ============================================================================
// Stream Preparation
// ============================================================================

/// Converts a TensorZero inference stream into Anthropic-compatible SSE events.
pub fn prepare_serialized_anthropic_events(
    mut stream: InferenceStream,
    response_model_prefix: String,
    include_raw_response: bool,
) -> impl Stream<Item = Result<Event, Error>> {
    async_stream::stream! {
        let mut state = AnthropicStreamingState::default();
        let mut is_first_chunk = true;
        let mut last_finish_reason: Option<FinishReason> = None;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    let error_event = e.build_streaming_error_event(false, include_raw_response);
                    yield Event::default()
                        .event("error")
                        .json_data(&error_event)
                        .map_err(|ser_err| {
                            Error::new(ErrorDetails::Inference {
                                message: format!("Failed to convert error to Event: {ser_err}"),
                            })
                        });
                    continue;
                }
            };

            let events = convert_chunk_to_anthropic_events(
                chunk,
                &mut state,
                &response_model_prefix,
                is_first_chunk,
            );
            is_first_chunk = false;

            for (event_type, event_data, finish_reason) in events {
                if let Some(fr) = finish_reason {
                    last_finish_reason = Some(fr);
                }
                yield Event::default()
                    .event(&event_type)
                    .json_data(&event_data)
                    .map_err(|e| {
                        Error::new(ErrorDetails::Inference {
                            message: format!("Failed to serialize SSE event: {e}"),
                        })
                    });
            }
        }

        // Emit message_delta with stop_reason
        let stop_reason = last_finish_reason.map(AnthropicStopReason::from);
        let message_delta = MessageDeltaEvent {
            event_type: "message_delta".to_string(),
            delta: MessageDelta {
                stop_reason,
                stop_sequence: None,
            },
            usage: MessageDeltaUsage {
                output_tokens: state.output_tokens,
            },
        };
        yield Event::default()
            .event("message_delta")
            .json_data(&message_delta)
            .map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: format!("Failed to serialize message_delta event: {e}"),
                })
            });

        // Emit message_stop
        let message_stop = MessageStopEvent {
            event_type: "message_stop".to_string(),
        };
        yield Event::default()
            .event("message_stop")
            .json_data(&message_stop)
            .map_err(|e| {
                Error::new(ErrorDetails::Inference {
                    message: format!("Failed to serialize message_stop event: {e}"),
                })
            });
    }
}

/// Convert a single inference response chunk into Anthropic SSE events.
/// Returns a vec of (event_type, event_data_json, optional_finish_reason).
fn convert_chunk_to_anthropic_events(
    chunk: InferenceResponseChunk,
    state: &mut AnthropicStreamingState,
    response_model_prefix: &str,
    is_first_chunk: bool,
) -> Vec<(String, Value, Option<FinishReason>)> {
    let mut events = Vec::new();

    match chunk {
        InferenceResponseChunk::Chat(c) => {
            // Emit message_start on the first chunk
            if is_first_chunk {
                let usage = c
                    .usage
                    .as_ref()
                    .map(|u| AnthropicUsage::from(*u))
                    .unwrap_or_default();
                let message_start = MessageStartEvent {
                    event_type: "message_start".to_string(),
                    message: MessageStartPayload {
                        id: format!("msg_{}", c.inference_id),
                        payload_type: "message".to_string(),
                        role: "assistant".to_string(),
                        content: vec![],
                        model: format!("{response_model_prefix}{}", c.variant_name),
                        stop_reason: None,
                        stop_sequence: None,
                        usage,
                    },
                };
                events.push((
                    "message_start".to_string(),
                    serde_json::to_value(&message_start).unwrap_or_default(),
                    None,
                ));
            }

            // Track output tokens
            if let Some(usage) = &c.usage
                && let Some(output_tokens) = usage.output_tokens
            {
                state.output_tokens = output_tokens;
            }

            // Process content blocks
            for block in c.content {
                match block {
                    ContentBlockChunk::Text(text_chunk) => {
                        let block_id = text_chunk.id.clone();
                        if !state.active_blocks.contains_key(&block_id) {
                            let index = state.content_block_index;
                            state.active_blocks.insert(block_id.clone(), index);
                            state.content_block_index += 1;

                            // content_block_start
                            let start_event = ContentBlockStartEvent {
                                event_type: "content_block_start".to_string(),
                                index,
                                content_block: ContentBlockStart::Text {
                                    text: String::new(),
                                },
                            };
                            events.push((
                                "content_block_start".to_string(),
                                serde_json::to_value(&start_event).unwrap_or_default(),
                                None,
                            ));
                        }

                        let index = state.active_blocks[&block_id];
                        // content_block_delta
                        let delta_event = ContentBlockDeltaEvent {
                            event_type: "content_block_delta".to_string(),
                            index,
                            delta: ContentBlockDelta::TextDelta {
                                text: text_chunk.text,
                            },
                        };
                        events.push((
                            "content_block_delta".to_string(),
                            serde_json::to_value(&delta_event).unwrap_or_default(),
                            None,
                        ));
                    }
                    ContentBlockChunk::ToolCall(tool_chunk) => {
                        let block_id = tool_chunk.id.clone();
                        if !state.active_blocks.contains_key(&block_id) {
                            let index = state.content_block_index;
                            state.active_blocks.insert(block_id.clone(), index);
                            state.content_block_index += 1;

                            let start_event = ContentBlockStartEvent {
                                event_type: "content_block_start".to_string(),
                                index,
                                content_block: ContentBlockStart::ToolUse {
                                    id: tool_chunk.id.clone(),
                                    name: tool_chunk.raw_name.unwrap_or_default(),
                                    input: Value::Object(Default::default()),
                                },
                            };
                            events.push((
                                "content_block_start".to_string(),
                                serde_json::to_value(&start_event).unwrap_or_default(),
                                None,
                            ));
                        }

                        let index = state.active_blocks[&block_id];
                        let delta_event = ContentBlockDeltaEvent {
                            event_type: "content_block_delta".to_string(),
                            index,
                            delta: ContentBlockDelta::InputJsonDelta {
                                partial_json: tool_chunk.raw_arguments,
                            },
                        };
                        events.push((
                            "content_block_delta".to_string(),
                            serde_json::to_value(&delta_event).unwrap_or_default(),
                            None,
                        ));
                    }
                    ContentBlockChunk::Thought(thought_chunk) => {
                        let block_id = thought_chunk.id.clone();
                        if !state.active_blocks.contains_key(&block_id) {
                            let index = state.content_block_index;
                            state.active_blocks.insert(block_id.clone(), index);
                            state.content_block_index += 1;

                            let start_event = ContentBlockStartEvent {
                                event_type: "content_block_start".to_string(),
                                index,
                                content_block: ContentBlockStart::Thinking {
                                    thinking: String::new(),
                                },
                            };
                            events.push((
                                "content_block_start".to_string(),
                                serde_json::to_value(&start_event).unwrap_or_default(),
                                None,
                            ));
                        }

                        let index = state.active_blocks[&block_id];
                        let delta_event = ContentBlockDeltaEvent {
                            event_type: "content_block_delta".to_string(),
                            index,
                            delta: ContentBlockDelta::ThinkingDelta {
                                thinking: thought_chunk.text.unwrap_or_default(),
                            },
                        };
                        events.push((
                            "content_block_delta".to_string(),
                            serde_json::to_value(&delta_event).unwrap_or_default(),
                            None,
                        ));
                    }
                    ContentBlockChunk::Unknown(_) => {
                        // Skip unknown content blocks
                    }
                }
            }

            // If there's a finish_reason, emit content_block_stop for all active blocks
            if let Some(finish_reason) = c.finish_reason {
                for index in state.active_blocks.values() {
                    let stop_event = ContentBlockStopEvent {
                        event_type: "content_block_stop".to_string(),
                        index: *index,
                    };
                    events.push((
                        "content_block_stop".to_string(),
                        serde_json::to_value(&stop_event).unwrap_or_default(),
                        None,
                    ));
                }
                state.active_blocks.clear();

                // Return finish_reason for the final message_delta
                if let Some(e) = events.last_mut() {
                    e.2 = Some(finish_reason);
                }
            }
        }
        InferenceResponseChunk::Json(c) => {
            // For JSON responses, emit as text blocks
            if is_first_chunk {
                let usage = c
                    .usage
                    .as_ref()
                    .map(|u| AnthropicUsage::from(*u))
                    .unwrap_or_default();
                let message_start = MessageStartEvent {
                    event_type: "message_start".to_string(),
                    message: MessageStartPayload {
                        id: format!("msg_{}", c.inference_id),
                        payload_type: "message".to_string(),
                        role: "assistant".to_string(),
                        content: vec![],
                        model: format!("{response_model_prefix}{}", c.variant_name),
                        stop_reason: None,
                        stop_sequence: None,
                        usage,
                    },
                };
                events.push((
                    "message_start".to_string(),
                    serde_json::to_value(&message_start).unwrap_or_default(),
                    None,
                ));
            }

            if let Some(usage) = &c.usage
                && let Some(output_tokens) = usage.output_tokens
            {
                state.output_tokens = output_tokens;
            }

            if !state.has_started_content {
                let start_event = ContentBlockStartEvent {
                    event_type: "content_block_start".to_string(),
                    index: 0,
                    content_block: ContentBlockStart::Text {
                        text: String::new(),
                    },
                };
                events.push((
                    "content_block_start".to_string(),
                    serde_json::to_value(&start_event).unwrap_or_default(),
                    None,
                ));
                state.has_started_content = true;
            }

            let delta_event = ContentBlockDeltaEvent {
                event_type: "content_block_delta".to_string(),
                index: 0,
                delta: ContentBlockDelta::TextDelta { text: c.raw },
            };
            events.push((
                "content_block_delta".to_string(),
                serde_json::to_value(&delta_event).unwrap_or_default(),
                None,
            ));

            if let Some(finish_reason) = c.finish_reason {
                let stop_event = ContentBlockStopEvent {
                    event_type: "content_block_stop".to_string(),
                    index: 0,
                };
                events.push((
                    "content_block_stop".to_string(),
                    serde_json::to_value(&stop_event).unwrap_or_default(),
                    Some(finish_reason),
                ));
            }
        }
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::inference::{ChatInferenceResponseChunk, JsonInferenceResponseChunk};
    use crate::inference::types::TextChunk;
    use crate::inference::types::usage::Usage;
    use crate::tool::ToolCallChunk;
    use uuid::Uuid;

    fn make_chat_chunk(
        content: Vec<ContentBlockChunk>,
        usage: Option<Usage>,
        finish_reason: Option<FinishReason>,
    ) -> InferenceResponseChunk {
        InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            content,
            usage,
            raw_usage: None,
            finish_reason,
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
            aggregated_response: None,
        })
    }

    fn make_json_chunk(
        raw: &str,
        usage: Option<Usage>,
        finish_reason: Option<FinishReason>,
    ) -> InferenceResponseChunk {
        InferenceResponseChunk::Json(JsonInferenceResponseChunk {
            inference_id: Uuid::now_v7(),
            episode_id: Uuid::now_v7(),
            variant_name: "test_variant".to_string(),
            raw: raw.to_string(),
            usage,
            raw_usage: None,
            finish_reason,
            original_chunk: None,
            raw_chunk: None,
            raw_response: None,
            aggregated_response: None,
        })
    }

    // ========================================================================
    // First chunk emits message_start
    // ========================================================================

    #[test]
    fn test_first_chat_chunk_emits_message_start() {
        let chunk = make_chat_chunk(
            vec![ContentBlockChunk::Text(TextChunk {
                id: "t1".to_string(),
                text: "Hello".to_string(),
            })],
            None,
            None,
        );
        let mut state = AnthropicStreamingState::default();
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "prefix::", true);

        // Should have: message_start, content_block_start, content_block_delta
        assert!(
            events.len() >= 3,
            "Expected at least 3 events, got {}",
            events.len()
        );
        assert_eq!(events[0].0, "message_start");
        assert_eq!(events[1].0, "content_block_start");
        assert_eq!(events[2].0, "content_block_delta");
    }

    #[test]
    fn test_subsequent_chunk_no_message_start() {
        let chunk = make_chat_chunk(
            vec![ContentBlockChunk::Text(TextChunk {
                id: "t1".to_string(),
                text: "world".to_string(),
            })],
            None,
            None,
        );
        let mut state = AnthropicStreamingState::default();
        // Simulate that t1 was already started
        state.active_blocks.insert("t1".to_string(), 0);
        state.content_block_index = 1;

        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "prefix::", false);

        // Should only have content_block_delta (no message_start, no content_block_start)
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "content_block_delta");
    }

    // ========================================================================
    // Text content block tracking
    // ========================================================================

    #[test]
    fn test_new_text_block_emits_start_and_delta() {
        let chunk = make_chat_chunk(
            vec![ContentBlockChunk::Text(TextChunk {
                id: "text_1".to_string(),
                text: "Hi".to_string(),
            })],
            None,
            None,
        );
        let mut state = AnthropicStreamingState::default();
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, "content_block_start");
        assert_eq!(events[1].0, "content_block_delta");

        // Verify the delta contains the text
        let delta_data = &events[1].1;
        assert_eq!(delta_data["delta"]["text"], "Hi");
        assert_eq!(delta_data["delta"]["type"], "text_delta");
    }

    #[test]
    fn test_continuing_text_block_emits_only_delta() {
        let mut state = AnthropicStreamingState::default();
        state.active_blocks.insert("text_1".to_string(), 0);
        state.content_block_index = 1;

        let chunk = make_chat_chunk(
            vec![ContentBlockChunk::Text(TextChunk {
                id: "text_1".to_string(),
                text: " there".to_string(),
            })],
            None,
            None,
        );
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "content_block_delta");
        assert_eq!(events[0].1["index"], 0);
    }

    // ========================================================================
    // Tool call chunks
    // ========================================================================

    #[test]
    fn test_tool_call_chunk_emits_start_and_delta() {
        let chunk = make_chat_chunk(
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "tc_1".to_string(),
                raw_name: Some("get_weather".to_string()),
                raw_arguments: "{\"city\":".to_string(),
            })],
            None,
            None,
        );
        let mut state = AnthropicStreamingState::default();
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, "content_block_start");
        assert_eq!(events[0].1["content_block"]["type"], "tool_use");
        assert_eq!(events[0].1["content_block"]["name"], "get_weather");

        assert_eq!(events[1].0, "content_block_delta");
        assert_eq!(events[1].1["delta"]["type"], "input_json_delta");
        assert_eq!(events[1].1["delta"]["partial_json"], "{\"city\":");
    }

    #[test]
    fn test_tool_call_chunk_continuation() {
        let mut state = AnthropicStreamingState::default();
        state.active_blocks.insert("tc_1".to_string(), 0);
        state.content_block_index = 1;

        let chunk = make_chat_chunk(
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "tc_1".to_string(),
                raw_name: None,
                raw_arguments: "\"Tokyo\"}".to_string(),
            })],
            None,
            None,
        );
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "content_block_delta");
        assert_eq!(events[0].1["delta"]["partial_json"], "\"Tokyo\"}");
    }

    // ========================================================================
    // Thinking chunks
    // ========================================================================

    #[test]
    fn test_thinking_chunk_emits_start_and_delta() {
        let chunk = make_chat_chunk(
            vec![ContentBlockChunk::Thought(
                crate::inference::types::streams::ThoughtChunk {
                    id: "th_1".to_string(),
                    text: Some("Let me think...".to_string()),
                    signature: None,
                    summary_id: None,
                    summary_text: None,
                    provider_type: None,
                    extra_data: None,
                },
            )],
            None,
            None,
        );
        let mut state = AnthropicStreamingState::default();
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, "content_block_start");
        assert_eq!(events[0].1["content_block"]["type"], "thinking");

        assert_eq!(events[1].0, "content_block_delta");
        assert_eq!(events[1].1["delta"]["type"], "thinking_delta");
        assert_eq!(events[1].1["delta"]["thinking"], "Let me think...");
    }

    // ========================================================================
    // Finish reason emits content_block_stop
    // ========================================================================

    #[test]
    fn test_finish_reason_emits_content_block_stop() {
        let mut state = AnthropicStreamingState::default();
        state.active_blocks.insert("t1".to_string(), 0);
        state.active_blocks.insert("tc1".to_string(), 1);
        state.content_block_index = 2;

        let chunk = make_chat_chunk(vec![], None, Some(FinishReason::Stop));
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        // Should emit content_block_stop for each active block
        let stop_events: Vec<_> = events
            .iter()
            .filter(|(t, _, _)| t == "content_block_stop")
            .collect();
        assert_eq!(stop_events.len(), 2, "should stop both active blocks");

        // State should be cleared
        assert!(state.active_blocks.is_empty());
    }

    // ========================================================================
    // Usage tracking
    // ========================================================================

    #[test]
    fn test_output_tokens_tracked_in_state() {
        let chunk = make_chat_chunk(
            vec![],
            Some(Usage {
                input_tokens: Some(10),
                output_tokens: Some(42),
                provider_cache_read_input_tokens: None,
                provider_cache_write_input_tokens: None,
                cost: None,
            }),
            None,
        );
        let mut state = AnthropicStreamingState::default();
        convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        assert_eq!(state.output_tokens, 42);
    }

    // ========================================================================
    // JSON chunk handling
    // ========================================================================

    #[test]
    fn test_json_first_chunk_emits_message_start_and_content() {
        let chunk = make_json_chunk("{\"key\":", None, None);
        let mut state = AnthropicStreamingState::default();
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", true);

        // message_start + content_block_start + content_block_delta
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].0, "message_start");
        assert_eq!(events[1].0, "content_block_start");
        assert_eq!(events[2].0, "content_block_delta");
        assert_eq!(events[2].1["delta"]["text"], "{\"key\":");
    }

    #[test]
    fn test_json_subsequent_chunk_only_delta() {
        let mut state = AnthropicStreamingState::default();
        state.has_started_content = true;

        let chunk = make_json_chunk("\"value\"}", None, None);
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "content_block_delta");
    }

    #[test]
    fn test_json_chunk_with_finish_reason() {
        let mut state = AnthropicStreamingState::default();
        state.has_started_content = true;

        let chunk = make_json_chunk("}", None, Some(FinishReason::Stop));
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        // delta + content_block_stop
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, "content_block_delta");
        assert_eq!(events[1].0, "content_block_stop");
        assert_eq!(events[1].2, Some(FinishReason::Stop));
    }

    // ========================================================================
    // Multiple content blocks with correct indices
    // ========================================================================

    #[test]
    fn test_multiple_content_blocks_assigned_sequential_indices() {
        let mut state = AnthropicStreamingState::default();

        // First block: text
        let chunk1 = make_chat_chunk(
            vec![ContentBlockChunk::Text(TextChunk {
                id: "t1".to_string(),
                text: "Hi".to_string(),
            })],
            None,
            None,
        );
        let events1 = convert_chunk_to_anthropic_events(chunk1, &mut state, "p::", false);
        assert_eq!(events1[0].1["index"], 0); // content_block_start index=0

        // Second block: tool_call
        let chunk2 = make_chat_chunk(
            vec![ContentBlockChunk::ToolCall(ToolCallChunk {
                id: "tc1".to_string(),
                raw_name: Some("search".to_string()),
                raw_arguments: "{}".to_string(),
            })],
            None,
            None,
        );
        let events2 = convert_chunk_to_anthropic_events(chunk2, &mut state, "p::", false);
        assert_eq!(events2[0].1["index"], 1); // content_block_start index=1

        assert_eq!(state.content_block_index, 2);
    }

    // ========================================================================
    // Unknown chunks are skipped
    // ========================================================================

    #[test]
    fn test_unknown_chunk_skipped() {
        let chunk = make_chat_chunk(
            vec![ContentBlockChunk::Unknown(
                crate::inference::types::streams::UnknownChunk {
                    id: "u1".to_string(),
                    data: serde_json::json!({"type": "redacted_thinking"}),
                    model_name: None,
                    provider_name: None,
                },
            )],
            None,
            None,
        );
        let mut state = AnthropicStreamingState::default();
        let events = convert_chunk_to_anthropic_events(chunk, &mut state, "p::", false);

        assert!(events.is_empty(), "unknown chunks should produce no events");
    }
}
