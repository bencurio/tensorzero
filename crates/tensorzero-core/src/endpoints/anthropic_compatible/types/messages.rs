//! Request and response types for the Anthropic Messages API compatible endpoint.
//!
//! This module contains all request/response types for the `/v1/messages` endpoint,
//! including message structures, parameter types, and conversion logic between
//! Anthropic-compatible formats and TensorZero's internal representations.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tensorzero_derive::TensorZeroDeserialize;
use uuid::Uuid;

use crate::cache::CacheParamsOptions;
use crate::config::{Namespace, UninitializedVariantInfo};
use crate::endpoints::inference::{
    ChatCompletionInferenceParams, InferenceCredentials, InferenceParams, InferenceResponse, Params,
};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::extra_body::UnfilteredInferenceExtraBody;
use crate::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use crate::inference::types::usage::{RawResponseEntry, RawUsageEntry};
use crate::inference::types::{
    ContentBlockChatOutput, FinishReason, Input, InputMessage, InputMessageContent, Role, System,
    Text, Thought, ToolCallWrapper,
};
use crate::tool::{DynamicToolParams, FunctionTool, ProviderTool, Tool};
use rust_decimal::Decimal;
use tensorzero_types::file::{Base64File, UrlFile};
use tensorzero_types::tool::{InferenceResponseToolCall, ToolResult};

// ============================================================================
// Request Message Types
// ============================================================================

/// A content block in Anthropic Messages API format.
#[derive(Clone, Debug, TensorZeroDeserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Option<Value>,
        #[serde(default)]
        is_error: bool,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: Option<String>,
    pub data: Option<String>,
    pub url: Option<String>,
}

/// A message in the Anthropic Messages API format.
#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    pub content: AnthropicMessageContent,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
}

/// Content can be either a plain string or an array of content blocks.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

// ============================================================================
// Request Parameter Types
// ============================================================================

/// Tool definition in Anthropic Messages API format.
#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Tool choice in Anthropic Messages API format.
#[derive(Clone, Debug, TensorZeroDeserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
    #[serde(rename = "none")]
    None,
}

/// Anthropic Messages API request parameters.
#[derive(Clone, Debug, Deserialize)]
pub struct AnthropicMessagesParams {
    /// The model identifier. In TensorZero, this is used to specify the function or model.
    /// Must start with `tensorzero::function_name::` or `tensorzero::model_name::`.
    pub model: String,
    /// The messages to generate a response for.
    pub messages: Vec<AnthropicMessage>,
    /// The maximum number of tokens to generate.
    pub max_tokens: u32,
    /// System prompt. Can be a string or array of content blocks.
    #[serde(default)]
    pub system: Option<Value>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Sampling temperature (0.0 to 1.0).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p sampling.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Stop sequences.
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Tool definitions.
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    /// Tool choice.
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,

    // TensorZero-specific parameters
    #[serde(rename = "tensorzero::variant_name")]
    pub tensorzero_variant_name: Option<String>,
    #[serde(rename = "tensorzero::dryrun")]
    pub tensorzero_dryrun: Option<bool>,
    #[serde(rename = "tensorzero::episode_id")]
    pub tensorzero_episode_id: Option<Uuid>,
    #[serde(rename = "tensorzero::namespace")]
    pub tensorzero_namespace: Option<Namespace>,
    #[serde(rename = "tensorzero::cache_options")]
    pub tensorzero_cache_options: Option<CacheParamsOptions>,
    #[serde(default, rename = "tensorzero::extra_body")]
    pub tensorzero_extra_body: UnfilteredInferenceExtraBody,
    #[serde(default, rename = "tensorzero::extra_headers")]
    pub tensorzero_extra_headers: UnfilteredInferenceExtraHeaders,
    #[serde(default, rename = "tensorzero::tags")]
    pub tensorzero_tags: HashMap<String, String>,
    #[serde(default, rename = "tensorzero::credentials")]
    pub tensorzero_credentials: InferenceCredentials,
    #[serde(rename = "tensorzero::internal_dynamic_variant_config")]
    pub tensorzero_internal_dynamic_variant_config: Option<UninitializedVariantInfo>,
    #[serde(default, rename = "tensorzero::provider_tools")]
    pub tensorzero_provider_tools: Vec<ProviderTool>,
    #[serde(default, rename = "tensorzero::params")]
    pub tensorzero_params: Option<InferenceParams>,
    #[serde(default, rename = "tensorzero::include_raw_usage")]
    pub tensorzero_include_raw_usage: bool,
    /// DEPRECATED (#5697 / 2026.4+): Use `tensorzero::include_raw_response` instead.
    #[serde(default, rename = "tensorzero::include_original_response")]
    pub tensorzero_include_original_response: bool,
    #[serde(default, rename = "tensorzero::include_raw_response")]
    pub tensorzero_include_raw_response: bool,
}

// ============================================================================
// Response Types
// ============================================================================

/// A content block in the Anthropic Messages API response.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

/// Stop reason in Anthropic Messages API format.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

impl From<FinishReason> for AnthropicStopReason {
    fn from(finish_reason: FinishReason) -> Self {
        match finish_reason {
            FinishReason::Stop => AnthropicStopReason::EndTurn,
            FinishReason::StopSequence => AnthropicStopReason::StopSequence,
            FinishReason::Length => AnthropicStopReason::MaxTokens,
            FinishReason::ContentFilter => AnthropicStopReason::EndTurn,
            FinishReason::ToolCall => AnthropicStopReason::ToolUse,
            FinishReason::Unknown => AnthropicStopReason::EndTurn,
        }
    }
}

/// Usage in Anthropic Messages API format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(
        with = "rust_decimal::serde::float_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub tensorzero_cost: Option<Decimal>,
}

impl From<crate::inference::types::Usage> for AnthropicUsage {
    fn from(usage: crate::inference::types::Usage) -> Self {
        AnthropicUsage {
            input_tokens: usage.input_tokens.unwrap_or(0),
            output_tokens: usage.output_tokens.unwrap_or(0),
            cache_creation_input_tokens: usage.provider_cache_write_input_tokens,
            cache_read_input_tokens: usage.provider_cache_read_input_tokens,
            tensorzero_cost: usage.cost,
        }
    }
}

/// The full Anthropic Messages API response.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<AnthropicResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
    /// TensorZero-specific: the episode ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_episode_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_usage: Option<Vec<RawUsageEntry>>,
    /// DEPRECATED (#5697 / 2026.4+): Use `tensorzero_raw_response` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_original_response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensorzero_raw_response: Option<Vec<RawResponseEntry>>,
}

// ============================================================================
// Conversion: AnthropicMessagesParams -> Params
// ============================================================================

const TENSORZERO_FUNCTION_NAME_PREFIX: &str = "tensorzero::function_name::";
const TENSORZERO_MODEL_NAME_PREFIX: &str = "tensorzero::model_name::";

impl Params {
    pub fn try_from_anthropic(anthropic_params: AnthropicMessagesParams) -> Result<Self, Error> {
        let (function_name, model_name) = if let Some(function_name) = anthropic_params
            .model
            .strip_prefix(TENSORZERO_FUNCTION_NAME_PREFIX)
        {
            (Some(function_name.to_string()), None)
        } else if let Some(model_name) = anthropic_params
            .model
            .strip_prefix(TENSORZERO_MODEL_NAME_PREFIX)
        {
            (None, Some(model_name.to_string()))
        } else {
            return Err(Error::new(ErrorDetails::InvalidAnthropicCompatibleRequest {
                message: "`model` field must start with `tensorzero::function_name::` or `tensorzero::model_name::`. For example, `tensorzero::function_name::my_function` for a function defined in your config, or `tensorzero::model_name::openai::gpt-4o-mini`.".to_string(),
            }));
        };

        if let Some(function_name) = &function_name
            && function_name.is_empty()
        {
            return Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
                message: "function_name (passed in model field after \"tensorzero::function_name::\") cannot be empty".to_string(),
            }.into());
        }

        if let Some(model_name) = &model_name
            && model_name.is_empty()
        {
            return Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
                message: "model_name (passed in model field after \"tensorzero::model_name::\") cannot be empty".to_string(),
            }.into());
        }

        let input =
            anthropic_messages_to_input(anthropic_params.messages, anthropic_params.system)?;

        let mut inference_params = anthropic_params.tensorzero_params.unwrap_or_default();

        inference_params.chat_completion = ChatCompletionInferenceParams {
            temperature: inference_params
                .chat_completion
                .temperature
                .or(anthropic_params.temperature),
            top_p: inference_params
                .chat_completion
                .top_p
                .or(anthropic_params.top_p),
            max_tokens: inference_params
                .chat_completion
                .max_tokens
                .or(Some(anthropic_params.max_tokens)),
            stop_sequences: inference_params
                .chat_completion
                .stop_sequences
                .or(anthropic_params.stop_sequences),
            ..inference_params.chat_completion
        };

        let (tool_choice, additional_tools) =
            convert_anthropic_tools(anthropic_params.tools, anthropic_params.tool_choice);

        let dynamic_tool_params = DynamicToolParams {
            allowed_tools: None,
            additional_tools,
            tool_choice,
            parallel_tool_calls: None,
            provider_tools: anthropic_params.tensorzero_provider_tools,
        };

        Ok(Params {
            function_name,
            model_name,
            episode_id: anthropic_params.tensorzero_episode_id,
            namespace: anthropic_params.tensorzero_namespace,
            input,
            stream: anthropic_params.stream,
            params: inference_params,
            variant_name: anthropic_params.tensorzero_variant_name,
            dryrun: anthropic_params.tensorzero_dryrun,
            dynamic_tool_params,
            output_schema: None,
            credentials: anthropic_params.tensorzero_credentials,
            cache_options: anthropic_params
                .tensorzero_cache_options
                .unwrap_or_default(),
            internal: false,
            tags: anthropic_params.tensorzero_tags,
            include_original_response: anthropic_params.tensorzero_include_original_response,
            include_raw_response: anthropic_params.tensorzero_include_raw_response,
            include_raw_usage: anthropic_params.tensorzero_include_raw_usage,
            include_aggregated_response: false,
            extra_body: anthropic_params.tensorzero_extra_body,
            extra_headers: anthropic_params.tensorzero_extra_headers,
            internal_dynamic_variant_config: anthropic_params
                .tensorzero_internal_dynamic_variant_config,
        })
    }
}

// ============================================================================
// Conversion: InferenceResponse -> AnthropicMessagesResponse
// ============================================================================

impl From<(InferenceResponse, String, bool, bool)> for AnthropicMessagesResponse {
    fn from(
        (
            inference_response,
            response_model_prefix,
            include_original_response,
            include_raw_response,
        ): (InferenceResponse, String, bool, bool),
    ) -> Self {
        match inference_response {
            InferenceResponse::Chat(response) => {
                let content = chat_content_to_anthropic_blocks(response.content);
                let tensorzero_original_response = if include_original_response {
                    response.original_response
                } else {
                    None
                };
                let tensorzero_raw_response = if include_raw_response {
                    response.raw_response
                } else {
                    None
                };

                AnthropicMessagesResponse {
                    id: format!("msg_{}", response.inference_id),
                    response_type: "message".to_string(),
                    role: "assistant".to_string(),
                    content,
                    model: format!("{response_model_prefix}{}", response.variant_name),
                    stop_reason: response.finish_reason.map(AnthropicStopReason::from),
                    stop_sequence: None,
                    usage: response.usage.into(),
                    tensorzero_episode_id: Some(response.episode_id.to_string()),
                    tensorzero_raw_usage: response.raw_usage,
                    tensorzero_original_response,
                    tensorzero_raw_response,
                }
            }
            InferenceResponse::Json(response) => {
                let content = vec![AnthropicResponseContentBlock::Text {
                    text: response.output.raw.unwrap_or_default(),
                }];
                let tensorzero_original_response = if include_original_response {
                    response.original_response
                } else {
                    None
                };
                let tensorzero_raw_response = if include_raw_response {
                    response.raw_response
                } else {
                    None
                };

                AnthropicMessagesResponse {
                    id: format!("msg_{}", response.inference_id),
                    response_type: "message".to_string(),
                    role: "assistant".to_string(),
                    content,
                    model: format!("{response_model_prefix}{}", response.variant_name),
                    stop_reason: response.finish_reason.map(AnthropicStopReason::from),
                    stop_sequence: None,
                    usage: response.usage.into(),
                    tensorzero_episode_id: Some(response.episode_id.to_string()),
                    tensorzero_raw_usage: response.raw_usage,
                    tensorzero_original_response,
                    tensorzero_raw_response,
                }
            }
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert Anthropic messages + system prompt to TensorZero `Input`.
fn anthropic_messages_to_input(
    messages: Vec<AnthropicMessage>,
    system: Option<Value>,
) -> Result<Input, Error> {
    let system_message = match system {
        Some(Value::String(s)) => Some(System::Text(s)),
        Some(Value::Array(blocks)) => {
            let mut text_parts = Vec::new();
            for block in blocks {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("text");
                match block_type {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(text.to_string());
                        }
                    }
                    other => {
                        return Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
                            message: format!(
                                "Unsupported system content block type: `{other}`. Only `text` is supported."
                            ),
                        }
                        .into());
                    }
                }
            }
            if text_parts.is_empty() {
                None
            } else {
                Some(System::Text(text_parts.join("\n")))
            }
        }
        Some(_) => {
            return Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
                message: "`system` must be a string or an array of content blocks".to_string(),
            }
            .into());
        }
        None => None,
    };

    let mut input_messages = Vec::new();

    for msg in messages {
        let role = match msg.role {
            AnthropicRole::User => Role::User,
            AnthropicRole::Assistant => Role::Assistant,
        };

        let content_blocks = match msg.content {
            AnthropicMessageContent::Text(text) => {
                vec![InputMessageContent::Text(Text { text })]
            }
            AnthropicMessageContent::Blocks(blocks) => {
                let mut contents = Vec::new();
                for block in blocks {
                    match block {
                        AnthropicContentBlock::Text { text } => {
                            contents.push(InputMessageContent::Text(Text { text }));
                        }
                        AnthropicContentBlock::Image { source } => {
                            let file = match source.source_type.as_str() {
                                "base64" => {
                                    let mime_type = source
                                        .media_type
                                        .as_deref()
                                        .and_then(|mt| mt.parse::<mime::MediaType>().ok());
                                    let data = source.data.unwrap_or_default();
                                    let base64_file =
                                        Base64File::new(None, mime_type, data, None, None)
                                            .map_err(|e| {
                                                Error::new(
                                            ErrorDetails::InvalidAnthropicCompatibleRequest {
                                                message: format!("Invalid base64 image: {e}"),
                                            },
                                        )
                                            })?;
                                    tensorzero_types::file::File::Base64(base64_file)
                                }
                                "url" => {
                                    let url_str = source.url.ok_or_else(|| {
                                        Error::new(
                                            ErrorDetails::InvalidAnthropicCompatibleRequest {
                                                message:
                                                    "`url` is required for url-type image source"
                                                        .to_string(),
                                            },
                                        )
                                    })?;
                                    let url: url::Url = url_str.parse().map_err(|e| {
                                        Error::new(
                                            ErrorDetails::InvalidAnthropicCompatibleRequest {
                                                message: format!("Invalid image URL: {e}"),
                                            },
                                        )
                                    })?;
                                    let mime_type = source
                                        .media_type
                                        .as_deref()
                                        .and_then(|mt| mt.parse::<mime::MediaType>().ok());
                                    tensorzero_types::file::File::Url(UrlFile {
                                        url,
                                        mime_type,
                                        detail: None,
                                        filename: None,
                                    })
                                }
                                other => {
                                    return Err(ErrorDetails::InvalidAnthropicCompatibleRequest {
                                        message: format!(
                                            "Unsupported image source type: `{other}`. Supported types: `base64`, `url`."
                                        ),
                                    }
                                    .into());
                                }
                            };
                            contents.push(InputMessageContent::File(file));
                        }
                        AnthropicContentBlock::ToolUse { id, name, input } => {
                            contents.push(InputMessageContent::ToolCall(
                                ToolCallWrapper::InferenceResponseToolCall(
                                    InferenceResponseToolCall {
                                        id,
                                        raw_name: name,
                                        raw_arguments: input.to_string(),
                                        name: None,
                                        arguments: None,
                                    },
                                ),
                            ));
                        }
                        AnthropicContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            let result_text = match content {
                                Some(Value::String(s)) => s,
                                Some(Value::Array(blocks)) => {
                                    let mut parts = Vec::new();
                                    for block in blocks {
                                        if let Some(text) =
                                            block.get("text").and_then(|t| t.as_str())
                                        {
                                            parts.push(text.to_string());
                                        }
                                    }
                                    parts.join("\n")
                                }
                                Some(other) => other.to_string(),
                                None => String::new(),
                            };
                            contents.push(InputMessageContent::ToolResult(ToolResult {
                                id: tool_use_id,
                                name: String::new(), // Anthropic format doesn't include name in tool_result
                                result: result_text,
                            }));
                        }
                        AnthropicContentBlock::Thinking {
                            thinking,
                            signature,
                        } => {
                            contents.push(InputMessageContent::Thought(Thought {
                                text: Some(thinking),
                                signature,
                                summary: None,
                                provider_type: None,
                                extra_data: None,
                            }));
                        }
                    }
                }
                contents
            }
        };

        input_messages.push(InputMessage {
            role,
            content: content_blocks,
        });
    }

    Ok(Input {
        system: system_message,
        messages: input_messages,
    })
}

/// Convert Anthropic tool definitions and tool choice to TensorZero tool params.
fn convert_anthropic_tools(
    tools: Option<Vec<AnthropicTool>>,
    tool_choice: Option<AnthropicToolChoice>,
) -> (Option<crate::tool::ToolChoice>, Option<Vec<Tool>>) {
    let additional_tools = tools.map(|tools| {
        tools
            .into_iter()
            .map(|t| {
                Tool::Function(FunctionTool {
                    name: t.name,
                    description: t.description.unwrap_or_default(),
                    parameters: t.input_schema,
                    strict: false,
                })
            })
            .collect()
    });

    let tool_choice = match tool_choice {
        Some(AnthropicToolChoice::Auto) => Some(crate::tool::ToolChoice::Auto),
        Some(AnthropicToolChoice::Any) => Some(crate::tool::ToolChoice::Required),
        Some(AnthropicToolChoice::Tool { name }) => Some(crate::tool::ToolChoice::Specific(name)),
        Some(AnthropicToolChoice::None) => Some(crate::tool::ToolChoice::None),
        None => None,
    };

    (tool_choice, additional_tools)
}

/// Convert TensorZero chat content blocks to Anthropic response content blocks.
fn chat_content_to_anthropic_blocks(
    content: Vec<ContentBlockChatOutput>,
) -> Vec<AnthropicResponseContentBlock> {
    let mut blocks = Vec::new();
    for block in content {
        match block {
            ContentBlockChatOutput::Text(text) => {
                blocks.push(AnthropicResponseContentBlock::Text { text: text.text });
            }
            ContentBlockChatOutput::ToolCall(tool_call) => {
                let input = serde_json::from_str(&tool_call.raw_arguments)
                    .unwrap_or_else(|_| Value::String(tool_call.raw_arguments.clone()));
                blocks.push(AnthropicResponseContentBlock::ToolUse {
                    id: tool_call.id,
                    name: tool_call.raw_name,
                    input,
                });
            }
            ContentBlockChatOutput::Thought(thought) => {
                blocks.push(AnthropicResponseContentBlock::Thinking {
                    thinking: thought.text.unwrap_or_default(),
                    signature: thought.signature,
                });
            }
            ContentBlockChatOutput::Unknown(_) => {
                // Skip unknown content blocks in Anthropic format
            }
        }
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_anthropic_stop_reason_from_finish_reason() {
        assert_eq!(
            AnthropicStopReason::from(FinishReason::Stop),
            AnthropicStopReason::EndTurn
        );
        assert_eq!(
            AnthropicStopReason::from(FinishReason::Length),
            AnthropicStopReason::MaxTokens
        );
        assert_eq!(
            AnthropicStopReason::from(FinishReason::ToolCall),
            AnthropicStopReason::ToolUse
        );
        assert_eq!(
            AnthropicStopReason::from(FinishReason::StopSequence),
            AnthropicStopReason::StopSequence
        );
    }

    #[test]
    fn test_anthropic_usage_from_internal() {
        let usage = crate::inference::types::Usage {
            input_tokens: Some(100),
            output_tokens: Some(50),
            provider_cache_read_input_tokens: Some(80),
            provider_cache_write_input_tokens: Some(20),
            cost: Some(Decimal::new(5, 2)),
        };
        let anthropic_usage: AnthropicUsage = usage.into();
        assert_eq!(anthropic_usage.input_tokens, 100);
        assert_eq!(anthropic_usage.output_tokens, 50);
        assert_eq!(anthropic_usage.cache_read_input_tokens, Some(80));
        assert_eq!(anthropic_usage.cache_creation_input_tokens, Some(20));
    }

    #[test]
    fn test_anthropic_messages_to_input_simple() {
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: AnthropicMessageContent::Text("Hello".to_string()),
        }];
        let input = anthropic_messages_to_input(messages, None).expect("should parse");
        assert_eq!(input.messages.len(), 1);
    }

    #[test]
    fn test_anthropic_messages_to_input_with_system() {
        let messages = vec![AnthropicMessage {
            role: AnthropicRole::User,
            content: AnthropicMessageContent::Text("Hello".to_string()),
        }];
        let system = Some(json!("You are a helpful assistant."));
        let input = anthropic_messages_to_input(messages, system).expect("should parse");
        assert!(input.system.is_some());
    }

    #[test]
    fn test_model_prefix_parsing() {
        let params = AnthropicMessagesParams {
            model: "tensorzero::function_name::my_func".to_string(),
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text("test".to_string()),
            }],
            max_tokens: 1024,
            system: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            tensorzero_variant_name: None,
            tensorzero_dryrun: None,
            tensorzero_episode_id: None,
            tensorzero_namespace: None,
            tensorzero_cache_options: None,
            tensorzero_extra_body: Default::default(),
            tensorzero_extra_headers: Default::default(),
            tensorzero_tags: Default::default(),
            tensorzero_credentials: Default::default(),
            tensorzero_internal_dynamic_variant_config: None,
            tensorzero_provider_tools: Default::default(),
            tensorzero_params: None,
            tensorzero_include_raw_usage: false,
            tensorzero_include_original_response: false,
            tensorzero_include_raw_response: false,
        };
        let p = Params::try_from_anthropic(params).expect("should parse");
        assert_eq!(p.function_name.as_deref(), Some("my_func"));
        assert_eq!(p.model_name, None);
    }

    #[test]
    fn test_invalid_model_prefix() {
        let params = AnthropicMessagesParams {
            model: "claude-3-opus".to_string(),
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text("test".to_string()),
            }],
            max_tokens: 1024,
            system: None,
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            tensorzero_variant_name: None,
            tensorzero_dryrun: None,
            tensorzero_episode_id: None,
            tensorzero_namespace: None,
            tensorzero_cache_options: None,
            tensorzero_extra_body: Default::default(),
            tensorzero_extra_headers: Default::default(),
            tensorzero_tags: Default::default(),
            tensorzero_credentials: Default::default(),
            tensorzero_internal_dynamic_variant_config: None,
            tensorzero_provider_tools: Default::default(),
            tensorzero_params: None,
            tensorzero_include_raw_usage: false,
            tensorzero_include_original_response: false,
            tensorzero_include_raw_response: false,
        };
        assert!(Params::try_from_anthropic(params).is_err());
    }
}
