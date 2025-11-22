# API Reference: Image Generation & Content Generation

This document details the API parameters, methods, and return values for OpenAI's `gpt-image-1` and Google's Gemini models (via `generateContent`) based on the provided reference documentation.

---

## OpenAI Image API

### 1. Create Image (Generation)

**Endpoint:** `POST https://api.openai.com/v1/images/generations`

Creates an image given a prompt.

#### Request Body Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `prompt` | string | **Yes** | A text description of the desired image(s). Max 32,000 chars (`gpt-image-1`), 4000 (`dall-e-3`), 1000 (`dall-e-2`). |
| `model` | string | No | The model to use. Defaults to `dall-e-2`. Options: `dall-e-2`, `dall-e-3`, `gpt-image-1`. |
| `n` | integer | No | Number of images to generate (1-10). Default: 1. For `dall-e-3`, only `n=1` is supported. |
| `quality` | string | No | Image quality. `standard` or `hd`. Default: `standard`. `gpt-image-1` supports `low`, `medium`, `high` (default `auto`). |
| `response_format` | string | No | Format of generated images. `url` or `b64_json`. Default: `url`. `gpt-image-1` does not support `url` (returns `b64_json`). |
| `size` | string | No | Image size. e.g., `1024x1024`. See PDF for model-specific sizes. Default: `1024x1024`. |
| `style` | string | No | Style of generated images (`dall-e-3` only). `vivid` or `natural`. Default: `vivid`. |
| `user` | string | No | Unique identifier for end-user. |
| `background` | string | No | (`gpt-image-1` only) Transparency setting: `transparent`, `opaque`, `auto`. Default: `auto`. |
| `moderation` | string | No | (`gpt-image-1` only) Moderation level: `low`, `auto`. Default: `auto`. |
| `output_format` | string | No | (`gpt-image-1` only) `png`, `jpeg`, `webp`. Default: `png`. |
| `output_compression`| integer | No | (`gpt-image-1` only) Compression (0-100) for `webp`/`jpeg`. Default: 100. |
| `partial_images` | integer | No | (`gpt-image-1` only) Number of partial images for streaming (0-3). Default: 0. |
| `stream` | boolean | No | (`gpt-image-1` only) Enable streaming. Default: `false`. |

### 2. Create Image Edit

**Endpoint:** `POST https://api.openai.com/v1/images/edits`

Creates an edited or extended image given source image(s) and a prompt. Supported by `dall-e-2` and `gpt-image-1`.

#### Request Body Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `image` | file/array | **Yes** | Image(s) to edit. PNG (all), WebP/JPG (`gpt-image-1`). < 4MB (`dall-e-2`), < 50MB (`gpt-image-1`). |
| `prompt` | string | **Yes** | Text description. Max length varies by model. |
| `mask` | file | No | Mask image. Transparent areas indicate edit regions. |
| `model` | string | No | `dall-e-2` or `gpt-image-1`. Default: `dall-e-2`. |
| `n` | integer | No | Number of images (1-10). Default: 1. |
| `size` | string | No | Image size. Default: `1024x1024`. |
| `response_format` | string | No | `url` or `b64_json`. Default: `url`. |
| `user` | string | No | User ID. |
| `background` | string | No | (`gpt-image-1` only) `transparent`, `opaque`, `auto`. |
| `input_fidelity` | string | No | (`gpt-image-1` only) `high`, `low`. Default: `low`. |
| `output_compression`| integer | No | (`gpt-image-1` only) Compression level (0-100). |
| `output_format` | string | No | (`gpt-image-1` only) `png`, `jpeg`, `webp`. |

### 3. Create Image Variation

**Endpoint:** `POST https://api.openai.com/v1/images/variations`

Creates a variation of a given image. Only supported by `dall-e-2`.

#### Request Body Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `image` | file | **Yes** | Base image. PNG, < 4MB, square. |
| `model` | string | No | Defaults to `dall-e-2`. |
| `n` | integer | No | Number of images (1-10). Default: 1. |
| `response_format` | string | No | `url` or `b64_json`. Default: `url`. |
| `size` | string | No | `256x256`, `512x512`, `1024x1024`. Default: `1024x1024`. |
| `user` | string | No | User ID. |

---

## Gemini API (generateContent)

The Gemini API uses a unified `generateContent` method for text, image, audio, and code generation.

**Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
**Stream Endpoint:** `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent`

### Request Body: `GenerateContentRequest`

| Field | Type | Description |
| :--- | :--- | :--- |
| `contents` | Array of [`Content`](#content) | **Required**. The content of the current conversation with the model. |
| `tools` | Array of [`Tool`](#tool) | Optional. A list of tools the model may use to generate the next response. |
| `toolConfig` | [`ToolConfig`](#toolconfig) | Optional. Tool configuration for any `Tool` specified in the request. |
| `safetySettings` | Array of [`SafetySetting`](#safetysetting) | Optional. A list of unique `SafetySetting` instances for blocking unsafe content. |
| `systemInstruction` | [`Content`](#content) | Optional. Developer set system instructions. Currently text only. |
| `generationConfig` | [`GenerationConfig`](#generationconfig) | Optional. Configuration options for model generation and outputs. |
| `cachedContent` | string | Optional. The name of the cached content to use as context. Format: `cachedContents/{cachedContent}`. |

### Response Body: `GenerateContentResponse`

| Field | Type | Description |
| :--- | :--- | :--- |
| `candidates` | Array of [`Candidate`](#candidate) | Candidate responses from the model. |
| `promptFeedback` | [`PromptFeedback`](#promptfeedback) | Returns the prompt's feedback related to content filters. |
| `usageMetadata` | [`UsageMetadata`](#usagemetadata) | Output only. Metadata on the generation request's token usage. |
| `modelVersion` | string | Output only. The model version used to generate the response. |

---

### Data Types & Definitions

#### Content
The base unit of input/output.
| Field | Type | Description |
| :--- | :--- | :--- |
| `parts` | Array of [`Part`](#part) | Ordered parts that constitute a single message. |
| `role` | string | Optional. The producer of the content (`user` or `model`). |

#### Part
A single part of a message.
| Field | Type | Description |
| :--- | :--- | :--- |
| `text` | string | Inline text. |
| `inlineData` | [`Blob`](#blob) | Inline media bytes. |
| `functionCall` | [`FunctionCall`](#functioncall) | A predicted function call. |
| `functionResponse` | [`FunctionResponse`](#functionresponse) | Result of a function call. |
| `fileData` | [`FileData`](#filedata) | URI based data. |
| `executableCode` | [`ExecutableCode`](#executablecode) | Code to be executed. |
| `codeExecutionResult` | [`CodeExecutionResult`](#codeexecutionresult) | Result of code execution. |
| `thought` | boolean | Optional. Indicates if the part is thought from the model. |
| `thoughtSignature` | string | Optional. An opaque signature for the thought so it can be reused in subsequent requests. |
| `partMetadata` | object | Optional. Custom metadata associated with the Part. |
| `videoMetadata` | [`VideoMetadata`](#videometadata) | Optional. Video metadata. |

#### Blob
| Field | Type | Description |
| :--- | :--- | :--- |
| `mimeType` | string | The IANA standard MIME type of the source data. |
| `data` | string (bytes) | Raw bytes for media formats. Base64 encoded in JSON. |

#### VideoMetadata
| Field | Type | Description |
| :--- | :--- | :--- |
| `startOffset` | string | Optional. The start offset of the video. |
| `endOffset` | string | Optional. The end offset of the video. |
| `fps` | number | Optional. The frame rate of the video. |

#### GenerationConfig
| Field | Type | Description |
| :--- | :--- | :--- |
| `stopSequences` | Array of string | Character sequences (up to 5) that will stop output generation. |
| `responseMimeType` | string | MIME type of generated text (`text/plain`, `application/json`, `text/x.enum`). |
| `responseSchema` | [`Schema`](#schema) | Output schema for generated candidate text (subset of OpenAPI schema). |
| `candidateCount` | integer | Number of generated responses to return. Default 1. |
| `maxOutputTokens` | integer | Max number of tokens to include in a response candidate. |
| `temperature` | number | Controls randomness [0.0, 2.0]. |
| `topP` | number | Nucleus sampling probability. |
| `topK` | integer | Top-k sampling count. |
| `seed` | integer | Seed used in decoding. |
| `presencePenalty` | number | Penalty for token presence. |
| `frequencyPenalty` | number | Penalty for token frequency. |
| `responseLogprobs` | boolean | If true, export logprobs results. |
| `logprobs` | integer | Number of top logprobs to return [0, 20]. |
| `enableEnhancedCivicAnswers` | boolean | Enables enhanced civic answers. |
| `speechConfig` | [`SpeechConfig`](#speechconfig) | Speech generation config. |
| `thinkingConfig` | [`ThinkingConfig`](#thinkingconfig) | Config for thinking features. |
| `imageConfig` | [`ImageConfig`](#imageconfig) | Config for image generation. |
| `mediaResolution` | enum | `MEDIA_RESOLUTION_LOW`, `MEDIUM`, `HIGH`. |

#### ImageConfig
| Field | Type | Description |
| :--- | :--- | :--- |
| `aspectRatio` | string | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9`. |
| `imageSize` | string | `1K`, `2K`, `4K`. |

#### ThinkingConfig
| Field | Type | Description |
| :--- | :--- | :--- |
| `includeThoughts` | boolean | Whether to include thoughts in response. |
| `thinkingBudget` | integer | Token budget for thoughts. |
| `thinkingLevel` | enum | `LOW`, `HIGH`. |

#### Tool
| Field | Type | Description |
| :--- | :--- | :--- |
| `functionDeclarations` | Array of `FunctionDeclaration` | List of functions the model can call. |
| `googleSearchRetrieval` | `GoogleSearchRetrieval` | Retrieval tool details. |
| `codeExecution` | `CodeExecution` | Enables code execution. |
| `googleSearch` | `GoogleSearch` | Enables Google Search. |

#### ToolConfig
| Field | Type | Description |
| :--- | :--- | :--- |
| `functionCallingConfig` | `FunctionCallingConfig` | Function calling mode/config. |

#### SafetySetting
| Field | Type | Description |
| :--- | :--- | :--- |
| `category` | enum ([`HarmCategory`](#harmcategory)) | **Required**. The category for this setting. |
| `threshold` | enum ([`HarmBlockThreshold`](#harmblockthreshold)) | **Required**. The probability threshold for blocking. |

#### Candidate
| Field | Type | Description |
| :--- | :--- | :--- |
| `content` | [`Content`](#content) | Generated content returned from the model. |
| `finishReason` | enum ([`FinishReason`](#finishreason)) | The reason why the model stopped generating. |
| `safetyRatings` | Array of [`SafetyRating`](#safetyrating) | Ratings for safety of the response. |
| `citationMetadata` | [`CitationMetadata`](#citationmetadata) | Citation info for model-generated candidate. |
| `tokenCount` | integer | Token count for this candidate. |
| `groundingAttributions` | Array of `GroundingAttribution` | Attribution info for sources. |
| `groundingMetadata` | [`GroundingMetadata`](#groundingmetadata) | Grounding metadata for the candidate. |
| `avgLogprobs` | number | Average log probability score. |
| `logprobsResult` | [`LogprobsResult`](#logprobsresult) | Log-likelihood scores for response tokens. |
| `index` | integer | Index of the candidate. |
| `finishMessage` | string | Details on finish reason. |

#### PromptFeedback
| Field | Type | Description |
| :--- | :--- | :--- |
| `blockReason` | enum | `SAFETY`, `OTHER`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `IMAGE_SAFETY`. |
| `safetyRatings` | Array of [`SafetyRating`](#safetyrating) | Safety ratings for the prompt. |

#### UsageMetadata
| Field | Type | Description |
| :--- | :--- | :--- |
| `promptTokenCount` | integer | Tokens in prompt. |
| `cachedContentTokenCount` | integer | Tokens in cached content. |
| `candidatesTokenCount` | integer | Tokens in candidates. |
| `totalTokenCount` | integer | Total tokens. |

#### Enums

**HarmCategory:**
`HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_SEXUALLY_EXPLICIT`, `HARM_CATEGORY_DANGEROUS_CONTENT`, `HARM_CATEGORY_HARASSMENT`, `HARM_CATEGORY_CIVIC_INTEGRITY` (Deprecated).

**HarmBlockThreshold:**
`BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`, `BLOCK_NONE`, `OFF`.

**FinishReason:**
`STOP`, `MAX_TOKENS`, `SAFETY`, `RECITATION`, `LANGUAGE`, `OTHER`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `MALFORMED_FUNCTION_CALL`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_OTHER`, `NO_IMAGE`, `IMAGE_RECITATION`, `UNEXPECTED_TOOL_CALL`, `TOO_MANY_TOOL_CALLS`, `MISSING_THOUGHT_SIGNATURE`.

#### GroundingMetadata
| Field | Type | Description |
| :--- | :--- | :--- |
| `groundingChunks` | Array of `GroundingChunk` | Supporting references. |
| `groundingSupports` | Array of `GroundingSupport` | Support scores. |
| `webSearchQueries` | Array of string | Web search queries used. |
| `searchEntryPoint` | `SearchEntryPoint` | Google search entry point. |
| `retrievalMetadata` | `RetrievalMetadata` | Retrieval metadata. |
| `googleMapsWidgetContextToken` | string | Google Maps widget token. |

#### LogprobsResult
| Field | Type | Description |
| :--- | :--- | :--- |
| `topCandidates` | Array of `TopCandidates` | Top candidates at each step. |
| `chosenCandidates` | Array of `Candidate` | Chosen candidates at each step. |
| `logProbabilitySum` | number | Sum of log probabilities. |