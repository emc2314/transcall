# API Reference: Image Generation & Content Generation

This document details the API parameters, methods, and return values for OpenAI's `gpt-image-1` and Google's Gemini models (via `generateContent`) based on the provided reference documentation.

---

## OpenAI Image API

### 1. Create Image (Generation)

**Endpoint:** `POST https://api.openai.com/v1/images/generations`

Creates an image given a prompt.

#### Request Body Parameters

| Parameter          | Type                                        | Required | Description                                                                                                                                                                             |
| :----------------- | :------------------------------------------ | :------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`           | string                                      | **Yes**  | A text description of the desired image(s). Max 32,000 chars (`gpt-image-1`), 4000 (`dall-e-3`), 1000 (`dall-e-2`).                                                                    |
| `background`       | string (`auto`, `transparent`, `opaque`)    | No       | (**`gpt-image-1` only**) Allows setting transparency for the background. Defaults to `auto`.                                                                                            |
| `model`            | string (`dall-e-2`, `dall-e-3`, `gpt-image-1`) | No       | The model to use. Defaults to `dall-e-2`.                                                                                                                                               |
| `moderation`       | string (`low`, `auto`)                      | No       | (**`gpt-image-1` only**) Controls the content-moderation level for generated images. Defaults to `auto`.                                                                               |
| `n`                | integer                                     | No       | Number of images to generate (1-10). Default: 1. For `dall-e-3`, only `n=1` is supported.                                                                                               |
| `output_compression` | integer (0-100)                             | No       | (**`gpt-image-1` only**) The compression level (0-100%) for generated images. Defaults to 100. Supported with `webp` or `jpeg` output formats.                                          |
| `output_format`    | string (`png`, `jpeg`, `webp`)              | No       | (**`gpt-image-1` only**) The format in which generated images are returned. Defaults to `png`.                                                                                          |
| `partial_images`   | integer (0-3)                               | No       | The number of partial images to generate for streaming responses. Defaults to 0.                                                                                                        |
| `quality`          | string (`auto`, `high`, `medium`, `low`, `hd`, `standard`) | No       | The quality of the generated image. Defaults to `auto`.                                                                                                                                 |
| `response_format`  | string (`url`, `b64_json`)                  | No       | The format for returned images. Defaults to `url`. `gpt-image-1` always returns base64-encoded images.                                                                                  |
| `size`             | string (`1024x1024`, `1536x1024`, etc.)      | No       | The size of the generated images. Defaults to `auto`.                                                                                                                                   |
| `stream`           | boolean                                     | No       | (**`gpt-image-1` only**) Generates the image in streaming mode. Defaults to `false`.                                                                                                    |
| `style`            | string (`vivid`, `natural`)                 | No       | (**`dall-e-3` only**) The style of the generated images. Defaults to `vivid`.                                                                                                           |
| `user`             | string                                      | No       | A unique identifier for the end-user.                                                                                                                                                   |

#### Response Structure

The response returns an object with the following top-level fields:

| Field             | Type                                     | Description                                                                                             |
| :---------------- | :--------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| `created`         | integer (Unix timestamp)                 | The Unix timestamp of when the image was created.                                                       |
| `data`            | array of objects                         | An array of image objects. Each object contains either `b64_json` or `url`.                             |
| `data.b64_json`   | string                                   | (Optional) The base64-encoded image data. Present if `response_format` is `b64_json`.                   |
| `data.url`        | string                                   | (Optional) The URL of the generated image. Present if `response_format` is `url`. Valid for 60 minutes. |
| `data.revised_prompt` | string                               | (**`dall-e-3` only**) The actual prompt used by the model, which may be a revised version of the input. |
| `background`      | string (`transparent`, `opaque`, `auto`) | (**`gpt-image-1` only**) The background parameter used for generation.                                  |
| `output_format`   | string (`png`, `webp`, `jpeg`)           | (**`gpt-image-1` only**) The output format used for the generated image.                                |
| `quality`         | string (`low`, `medium`, `high`)         | (**`gpt-image-1` only**) The quality of the generated image.                                            |
| `size`            | string (`1024x1024`, etc.)               | (**`gpt-image-1` only**) The size of the generated image.                                               |
| `usage`           | object                                   | (**`gpt-image-1` only**) Token usage information for the generation.                                    |

### 2. Create Image Edit

**Endpoint:** `POST https://api.openai.com/v1/images/edits`

Creates an edited image given source image(s) and a prompt, optionally with a mask. Supported by `dall-e-2` and `gpt-image-1`.

#### Request Body Parameters

| Parameter            | Type                                      | Required | Description                                                                                                                                                                                          |
| :------------------- | :---------------------------------------- | :------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image`              | string or array                           | **Yes**  | The image(s) to edit. For `gpt-image-1`, up to 16 `png`, `webp`, or `jpg` files (<50MB). For `dall-e-2`, one square `png` file (<4MB).                                                            |
| `prompt`             | string                                    | **Yes**  | A text description of the desired image modification. Max 32000 chars (`gpt-image-1`), 1000 (`dall-e-2`).                                                                                             |
| `background`         | string (`auto`, `transparent`, `opaque`)  | No       | (**`gpt-image-1` only**) Allows setting transparency for the background. Defaults to `auto`.                                                                                                        |
| `input_fidelity`     | string (`low`, `high`)                    | No       | (**`gpt-image-1` only**) Controls how much effort the model exerts to match the style and features of input images. Defaults to `low`.                                                               |
| `mask`               | file                                      | No       | An additional PNG image (<4MB, same dimensions as `image`) where transparent areas indicate edit regions. Opaque areas remain unchanged.                                                                 |
| `model`              | string (`dall-e-2`, `gpt-image-1`)        | No       | The model to use for image generation. Defaults to `dall-e-2`.                                                                                                                                       |
| `n`                  | integer                                   | No       | Number of images to generate (1-10). Default: 1.                                                                                                                                                     |
| `output_compression` | integer (0-100)                           | No       | (**`gpt-image-1` only**) The compression level (0-100%) for generated images. Defaults to 100. Supported with `webp` or `jpeg` output formats.                                                    |
| `output_format`      | string (`png`, `jpeg`, `webp`)            | No       | (**`gpt-image-1` only**) The format in which generated images are returned. Defaults to `png`.                                                                                                      |
| `partial_images`     | integer (0-3)                             | No       | The number of partial images to generate for streaming responses. Defaults to 0.                                                                                                                     |
| `quality`            | string (`auto`, `high`, `medium`, `low`, `standard`) | No       | The quality of the generated image. Defaults to `auto`.                                                                                                                                              |
| `response_format`    | string (`url`, `b64_json`)                | No       | The format for returned images. Defaults to `url`. `gpt-image-1` always returns base64-encoded images.                                                                                              |
| `size`               | string (`1024x1024`, etc.)                | No       | The size of the generated images. Defaults to `1024x1024`.                                                                                                                                           |
| `stream`             | boolean                                   | No       | (**`gpt-image-1` only**) Edits the image in streaming mode. Defaults to `false`.                                                                                                                     |
| `user`               | string                                    | No       | A unique identifier for the end-user.                                                                                                                                                                |

#### Response Structure

The response structure is the same as for "Create Image (Generation)".

### 3. Create Image Variation

**Endpoint:** `POST https://api.openai.com/v1/images/variations`

Creates a variation of a given image. Only supported by `dall-e-2`.

#### Request Body Parameters

| Parameter         | Type     | Required | Description                                     |
| :---------------- | :------- | :------- | :---------------------------------------------- |
| `image`           | file     | **Yes**  | Base image. PNG, < 4MB, square.                 |
| `model`           | string   | No       | Defaults to `dall-e-2`.                         |
| `n`               | integer  | No       | Number of images (1-10). Default: 1.            |
| `response_format` | string   | No       | `url` or `b64_json`. Default: `url`.            |
| `size`            | string   | No       | `256x256`, `512x512`, `1024x1024`. Default: `1024x1024`. |
| `user`            | string   | No       | User ID.                                        |

---

## Gemini API (generateContent)

The Gemini API uses a unified `generateContent` method for text, image, audio, and code generation. The `v1` and `v1beta1` versions largely share the same core request/response structures, with `v1beta1` potentially including more experimental features.

**Endpoint:** `POST https://generativelanguage.googleapis.com/v1/models/{model}:generateContent`
**Stream Endpoint:** `POST https://generativelanguage.googleapis.com/v1/models/{model}:streamGenerateContent`

### Request Body: `GenerateContentRequest`

| Field             | Type                                | Description                                                                                                                                          |
| :---------------- | :---------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| `contents`        | Array of [`Content`](#content)      | **Required**. The content of the current conversation with the model. For single-turn interactions, it contains a single instance. For multi-turn conversations, it's a repeated field that includes the conversation history along with the latest request. |
| `generationConfig`| [`GenerationConfig`](#generationconfig) | Optional. Configuration options for model generation and outputs.                                                                                    |
| `safetySettings`  | Array of [`SafetySetting`](#safetysetting) | Optional. A list of unique `SafetySetting` instances for blocking unsafe content.                                                                    |
| `tools`           | Array of [`Tool`](#tool)            | Optional. A list of tools the model may use to generate the next response.                                                                           |
| `toolConfig`      | [`ToolConfig`](#toolconfig)         | Optional. Tool configuration for any `Tool` specified in the request.                                                                                |
| `systemInstruction` | [`Content`](#content)               | Optional. Developer-set system instructions to guide model behavior. Currently text only.                                                          |
| `model`           | string                              | Optional. Specifies the model to be used for content generation.                                                                                     |
| `cachedContent`   | string                              | Optional. The name of the cached content to use as context. Format: `cachedContents/{cachedContent}`.                                                |
| `labels`          | map<string, string>                 | Optional. User-defined metadata for the request, used for billing and reporting.                                                                     |

### Response Body: `GenerateContentResponse`

| Field             | Type                                    | Description                                                                                                                                                                                                                                              |
| :---------------- | :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `candidates`      | Array of [`Candidate`](#candidate)      | Candidate responses from the model.                                                                                                                                                                                                      |
| `modelVersion`    | string                                  | Output only. The model version used to generate the response.                                                                                                                                                                            |
| `createTime`      | string (RFC 3339 timestamp)             | Output only. The timestamp when the request was made to the server.                                                                                                                                                                      |
| `responseId`      | string                                  | Output only. A unique identifier for each response.                                                                                                                                                                                      |
| `promptFeedback`  | [`PromptFeedback`](#promptfeedback)     | Returns the prompt's feedback related to content filters. Only sent in the first stream chunk and when no candidates were generated due to content violations.                                                                             |
| `usageMetadata`   | [`UsageMetadata`](#usagemetadata)       | Output only. Metadata on the generation request's token usage.                                                                                                                                                                           |

---

### Data Types & Definitions (Gemini)

#### Content
The base unit of input/output, representing a part of the conversation.

| Field | Type                        | Description                                     |
| :---- | :-------------------------- | :---------------------------------------------- |
| `parts` | Array of [`Part`](#part)    | Ordered parts that constitute a single message. |
| `role`  | string (`user`, `model`)    | Optional. The producer of the content.          |

#### Part
A single part of a message, which can be text, inline data, function calls, etc.

| Field                   | Type                              | Description                                                                                                                                         |
| :---------------------- | :-------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| `text`                  | string                            | Inline text.                                                                                                                                        |
| `inlineData`            | [`Blob`](#blob)                   | Inline media bytes.                                                                                                                                 |
| `fileData`              | [`FileData`](#fileData)           | URI based data, typically for Google Cloud Storage files.                                                                                           |
| `functionCall`          | [`FunctionCall`](#functionCall)   | A predicted function call by the model.                                                                                                             |
| `functionResponse`      | [`FunctionResponse`](#functionResponse) | Result of a function call provided back to the model.                                                                                               |
| `executableCode`        | [`ExecutableCode`](#executableCode) | Code to be executed (for code execution tools).                                                                                                     |
| `codeExecutionResult`   | [`CodeExecutionResult`](#codeExecutionResult) | Result of code execution.                                                                                                                           |
| `thought`               | boolean or string or object       | Optional. Indicates if the part is a thought from the model, or the thought content itself.                                                         |
| `thoughtSignature`      | string                            | Optional. An opaque signature for the thought, allowing reuse in subsequent requests.                                                               |
| `partMetadata`          | object                            | Optional. Custom metadata associated with the Part.                                                                                                 |
| `videoMetadata`         | [`VideoMetadata`](#videoMetadata) | Optional. Metadata for video parts.                                                                                                                 |

#### Blob
Represents raw media bytes.

| Field      | Type     | Description                                               |
| :--------- | :------- | :-------------------------------------------------------- |
| `mimeType` | string   | The IANA standard MIME type of the source data (e.g., `image/png`). |
| `data`     | string (base64) | Raw bytes for media formats, base64 encoded in JSON.      |

#### FileData
Represents data from a URI.

| Field      | Type     | Description                                               |
| :--------- | :------- | :-------------------------------------------------------- |
| `mimeType` | string   | The IANA standard MIME type of the source data.             |
| `fileUri`  | string   | The URI of the file (e.g., `gs://bucket/object`).         |

#### FunctionCall
Represents a function call predicted by the model.

| Field       | Type                 | Description                       |
| :---------- | :------------------- | :-------------------------------- |
| `name`      | string               | The name of the function to call. |
| `args`      | map<string, object>  | The arguments to the function.    |

#### FunctionResponse
Represents the result of a function call.

| Field      | Type                 | Description                                                 |
| :--------- | :------------------- | :---------------------------------------------------------- |
| `name`     | string               | The name of the function that was called.                   |
| `response` | map<string, object>  | The result of the function call, in JSON format.          |

#### ExecutableCode
Represents code to be executed.

| Field    | Type     | Description            |
| :------- | :------- | :--------------------- |
| `language` | string   | The programming language (e.g., `python`). |
| `code`   | string   | The code to execute.   |

#### CodeExecutionResult
Represents the result of code execution.

| Field     | Type     | Description                                    |
| :-------- | :------- | :--------------------------------------------- |
| `output`  | string   | The standard output from the execution.        |
| `error`   | string   | The standard error from the execution.         |
| `exitCode` | integer  | The exit code of the execution.                |

#### VideoMetadata
Metadata for video parts.

| Field       | Type     | Description                       |
| :---------- | :------- | :-------------------------------- |
| `startOffset` | string   | Optional. The start offset of the video (duration). |
| `endOffset`   | string   | Optional. The end offset of the video (duration). |
| `fps`         | number   | Optional. The frame rate of the video.    |

#### GenerationConfig
Configuration options for model generation and outputs.

| Field                  | Type                        | Description                                                                                             |
| :--------------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------ |
| `stopSequences`        | Array of string             | Character sequences (up to 5) that will stop output generation.                                         |
| `responseMimeType`     | string (`text/plain`, `application/json`, `text/x.enum`) | MIME type of generated text. Used to request structured JSON output.                                    |
| `responseSchema`       | [`Schema`](#schema)         | Output schema for generated candidate text (subset of OpenAPI schema).                                  |
| `candidateCount`       | integer                     | Number of generated responses to return. Default 1.                                                     |
| `maxOutputTokens`      | integer                     | Max number of tokens to include in a response candidate.                                                |
| `temperature`          | number ([0.0, 2.0])         | Controls randomness. Higher values mean more random output.                                             |
| `topP`                 | number                      | Nucleus sampling probability.                                                                           |
| `topK`                 | integer                     | Top-k sampling count.                                                                                   |
| `seed`                 | integer                     | Seed used in decoding for deterministic output.                                                         |
| `presencePenalty`      | number                      | Penalty for token presence.                                                                             |
| `frequencyPenalty`     | number                      | Penalty for token frequency.                                                                            |
| `responseLogprobs`     | boolean                     | If true, export logprobs results.                                                                       |
| `logprobs`             | integer ([0, 20])           | Number of top logprobs to return.                                                                       |
| `enableEnhancedCivicAnswers` | boolean                     | Enables enhanced civic answers.                                                                         |
| `speechConfig`         | [`SpeechConfig`](#speechconfig) | Speech generation config.                                                                               |
| `thinkingConfig`       | [`ThinkingConfig`](#thinkingconfig) | Config for thinking features.                                                                           |
| `imageConfig`          | [`ImageConfig`](#imageconfig) | Config for image generation parameters (e.g., aspect ratio, size).                                      |
| `mediaResolution`      | enum (`MEDIA_RESOLUTION_LOW`, `MEDIUM`, `HIGH`) | Specifies resolution for media outputs.                                                                 |

#### ImageConfig
Config for image generation.

| Field         | Type     | Description                               |
| :------------ | :------- | :---------------------------------------- |
| `aspectRatio` | string   | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9`. |
| `imageSize`   | string   | `1K`, `2K`, `4K`.                         |

#### ThinkingConfig
Config for thinking features.

| Field           | Type     | Description                                   |
| :-------------- | :------- | :-------------------------------------------- |
| `includeThoughts` | boolean  | Whether to include thoughts in response.      |
| `thinkingBudget`  | integer  | Token budget for thoughts.                    |
| `thinkingLevel`   | enum (`LOW`, `HIGH`) | Level of detail for thoughts.                 |

#### SpeechConfig
Config for speech generation. (Details not explicitly retrieved, assumes standard speech config fields)

#### Tool
A list of tools the model may use to generate the next response.

| Field                    | Type                                | Description                                |
| :----------------------- | :---------------------------------- | :----------------------------------------- |
| `functionDeclarations`   | Array of [`FunctionDeclaration`](#functiondeclaration) | List of functions the model can call.      |
| `googleSearchRetrieval`  | [`GoogleSearchRetrieval`](#googlesearchretrieval) | Retrieval tool details.                    |
| `codeExecution`          | [`CodeExecution`](#codeexecution)   | Enables code execution.                    |
| `googleSearch`           | [`GoogleSearch`](#googlesearch)     | Enables Google Search.                     |

#### FunctionDeclaration
Declares a function that the model can call.

| Field        | Type                          | Description                            |
| :----------- | :---------------------------- | :------------------------------------- |
| `name`       | string                        | The name of the function.              |
| `description`| string                        | An optional description of the function. |
| `parameters` | [`Schema`](#schema)           | The parameters of the function (OpenAPI schema). |

#### GoogleSearchRetrieval
Details for the Google Search Retrieval tool.

| Field        | Type        | Description                               |
| :----------- | :---------- | :---------------------------------------- |
| `disableWebSearch` | boolean | If true, disables web search.             |

#### CodeExecution
Details for the Code Execution tool. (Details not explicitly retrieved, assumes standard code execution fields)

#### GoogleSearch
Details for the Google Search tool. (Details not explicitly retrieved, assumes standard Google Search fields)

#### ToolConfig
Tool configuration for any `Tool` specified in the request.

| Field                | Type                                      | Description                  |
| :------------------- | :---------------------------------------- | :--------------------------- |
| `functionCallingConfig` | [`FunctionCallingConfig`](#functioncallingconfig) | Function calling mode/config. |

#### FunctionCallingConfig
Configuration for how the model should use functions.

| Field  | Type                                | Description                                     |
| :----- | :---------------------------------- | :---------------------------------------------- |
| `mode` | enum (`AUTO`, `ANY`, `NONE`, `REQUIRED`) | Mode for function calling.                      |
| `allowedFunctionNames` | Array of string           | Optional. List of function names the model is allowed to call. |

#### SafetySetting
A setting for blocking unsafe content.

| Field      | Type                          | Description                                         |
| :--------- | :---------------------------- | :-------------------------------------------------- |
| `category` | enum ([`HarmCategory`](#harmcategory)) | **Required**. The category for this setting.        |
| `threshold`| enum ([`HarmBlockThreshold`](#harmblockthreshold)) | **Required**. The probability threshold for blocking. |

#### HarmCategory (Enum)
Categories of potential harmful content.
`HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_SEXUALLY_EXPLICIT`, `HARM_CATEGORY_DANGEROUS_CONTENT`, `HARM_CATEGORY_HARASSMENT`, `HARM_CATEGORY_CIVIC_INTEGRITY` (Deprecated).

#### HarmBlockThreshold (Enum)
Probability thresholds for blocking content.
`BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`, `BLOCK_NONE`, `OFF`.

#### Candidate
Represents a generated output from the model.

| Field                 | Type                                    | Description                                                                                                |
| :-------------------- | :-------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| `content`             | [`Content`](#content)                   | Generated content returned from the model.                                                                 |
| `finishReason`        | enum ([`FinishReason`](#finishreason)) | The reason why the model stopped generating.                                                               |
| `safetyRatings`       | Array of [`SafetyRating`](#safetyrating) | Ratings for safety of the response.                                                                        |
| `citationMetadata`    | [`CitationMetadata`](#citationmetadata) | Citation information for model-generated candidate.                                                        |
| `tokenCount`          | integer                                 | Token count for this candidate.                                                                            |
| `groundingAttributions` | Array of `GroundingAttribution`         | Attribution information for sources used in grounding.                                                     |
| `groundingMetadata`   | [`GroundingMetadata`](#groundingmetadata) | Grounding metadata for the candidate.                                                                      |
| `webSearchQueries`    | Array of string                         | Web search queries used (if applicable).                                                                   |
| `searchEntryPoint`    | [`SearchEntryPoint`](#searchentrypoint) | Google search entry point.                                                                                 |
| `retrievalMetadata`   | [`RetrievalMetadata`](#retrievalmetadata) | Retrieval metadata.                                                                                        |
| `googleMapsWidgetContextToken` | string                           | Google Maps widget context token.                                                                          |
| `index`               | integer                                 | The 0-based index of the candidate.                                                                        |
| `finishMessage`       | string                                  | Details on finish reason.                                                                                  |
| `tokenLogProbs`       | Array of [`TokenLogProb`](#tokenlogprob) | A list of token log probabilities for each token in the `content`.                                         |

#### FinishReason (Enum)
Reasons why the model stopped generating content.
`STOP`, `MAX_TOKENS`, `SAFETY`, `RECITATION`, `LANGUAGE`, `OTHER`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `MALFORMED_FUNCTION_CALL`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`, `IMAGE_OTHER`, `NO_IMAGE`, `IMAGE_RECITATION`, `UNEXPECTED_TOOL_CALL`, `TOO_MANY_TOOL_CALLS`, `MISSING_THOUGHT_SIGNATURE`.

#### SafetyRating
Safety attributes for a generated content segment.

| Field      | Type                          | Description                                 |
| :--------- | :---------------------------- | :------------------------------------------ |
| `category` | enum ([`HarmCategory`](#harmcategory)) | The category for this rating.               |
| `probability`| enum (`UNSPECIFIED`, `NEGLIGIBLE`, `LOW`, `MEDIUM`, `HIGH`) | The probability of harm for this category. |
| `blocked`  | boolean                       | Whether this category was blocked.          |

#### CitationMetadata
Citation information for model-generated content.

| Field        | Type                           | Description                                     |
| :----------- | :----------------------------- | :---------------------------------------------- |
| `citations`  | Array of [`CitationSource`](#citationsource) | A list of citations.                            |

#### CitationSource
A single source for a citation.

| Field          | Type     | Description                                     |
| :------------- | :------- | :---------------------------------------------- |
| `uri`          | string   | The URI of the cited source.                    |
| `startIndex`   | integer  | The start index of the cited text in the content. |
| `endIndex`     | integer  | The end index of the cited text in the content.   |
| `license`      | string   | The license of the source.                      |
| `publicationDate` | [`Date`](#date) | The publication date of the source.             |

#### Date
Represents a calendar date.

| Field   | Type    | Description                |
| :------ | :------ | :------------------------- |
| `year`  | integer | Year of the date.          |
| `month` | integer | Month of the date.         |
| `day`   | integer | Day of the month.          |

#### GroundingAttribution
Attribution information for grounded content.

| Field         | Type         | Description                                       |
| :------------ | :----------- | :------------------------------------------------ |
| `segment`     | [`Segment`](#segment) | The text segment that is grounded.               |
| `confidence`  | number       | Confidence score for the attribution.             |
| `webSearchQuery` | string       | The web search query used for grounding.          |
| `retrieval`   | [`Retrieval`](#retrieval) | Retrieval information for grounding.              |

#### Segment
Represents a text segment.

| Field       | Type    | Description                               |
| :---------- | :------- | :---------------------------------------- |
| `startIndex` | integer | The start index of the segment.           |
| `endIndex`  | integer | The end index of the segment.             |
| `text`      | string  | The text of the segment.                  |

#### Retrieval
Retrieval information for grounding.

| Field      | Type        | Description                                       |
| :--------- | :---------- | :------------------------------------------------ |
| `uri`      | string      | The URI of the retrieved document.                |
| `title`    | string      | The title of the retrieved document.              |
| `snippet`  | string      | A snippet of text from the retrieved document.    |
| `query`    | string      | The query used for retrieval.                     |

#### GroundingMetadata
Metadata for grounding.

| Field            | Type                          | Description                                     |
| :--------------- | :---------------------------- | :---------------------------------------------- |
| `groundingChunks`| Array of `GroundingChunk`     | Supporting references.                          |
| `groundingSupports` | Array of `GroundingSupport`   | Support scores.                                 |
| `webSearchQueries` | Array of string               | Web search queries used.                        |
| `searchEntryPoint` | [`SearchEntryPoint`](#searchentrypoint) | Google search entry point.                      |
| `retrievalMetadata` | [`RetrievalMetadata`](#retrievalmetadata) | Retrieval metadata.                             |
| `googleMapsWidgetContextToken` | string        | Google Maps widget context token.               |

#### SearchEntryPoint
Details on the Google Search entry point. (Details not explicitly retrieved)

#### RetrievalMetadata
Metadata on retrieval. (Details not explicitly retrieved)

#### TokenLogProb
Log probability for a single token.

| Field       | Type                | Description                                     |
| :---------- | :------------------ | :---------------------------------------------- |
| `token`     | string              | The token.                                      |
| `logprob`   | number              | The log probability of the token.               |
| `topLogProbs` | Array of `TopLogProb` | Optional. Top log probabilities for the token.  |

#### TopLogProb
A top log probability entry.

| Field   | Type   | Description                                       |
| :------ | :----- | :------------------------------------------------ |
| `token` | string | The token.                                        |
| `logprob` | number | The log probability of the token.                 |

#### PromptFeedback
Feedback related to content filters for the prompt.

| Field       | Type                          | Description                                 |
| :---------- | :---------------------------- | :------------------------------------------ |
| `blockReason` | enum (`SAFETY`, `OTHER`, etc.) | The reason why the prompt was blocked.      |
| `safetyRatings` | Array of [`SafetyRating`](#safetyrating) | Safety ratings for the prompt.              |

#### UsageMetadata
Metadata on the generation request's token usage.

| Field                 | Type    | Description                               |
| :-------------------- | :------ | :---------------------------------------- |
| `promptTokenCount`    | integer | Tokens in prompt.                         |
| `cachedContentTokenCount` | integer | Tokens in cached content.                 |
| `candidatesTokenCount`| integer | Tokens in candidates.                     |
| `totalTokenCount`     | integer | Total tokens.                             |

#### Schema
Generic schema definition, typically used for tool parameters or structured output.

| Field           | Type                               | Description                                     |
| :-------------- | :--------------------------------- | :---------------------------------------------- |
| `type`          | string                             | The type of the schema (e.g., `object`, `array`, `string`, `integer`). |
| `properties`    | map<string, `Schema`>              | Properties of an object schema.                 |
| `items`         | [`Schema`](#schema)                | Items of an array schema.                       |
| `required`      | Array of string                    | Required properties for an object schema.       |
| `enum`          | Array of string                    | Allowed values for an enum schema.              |
| `description`   | string                             | A description of the schema.                    |
| `format`        | string                             | The format of the schema (e.g., `date-time`).   |
| `nullable`      | boolean                            | Whether the value can be null.                  |
| `default`       | object                             | The default value.                              |
| `example`       | object                             | An example value.                               |
| `externalDocs`  | [`ExternalDocumentation`](#externaldocumentation) | External documentation for this schema.         |

#### ExternalDocumentation
External documentation for a schema.

| Field       | Type   | Description      |
| :---------- | :----- | :--------------- |
| `description` | string | Description.     |
| `url`       | string | URL to external documentation. |