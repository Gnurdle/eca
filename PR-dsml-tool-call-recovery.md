# Recover DeepSeek DSML tool calls emitted as assistant content

## Why

DeepSeek V4/V4.1 models — including **V4.1-Flash** — may serialise tool calls
using their native **DSML (DeepSeek Markup Language)** text format instead of the
structured `tool_calls` field:

```
<｜tool_calls>
<｜invoke name="read_file">
<｜parameter name="path" string="true">src/foo.clj</｜parameter>
</｜invoke>
</｜tool_calls>
```

The delimiter is the **full-width vertical bar `｜` (U+FF5C)**. V4.1 additionally
writes the tags with whitespace inside them (the "spaced tags" variant that broke
older DeepSeek detectors), and some proxies/renders normalise the delimiter to
ASCII `|`.

When the serving stack in front of the model does not translate DSML into
structured tool calls, **the whole DSML block is delivered as ordinary assistant
content**. ECA reads tool calls only from the structured `:tool_calls` field, so
the call is never executed and never reported:

- no error is raised — a failed tool-call match is silent by design;
- the raw markup is shown to the user as if the model had written it;
- the turn ends as a normal `finish "stop"`, indistinguishable from the model
  simply answering.

This is a widely-hit class of bug — vLLM shipped a dedicated
`--tool-parser deepseek_v41`; qwen-code, sigit, egirl, opencode-deepseek and
others added "DSML / XML tool-call recovery"; and DeepSeek's own model
discussions report the markup leaking into `message.content` unparsed ("the turn
ends silently"). ECA had no DSML handling at all.

## What changed

**New `src/eca/llm_providers/dsml.clj`** — a pure, unit-tested parser plus a
streaming-safe filter:

- `dsml/extract` — parse a complete assistant text into tool calls **and** the
  text with the markup removed (so it is never shown as prose);
- `dsml/make-filter` — incremental filter for the streaming path: holds back a
  short tail so an opener split across chunks is still recognised, suppresses the
  block from the streamed text, and captures it;
- `dsml/recover` — parse captured block source. **Only complete calls are
  recovered** — a truncated block is dropped, never executed (mirroring ECA's
  existing "never run a call we could not fully parse" policy).

**`src/eca/llm_providers/openai_chat.clj`** — wires recovery into both paths. This
is the base for `deepseek`, `openrouter` and every OpenAI-compatible provider
(all `:api "openai-chat"`):

- non-streaming: `response-body->result` synthesises `:tool_calls` from DSML
  content so the existing mapping path handles it, and strips the markup;
- streaming: emitted text is routed through the DSML filter, and at stream-end
  any recovered calls are injected into the same `tool-calls*` accumulator the
  structured path uses — so execution, approval and rendering are unchanged.

**Surface the silent drop** — the stream-end branch that previously emitted a bare
`finish "stop"` when tool calls arrived but none were executable now first emits a
visible `:tool-call-dropped` message (rendered as a `:system` notice in
`features/chat.clj`). Covered cases: invalid/truncated JSON arguments (the
pre-existing silent case) and an unparseable/truncated DSML block.
`features/rewrite.clj` already defaults unknown message types to `nil`.

## Tests

New `test/eca/llm_providers/dsml_test.clj` — canonical block; ASCII + spaced-tags
variants; `string="false"` JSON decoding; verbatim multi-line values; the
streaming filter across a chunk split; a truncated block being dropped (not
executed); and the provider wiring (non-streaming recovery + structured calls
left untouched).

All green:

| command | result |
|---|---|
| `clojure -M:test --focus eca.llm-providers.dsml-test` | 10 tests, 25 assertions, 0 failures |
| `clojure -M:test --focus eca.llm-providers.openai-chat-test --focus eca.llm-api-test` | 71 tests, 642 assertions, 0 failures |
| `clojure -M:test --focus eca.features.chat-test` | 84 tests, 604 assertions, 0 failures |

## Known limitation

DSML values are not escaped, so a `string="true"` value that itself contains a
literal `</｜parameter>` end-tag (e.g. the model echoing a whole DSML block inside
a document) is ambiguous and is cut at the first end-tag. This is inherent to the
format and matches other DSML parsers; it is documented on `dsml/parse-block`.

## Follow-up (not in this PR)

- A dedicated render path for the dropped-call notice instead of reusing the
  `:system` text channel.
- Folding DSML recovery into providers that do not extend `openai-chat`.
