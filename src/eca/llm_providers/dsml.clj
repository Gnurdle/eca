(ns eca.llm-providers.dsml
  "Recovery for DeepSeek's native DSML (DeepSeek Markup Language) tool-call format.

  DeepSeek V4/V4.1 models — including V4.1-Flash — may serialise tool calls as
  DSML tag blocks inside the assistant *content* instead of the OpenAI structured
  `tool_calls` field:

      <｜tool_calls>
      <｜invoke name=\"read_file\">
      <｜parameter name=\"path\" string=\"true\">src/foo.clj</｜parameter>
      </｜invoke>
      </｜tool_calls>

  When the serving stack (or an OpenAI-compatible gateway in front of it) does
  not translate DSML into structured tool calls, the raw markup is delivered as
  ordinary assistant text and the call is silently lost — the turn simply looks
  like the model answered without calling anything.

  Background:
    - DeepSeek 'encoding' README — documents the DSML schema block injected when
      tools are present, including the `string=\"true|false\"` parameter marker.
    - vLLM write-up on V4.1-Flash 'spaced tags' — V4.1 writes the tags with
      whitespace inside them, which older DeepSeek detectors do not match; a
      failed match in a tool-call parser is silent by design.
    - deepseek-ai model discussion — 'model output sometimes contains DSML
      tool-call markup, but the API/client does not parse it into structured
      tool_calls; the whole assistant message is returned as plain text'.

  The delimiter is the FULL-WIDTH vertical bar ｜ (U+FF5C). Some proxies emit an
  ASCII | instead and/or insert whitespace inside the tags, so all matching here
  tolerates both.

  Pure and self-contained so it can be unit-tested directly."
  (:require
   [cheshire.core :as json]
   [clojure.string :as string]))

(set! *warn-on-reflection* true)

;; ｜ = U+FF5C (full-width vertical line, the canonical DSML delimiter); |
;; = ASCII, emitted by some proxies/UI renderings.
(def ^:private pipe "[\uFF5C|]")

(defn- opener [name]
  (re-pattern (str "(?s)<\\s*" pipe "?\\s*" name "\\s*>")))

(defn- closer [name]
  (re-pattern (str "(?s)</\\s*" pipe "?\\s*" name "\\s*>")))

(def ^:private tool-calls-open (opener "tool_calls"))
(def ^:private tool-calls-close (closer "tool_calls"))

(def ^:private block-re
  "A complete <｜tool_calls>…</｜tool_calls> block."
  (re-pattern (str "(?s)<\\s*" pipe "?\\s*tool_calls\\s*>(.*?)</\\s*" pipe "?\\s*tool_calls\\s*>")))

(def ^:private invoke-re
  "A <｜invoke name=\"…\">…</｜invoke> element; group 1 is the tool name."
  (re-pattern (str "(?s)<\\s*" pipe "?\\s*invoke\\s+name\\s*=\\s*\"([^\"]*)\"\\s*>(.*?)</\\s*" pipe "?\\s*invoke\\s*>")))

(def ^:private parameter-re
  "A <｜parameter name=\"…\" string=\"true|false\">value</｜parameter> element.
   Group 1 = parameter name, 2 = the string flag (may be absent), 3 = the value."
  (re-pattern (str "(?s)<\\s*" pipe "?\\s*parameter\\s+name\\s*=\\s*\"([^\"]*)\"\\s*"
                   "(?:string\\s*=\\s*\"(true|false)\"\\s*)?>(.*?)</\\s*" pipe "?\\s*parameter\\s*>")))

(defn- parameter-value
  "`string=\"true\"` (or the attribute being absent) means the value is a literal
  string; `string=\"false\"` means it is a JSON-encoded non-string (number,
  boolean, object, array), which we decode — falling back to the raw text if it
  does not parse."
  [string-flag ^String v]
  (if (= "false" string-flag)
    (try (json/parse-string (string/trim v))
         (catch Exception _ v))
    v))

(defn parse-block
  "Parse one DSML `tool_calls` block (or the raw inner source of one) into a
  vector of `{:full-name <string> :arguments {<string> <value>}}`.

  Tool names and argument keys are kept as raw strings so the result matches the
  shape produced by parsing a provider's structured `tool_calls`.

  Known limitation: DSML values are not escaped, so a `string=\"true\"` value that
  itself contains a literal `</｜parameter>` end-tag (e.g. the model echoing a
  complete DSML block inside a document) is ambiguous and is cut at the first
  end-tag. This mirrors the behaviour of other DSML parsers; there is no
  un-ambiguous parse without an escaping convention."
  [block]
  (->> (re-seq invoke-re (str block))
       (mapv (fn [[_ tool-name body]]
               {:full-name tool-name
                :arguments (into {}
                                 (map (fn [[_ pname string-flag pvalue]]
                                        [pname (parameter-value string-flag pvalue)]))
                                 (re-seq parameter-re body))}))
       vec))

(defn dsml?
  "True when `text` contains at least one DSML tool-call opener."
  [text]
  (boolean (and text (re-find tool-calls-open (str text)))))

(defn recover
  "Parse captured DSML source (as produced by `make-filter`) into tool calls.

  Only COMPLETE tool calls are recovered: a block whose stream was cut off is
  intentionally NOT executed, mirroring the provider's policy of never running a
  tool call it could not fully parse. Returns [] when nothing complete is found,
  so callers can treat a present-but-unparseable block as a visible failure."
  [blocks]
  (let [blocks (str blocks)
        complete (vec (mapcat parse-block (map second (re-seq block-re blocks))))]
    (cond
      (seq complete) complete
      (string/blank? blocks) []
      ;; No <｜tool_calls> wrapper matched: accept complete bare <｜invoke> elements
      ;; (some emitters omit the wrapper). Incomplete elements yield nothing.
      :else (parse-block blocks))))

(defn extract
  "Scan a complete assistant `text` for DSML tool-call blocks.

  Returns `{:tool-calls [{:full-name .. :arguments {..}} …] :text <text with the
  DSML blocks removed>}`. When there is no DSML, `:tool-calls` is empty and
  `:text` is the trimmed input."
  [text]
  (let [text (str text)]
    (if-not (dsml? text)
      {:tool-calls [] :text (string/trim text)}
      {:tool-calls (vec (mapcat parse-block (map second (re-seq block-re text))))
       :text (string/trim (string/replace text block-re ""))})))

;; --- streaming filter -------------------------------------------------------
;;
;; The chat-completions provider streams assistant text to the UI before the
;; turn ends, so DSML must be stripped as it arrives — by the time the turn is
;; complete the raw markup would already be on screen. This state machine holds
;; back a short tail so an opening tag split across chunks is still recognised,
;; and captures whole blocks instead of emitting them.

;; Longest plausible opener ("<｜tool_calls>", with slack for the spaced-tags
;; variant and surrounding whitespace). Used as the streaming hold-back tail.
(def ^:private max-opener-len 32)

(defn- first-match-span
  "Return `[start end]` of the first match of `re` in `s`, or nil."
  [^java.util.regex.Pattern re ^String s]
  (let [m (.matcher re s)]
    (when (.find m)
      [(.start m) (.end m)])))

(defn make-filter
  "A stateful, streaming-safe DSML stripper.

  Returns a map of fns over an internal state:
    :push!   (fn [chunk] -> text safe to emit now)
    :finish! (fn []     -> trailing text to emit at end-of-stream)
    :blocks  (fn []     -> captured DSML source, for `recover`)

  While inside a <｜tool_calls> block nothing is emitted; a short tail is held
  back outside blocks so an opener split across chunks is recognised."
  []
  (let [state* (atom {:in-block? false :held "" :blocks ""})]
    {:push! (fn [chunk]
              (swap! state* update :held str (str chunk))
              (let [sb (StringBuilder.)]
                (loop []
                  (let [{:keys [in-block? held]} @state*]
                    (cond
                      in-block?
                      (when-let [[_ end] (first-match-span tool-calls-close held)]
                        (swap! state* update :blocks str (subs held 0 end))
                        (swap! state* assoc :held (subs held end) :in-block? false)
                        (recur))

                      :else
                      (if-let [[start _] (first-match-span tool-calls-open held)]
                        (do (.append sb (subs held 0 start))
                            (swap! state* assoc :held (subs held start) :in-block? true)
                            (recur))
                        ;; Not in a block and no opener yet: emit everything but a
                        ;; tail, so we never split a possible opener across emits.
                        (let [n (count held)
                              emit-len (max 0 (- n max-opener-len))]
                          (.append sb (subs held 0 emit-len))
                          (swap! state* assoc :held (subs held emit-len)))))))
                (str sb)))
     :finish! (fn []
                (let [{:keys [in-block? held]} @state*]
                  (when in-block?
                    ;; Stream cut mid-block: keep it (best-effort recovery).
                    (swap! state* update :blocks str held)
                    (swap! state* assoc :held "" :in-block? false))
                  (let [trailing (:held @state*)]
                    (swap! state* assoc :held "")
                    trailing)))
     :blocks (fn [] (:blocks @state*))}))
