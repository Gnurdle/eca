(ns eca.llm-providers.dsml-test
  "Tests for DeepSeek DSML tool-call recovery (see eca.llm-providers.dsml).

  DeepSeek V4/V4.1 models (incl. V4.1-Flash) may emit tool calls as DSML markup
  in assistant content instead of the structured :tool_calls field. These tests
  pin the parser, the streaming filter, and the provider wiring."
  (:require
   [clojure.string :as string]
   [clojure.test :refer [deftest is testing]]
   [eca.llm-providers.dsml :as dsml]
   [eca.llm-providers.openai-chat :as openai-chat]))

(def ^:private canonical
  (str "Here you go.\n"
       "<｜tool_calls>\n"
       "<｜invoke name=\"grog-mcp__assoc_store\">\n"
       "<｜parameter name=\"key\" string=\"true\">k</｜parameter>\n"
       "<｜parameter name=\"value\" string=\"true\">v</｜parameter>\n"
       "</｜invoke>\n"
       "</｜tool_calls>"))

(deftest extract-canonical-test
  (testing "a canonical DSML block becomes a tool call and leaves the prose"
    (let [{:keys [tool-calls text]} (dsml/extract canonical)]
      (is (= 1 (count tool-calls)))
      (is (= "grog-mcp__assoc_store" (:full-name (first tool-calls))))
      (is (= {"key" "k" "value" "v"} (:arguments (first tool-calls))))
      (is (= "Here you go." text)))))

(deftest extract-plain-text-untouched-test
  (testing "text with no DSML returns no calls and the text unchanged"
    (let [{:keys [tool-calls text]} (dsml/extract "just an answer, with a | pipe")]
      (is (empty? tool-calls))
      (is (= "just an answer, with a | pipe" text)))))

(deftest extract-spaced-ascii-tags-test
  (testing "an ASCII pipe and the V4.1 spaced-tags variant are recognised"
    (let [s (str "< | tool_calls >\n"
                 "< | invoke name=\"read_file\">\n"
                 "< | parameter name=\"path\" string=\"true\">src/foo.clj</ | parameter >\n"
                 "</ | invoke >\n"
                 "</ | tool_calls >")
          {:keys [tool-calls]} (dsml/extract s)]
      (is (= ["read_file"] (mapv :full-name tool-calls)))
      (is (= {"path" "src/foo.clj"} (:arguments (first tool-calls)))))))

(deftest extract-decodes-non-string-params-test
  (testing "string=\"false\" parameters are JSON-decoded"
    (let [s (str "<｜tool_calls>"
                 "<｜invoke name=\"a\"><｜parameter name=\"n\" string=\"false\">42</｜parameter></｜invoke>"
                 "<｜invoke name=\"b\"><｜parameter name=\"ok\" string=\"false\">true</｜parameter></｜invoke>"
                 "</｜tool_calls>")
          {:keys [tool-calls]} (dsml/extract s)]
      (is (= ["a" "b"] (mapv :full-name tool-calls)))
      (is (= 42 (get-in tool-calls [0 :arguments "n"])))
      (is (true? (get-in tool-calls [1 :arguments "ok"]))))))

(deftest extract-preserves-multiline-values-test
  (testing "a multi-line string value (with pipes) is preserved verbatim"
    (let [v "a|b\nc|d"
          s (str "<｜tool_calls><｜invoke name=\"write\">"
                 "<｜parameter name=\"content\" string=\"true\">" v "</｜parameter>"
                 "</｜invoke></｜tool_calls>")
          {:keys [tool-calls]} (dsml/extract s)]
      (is (= v (get-in tool-calls [0 :arguments "content"]))))))

(deftest streaming-filter-reassembles-split-block-test
  (testing "a DSML block split across chunks is suppressed from text and recovered"
    (let [f (dsml/make-filter)
          chunks ["I will call the tool " "<｜tool_" "calls><｜invoke name=\"read_file\">"
                  "<｜parameter name=\"path\" string=\"true\">a</｜parameter>"
                  "</｜invoke></｜tool_calls> and then " "finish."]
          emitted (apply str (map (fn [c] ((:push! f) c)) chunks))
          calls (dsml/recover ((:blocks f)))]
      (is (= "I will call the tool  and then finish." (str emitted ((:finish! f)))))
      (is (= ["read_file"] (mapv :full-name calls)))
      (is (= {"path" "a"} (:arguments (first calls)))))))

(deftest streaming-filter-drops-truncated-block-test
  (testing "a truncated block is suppressed and NOT executed"
    (let [f (dsml/make-filter)
          emitted (str ((:push! f) "<｜tool_calls><｜invoke name=\"x\"><｜parameter name=\"p\" string=\"true\">q</｜parameter>")
                       ((:finish! f)))]
      (is (= "" emitted))
      (is (empty? (dsml/recover ((:blocks f)))))
      (is (not (string/blank? ((:blocks f))))
          "the block is retained so the caller can flag the drop"))))

(deftest streaming-filter-plain-text-test
  (testing "text with no DSML streams through unchanged"
    (let [f (dsml/make-filter)]
      (is (= "hello world" (str ((:push! f) "hello ") ((:push! f) "world") ((:finish! f)))))
      (is (= "" ((:blocks f)))))))

(deftest response-body-recovers-dsml-test
  (testing "non-streaming: DSML content is recovered into :tool_calls and stripped from output"
    (let [body {:choices [{:message {:role "assistant" :content canonical}}] :usage {}}
          result (#'openai-chat/response-body->result body (fn [& _] nil))]
      (is (= ["grog-mcp__assoc_store"] (mapv :full-name (:tools-to-call result))))
      (is (= {"key" "k" "value" "v"} (:arguments (first (:tools-to-call result)))))
      (is (= "Here you go." (:output-text result))))))

(deftest response-body-leaves-structured-untouched-test
  (testing "a structured tool_calls response is unaffected by the DSML path"
    (let [body {:choices [{:message {:role "assistant" :content "hi"
                                     :tool_calls [{:id "x" :type "function"
                                                   :function {:name "shell_command"
                                                              :arguments "{\"command\":\"ls\"}"}}]}}]
                :usage {}}
          result (#'openai-chat/response-body->result body (fn [& _] nil))]
      (is (= ["shell_command"] (mapv :full-name (:tools-to-call result))))
      (is (= "hi" (:output-text result))))))
