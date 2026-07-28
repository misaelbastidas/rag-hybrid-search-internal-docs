import json
import re

from anthropic import Anthropic

JUDGE_MODEL = "claude-haiku-4-5-20251001"

CORRECTNESS_SYSTEM_PROMPT = """You judge whether a generated answer conveys the same factual content as a golden reference answer.
Respond with ONLY a JSON object: {"correctness": 0.0 to 1.0, "reason": "short explanation"}.
Score 1.0 only if all key facts match. Score partial credit for partially correct or incomplete answers.
Wording/language/style differences don't matter, only factual content."""

FAITHFULNESS_SYSTEM_PROMPT = """You judge whether a generated answer contains ONLY claims that are supported by the given source context.
Respond with ONLY a JSON object: {"faithfulness": 0.0 to 1.0, "reason": "short explanation"}.
Score 1.0 if every factual claim in the answer is backed by the context. Score lower for each unsupported or fabricated claim."""

DECLINE_SYSTEM_PROMPT = """You judge whether an answer honestly admits the specific requested information is not available/not specified in the source material, rather than fabricating a confident but unsupported specific answer.
Respond with ONLY a JSON object: {"appropriately_declined": true or false, "reason": "short explanation"}.
true = the answer says the specific fact isn't stated/covered (even if it discusses related context). false = the answer states a specific fact as if it were confirmed by the source, when it was not."""


def _judge(client: Anthropic, system_prompt: str, user_content: str, field_name: str) -> tuple[float, str]:
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=250,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return 0.0, f"judge returned non-JSON output: {text!r}"
    parsed = json.loads(json_match.group(0))
    return float(parsed[field_name]), parsed.get("reason", "")


def judge_correctness(
    client: Anthropic, question: str, expected_answer: str, generated_answer: str
) -> tuple[float, str]:
    user_content = (
        f"Question: {question}\n\nGolden reference answer: {expected_answer}\n\nGenerated answer: {generated_answer}"
    )
    return _judge(client, CORRECTNESS_SYSTEM_PROMPT, user_content, "correctness")


def judge_faithfulness(client: Anthropic, generated_answer: str, source_texts: list[str]) -> tuple[float, str]:
    context = "\n\n---\n\n".join(source_texts)
    user_content = f"Context:\n{context}\n\nAnswer to check: {generated_answer}"
    return _judge(client, FAITHFULNESS_SYSTEM_PROMPT, user_content, "faithfulness")


def judge_appropriate_decline(client: Anthropic, question: str, generated_answer: str) -> tuple[bool, str]:
    user_content = f"Question: {question}\n\nAnswer: {generated_answer}"
    score, reason = _judge(client, DECLINE_SYSTEM_PROMPT, user_content, "appropriately_declined")
    return bool(score), reason


def score_retrieval_relevance(retrieved_source_paths: list[str], expected_source_files: list[str]) -> float:
    """Fraction of expected source files that appear among the retrieved sources. No LLM call, pure overlap."""
    if not expected_source_files:
        return 1.0 if not retrieved_source_paths else 0.0
    retrieved_names = {path.split("\\")[-1].split("/")[-1] for path in retrieved_source_paths}
    expected = set(expected_source_files)
    found = expected & retrieved_names
    return len(found) / len(expected)
