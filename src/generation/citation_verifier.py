import json
import re
from dataclasses import dataclass

from anthropic import Anthropic

from .base import GeneratedAnswer

CITATION_INDEX_RE = re.compile(r"\[(\d+)\]")

VERIFY_MODEL = "claude-haiku-4-5-20251001"

VERIFY_SYSTEM_PROMPT = """You verify whether a claim is actually supported by a source text.
Respond with ONLY a JSON object: {"supported": true or false, "reason": "short explanation"}.
"supported" must be false if the source does not clearly state the claim, even if it's topically related."""


@dataclass
class CitationCheck:
    claim: str
    citation_index: int
    supported: bool
    reason: str


def extract_citations(answer_text: str) -> list[tuple[str, int]]:
    """Returns (paragraph, citation_number) pairs — one per citation found, using the whole
    paragraph as the claim so an intro line's citation still covers a following markdown list."""
    results = []
    for block in re.split(r"\n\s*\n", answer_text):
        block = block.strip()
        if not block:
            continue
        for index in {int(m.group(1)) for m in CITATION_INDEX_RE.finditer(block)}:
            results.append((block, index))
    return results


class CitationVerifier:
    def __init__(self, client: Anthropic | None = None, model: str = VERIFY_MODEL):
        self.client = client or Anthropic()
        self.model = model

    def verify(self, generated: GeneratedAnswer) -> list[CitationCheck]:
        claims = extract_citations(generated.answer)
        checks = []
        for sentence, citation_index in claims:
            if citation_index < 1 or citation_index > len(generated.sources):
                checks.append(CitationCheck(sentence, citation_index, False, "citation index out of range"))
                continue

            source_text = generated.sources[citation_index - 1].text
            verdict = self._judge(sentence, source_text)
            checks.append(CitationCheck(sentence, citation_index, verdict["supported"], verdict["reason"]))
        return checks

    def _judge(self, claim: str, source_text: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=VERIFY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Source text:\n{source_text}\n\nClaim: {claim}"}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return {"supported": False, "reason": f"verifier returned non-JSON output: {text!r}"}
        return json.loads(json_match.group(0))
