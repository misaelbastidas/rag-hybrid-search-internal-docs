import json
from pathlib import Path

from anthropic import Anthropic

from generation.rag_pipeline import RAGPipeline

from .base import EvalResult, GoldenExample
from .metrics import judge_appropriate_decline, judge_correctness, judge_faithfulness, score_retrieval_relevance

WEIGHTS = {"correctness": 0.4, "faithfulness": 0.25, "retrieval_relevance": 0.2, "citation_accuracy": 0.15}


def load_golden_dataset(path: str | Path) -> list[GoldenExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenExample(**item) for item in data]


def _composite(correctness: float, faithfulness: float, retrieval_relevance: float, citation_accuracy: float) -> float:
    return (
        WEIGHTS["correctness"] * correctness
        + WEIGHTS["faithfulness"] * faithfulness
        + WEIGHTS["retrieval_relevance"] * retrieval_relevance
        + WEIGHTS["citation_accuracy"] * citation_accuracy
    )


def evaluate_example(example: GoldenExample, pipeline: RAGPipeline, judge_client: Anthropic) -> EvalResult:
    result = pipeline.ask(example.question)
    retrieved_paths = [s.source_path for s in result.sources]
    retrieval_relevance = score_retrieval_relevance(retrieved_paths, example.source_files)

    if example.answer_type == "no_answer":
        if not result.answered:
            correctly_declined, notes = True, "correctly declined (retrieval confidence gate)"
        else:
            # Retrieval found a plausible-but-wrong source (high confidence), so generation ran.
            # Still correct if the answer honestly admits the specific fact isn't in the source.
            correctly_declined, reason = judge_appropriate_decline(judge_client, example.question, result.answer)
            notes = f"answered anyway; honest decline={correctly_declined}: {reason}"

        score = 1.0 if correctly_declined else 0.0
        return EvalResult(
            id=example.id,
            question=example.question,
            answer_type=example.answer_type,
            generated_answer=result.answer,
            answered=result.answered,
            correctness=score,
            faithfulness=score,
            retrieval_relevance=retrieval_relevance,
            citation_accuracy=score,
            composite=score,
            notes=notes,
        )

    if not result.answered:
        return EvalResult(
            id=example.id,
            question=example.question,
            answer_type=example.answer_type,
            generated_answer=None,
            answered=False,
            correctness=0.0,
            faithfulness=0.0,
            retrieval_relevance=retrieval_relevance,
            citation_accuracy=0.0,
            composite=0.0,
            notes="WRONGLY DECLINED a question that should have been answerable",
        )

    assert result.answer is not None
    correctness, correctness_reason = judge_correctness(
        judge_client, example.question, example.expected_answer, result.answer
    )
    faithfulness, faithfulness_reason = judge_faithfulness(
        judge_client, result.answer, [s.text for s in result.sources]
    )
    citation_accuracy = result.confidence.citation_coverage if result.confidence else 0.0

    composite = _composite(correctness, faithfulness, retrieval_relevance, citation_accuracy)

    return EvalResult(
        id=example.id,
        question=example.question,
        answer_type=example.answer_type,
        generated_answer=result.answer,
        answered=True,
        correctness=correctness,
        faithfulness=faithfulness,
        retrieval_relevance=retrieval_relevance,
        citation_accuracy=citation_accuracy,
        composite=composite,
        notes=f"correctness: {correctness_reason} | faithfulness: {faithfulness_reason}",
    )


def run_eval(
    golden_examples: list[GoldenExample],
    pipeline: RAGPipeline | None = None,
    judge_client: Anthropic | None = None,
) -> list[EvalResult]:
    pipeline = pipeline or RAGPipeline()
    judge_client = judge_client or Anthropic()
    return [evaluate_example(ex, pipeline, judge_client) for ex in golden_examples]


def summarize(results: list[EvalResult]) -> dict:
    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_type: dict[str, list[EvalResult]] = {}
    for r in results:
        by_type.setdefault(r.answer_type, []).append(r)

    summary = {
        "overall": {
            "n": len(results),
            "correctness": avg([r.correctness for r in results]),
            "faithfulness": avg([r.faithfulness for r in results]),
            "retrieval_relevance": avg([r.retrieval_relevance for r in results]),
            "citation_accuracy": avg([r.citation_accuracy for r in results]),
            "composite": avg([r.composite for r in results]),
        }
    }
    for answer_type, group in by_type.items():
        summary[answer_type] = {
            "n": len(group),
            "correctness": avg([r.correctness for r in group]),
            "faithfulness": avg([r.faithfulness for r in group]),
            "retrieval_relevance": avg([r.retrieval_relevance for r in group]),
            "citation_accuracy": avg([r.citation_accuracy for r in group]),
            "composite": avg([r.composite for r in group]),
        }
    return summary
