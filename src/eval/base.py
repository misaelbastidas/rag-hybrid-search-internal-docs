from dataclasses import dataclass, field


@dataclass
class GoldenExample:
    id: str
    question: str
    expected_answer: str
    answer_type: str  # "lookup" | "multi_hop" | "no_answer" | "ambiguous"
    source_files: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalResult:
    id: str
    question: str
    answer_type: str
    generated_answer: str | None
    answered: bool
    correctness: float
    faithfulness: float
    retrieval_relevance: float
    citation_accuracy: float
    composite: float
    notes: str
