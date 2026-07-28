import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from eval.runner import load_golden_dataset, run_eval, summarize
from generation.rag_pipeline import RAGPipeline
from ingestion.vector_store import VectorStore
from ingestion.sparse_store import BM25Store
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Run the golden Q&A eval suite")
    parser.add_argument("dataset", nargs="?", default="data/eval/golden_dataset.json")
    parser.add_argument("--verbose", action="store_true", help="print per-question results")
    parser.add_argument("--persist-dir", default="data/processed/chroma", dest="persist_dir")
    parser.add_argument("--bm25-path", default="data/processed/bm25_index.pkl", dest="bm25_path")
    parser.add_argument("--save-summary", default=None, dest="save_summary", help="write summary JSON to this path")
    args = parser.parse_args()

    vector_store = VectorStore(persist_dir=args.persist_dir)
    bm25_store = BM25Store(index_path=args.bm25_path)
    retriever = HybridRetriever(
        dense=DenseRetriever(vector_store=vector_store),
        sparse=SparseRetriever(bm25_store=bm25_store, vector_store=vector_store),
    )
    pipeline = RAGPipeline(retriever=retriever)

    examples = load_golden_dataset(args.dataset)
    print(f"Running {len(examples)} golden examples against {args.persist_dir}...")
    results = run_eval(examples, pipeline=pipeline)

    if args.verbose:
        for r in results:
            print(f"\n[{r.id}] ({r.answer_type}) {r.question}")
            print(f"  composite={r.composite:.2f} correctness={r.correctness:.2f} "
                  f"faithfulness={r.faithfulness:.2f} retrieval={r.retrieval_relevance:.2f} "
                  f"citation={r.citation_accuracy:.2f}")
            if r.composite < 0.5:
                print(f"  ANSWER: {r.generated_answer}")
                print(f"  NOTES: {r.notes}")

    summary = summarize(results)
    print("\n=== SUMMARY ===")
    for key, stats in summary.items():
        print(
            f"{key:12s} n={stats['n']:3d}  composite={stats['composite']:.3f}  "
            f"correctness={stats['correctness']:.3f}  faithfulness={stats['faithfulness']:.3f}  "
            f"retrieval={stats['retrieval_relevance']:.3f}  citation={stats['citation_accuracy']:.3f}"
        )

    if args.save_summary:
        Path(args.save_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
