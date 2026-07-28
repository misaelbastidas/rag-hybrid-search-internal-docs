import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from ingestion.pipeline import ingest_directory
from ingestion.vector_store import VectorStore
from ingestion.sparse_store import BM25Store
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever
from generation.rag_pipeline import RAGPipeline


def cmd_ingest(args):
    chunker_kwargs = {}
    if args.chunk_size is not None:
        chunker_kwargs["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        chunker_kwargs["chunk_overlap"] = args.chunk_overlap

    result = ingest_directory(
        args.dir,
        strategy=args.strategy,
        chunker_kwargs=chunker_kwargs or None,
        persist_dir=args.persist_dir,
        bm25_path=args.bm25_path,
    )
    print(result)


def cmd_ask(args):
    vector_store = VectorStore(persist_dir=args.persist_dir)
    bm25_store = BM25Store(index_path=args.bm25_path)
    retriever = HybridRetriever(
        dense=DenseRetriever(vector_store=vector_store),
        sparse=SparseRetriever(bm25_store=bm25_store, vector_store=vector_store),
    )
    result = RAGPipeline(retriever=retriever).ask(args.question, candidate_k=args.candidate_k, top_k=args.top_k)

    if not result.answered:
        print(result.message)
        return

    print("Sources:")
    for i, c in enumerate(result.sources, 1):
        print(f"  [{i}] {c.source_path} - {c.section_heading or f'page {c.page_num}'}")
    print()

    print("Answer:")
    print(result.answer)
    print()

    conf = result.confidence
    print(
        f"Confidence: composite={conf.composite:.2f} "
        f"(retrieval={conf.retrieval_confidence:.2f}, "
        f"citation_coverage={conf.citation_coverage:.2f}, "
        f"completeness={conf.completeness:.2f})"
    )


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline CLI")
    subparsers = parser.add_subparsers(required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from a directory")
    ingest_parser.add_argument("dir", help="Directory to ingest (e.g. data/raw)")
    ingest_parser.add_argument("--strategy", default="recursive", choices=["fixed_size", "recursive", "semantic"])
    ingest_parser.add_argument("--chunk-size", type=int, default=None, dest="chunk_size")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=None, dest="chunk_overlap")
    ingest_parser.add_argument("--persist-dir", default="data/processed/chroma", dest="persist_dir")
    ingest_parser.add_argument("--bm25-path", default="data/processed/bm25_index.pkl", dest="bm25_path")
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against ingested documents")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--candidate-k", type=int, default=20, dest="candidate_k")
    ask_parser.add_argument("--top-k", type=int, default=8, dest="top_k")
    ask_parser.add_argument("--persist-dir", default="data/processed/chroma", dest="persist_dir")
    ask_parser.add_argument("--bm25-path", default="data/processed/bm25_index.pkl", dest="bm25_path")
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
