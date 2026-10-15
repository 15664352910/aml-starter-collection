from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m src.cli` / `python src/cli.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory import MemorySystem, load_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Session memory: add (raw session + embedding) & hybrid search",
    )
    p.add_argument(
        "--store-dir",
        default=None,
        help="Memory store directory (default: MEMORY_STORE_DIR or ./store)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Ingest sessions from a data path")
    add.add_argument(
        "--data",
        required=True,
        help="Data path (file or directory), e.g. data or data/longmemeval_refined/longmemeval_refined.json",
    )
    add.add_argument(
        "--bucket-id",
        default=None,
        help="Override bucket id for all samples (default: sample/conversation id)",
    )
    add.add_argument("--no-embed", action="store_true", help="Only store text (phase-1)")
    add.add_argument(
        "--force",
        action="store_true",
        help="Rewrite existing memories (default: idempotent skip)",
    )

    search = sub.add_parser("search", help="Hybrid search within a bucket")
    search.add_argument("--bucket-id", required=True, help="Conversation / user bucket")
    search.add_argument("--query", required=True, help="Query text")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fusion weight for BM25 (default: HYBRID_ALPHA or 0.5)",
    )
    search.add_argument(
        "--json",
        action="store_true",
        help="Print full scored hits as JSON",
    )

    upsert = sub.add_parser("upsert", help="Upsert a single memory")
    upsert.add_argument("--id", required=True)
    upsert.add_argument("--bucket-id", required=True)
    upsert.add_argument("--text", required=True)
    upsert.add_argument("--metadata", default="{}", help="JSON metadata object")
    upsert.add_argument("--no-embed", action="store_true")

    sub.add_parser("list-buckets", help="List stored bucket ids")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    mem = MemorySystem(settings=settings, store_dir=args.store_dir)

    if args.command == "add":
        report = mem.add_from_data(
            args.data,
            bucket_id=args.bucket_id,
            embed=not args.no_embed,
            skip_if_exists=not args.force,
        )
        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        hits = mem.search(
            bucket_id=args.bucket_id,
            query=args.query,
            top_k=args.top_k,
            alpha=args.alpha,
        )
        if args.json:
            print(json.dumps(mem.to_dict_hits(hits), ensure_ascii=False, indent=2))
        else:
            for i, h in enumerate(hits, 1):
                print(f"===== #{i} score={h.score:.4f} id={h.id} =====")
                print(h.text)
                print()
        return 0

    if args.command == "upsert":
        meta = json.loads(args.metadata)
        result = mem.upsert(
            id=args.id,
            text=args.text,
            metadata=meta,
            bucket_id=args.bucket_id,
            embed=not args.no_embed,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return 0

    if args.command == "list-buckets":
        for b in mem.store.list_buckets():
            n = len(mem.store.list_bucket(b))
            print(f"{b}\t{n}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
