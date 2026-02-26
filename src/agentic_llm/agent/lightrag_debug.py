"""
lightrag_debug.py — CLI để inspect LightRAG data và test query trực tiếp.

Commands
--------
  query   : Test query trực tiếp, in raw result của LightRAG (không qua agent)
  entities: Liệt kê entities trong graph, filter theo keyword
  relations: Liệt kê relations (edges) trong graph
  chunks  : Xem raw chunks đã được index
  stats   : Thống kê tổng quan về graph

Examples
--------
  python lightrag_debug.py query "vượt đèn đỏ bị phạt bao nhiêu"
  python lightrag_debug.py query "vượt đèn đỏ" --mode hybrid --top-k 15
  python lightrag_debug.py query "vượt đèn đỏ" --mode naive   # vector only, không dùng graph
  python lightrag_debug.py entities --filter "đèn đỏ"
  python lightrag_debug.py entities --filter "nghị định"
  python lightrag_debug.py relations --filter "xử phạt"
  python lightrag_debug.py chunks --filter "vượt đèn" --limit 5
  python lightrag_debug.py stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

LIGHTRAG_WORKING_DIR = os.environ.get("LIGHTRAG_WORKING_DIR", "./lightrag_data")


# ---------------------------------------------------------------------------
# LightRAG init (reuse embedding_service)
# ---------------------------------------------------------------------------

async def _get_rag():
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete
    from embedding_service import VIETNAMESE_EMBEDDING_FUNC, get_embedding_service

    await get_embedding_service()
    rag = LightRAG(
        working_dir=LIGHTRAG_WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=VIETNAMESE_EMBEDDING_FUNC,
    )
    await rag.initialize_storages()
    return rag


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_query(args):
    """Test query trực tiếp, in kết quả raw."""
    from lightrag import QueryParam

    rag = await _get_rag()
    print(f"\n{'='*60}")
    print(f"Query  : {args.query}")
    print(f"Mode   : {args.mode}")
    print(f"Top-K  : {args.top_k}")
    print(f"{'='*60}\n")

    try:
        result = await rag.aquery(
            args.query,
            param=QueryParam(mode=args.mode, top_k=args.top_k, user_prompt = "Hãy tìm thông tin chi tiết về truy vấn. Lưu ý thông tin phải chính xác tuyệt đối và bám sát truy vấn, không được trả về câu trả lời sai lệch"),
        )
    except TypeError:
        # top_k không được support trong version này
        result = await rag.aquery(
            args.query,
            param=QueryParam(mode=args.mode),
        )

    if not result or not result.strip():
        print("⚠️  Không có kết quả.")
    else:
        print(result)

    print(f"\n{'='*60}")
    print(f"Độ dài kết quả: {len(result)} ký tự")


async def cmd_entities(args):
    """Liệt kê entities trong graph."""
    data_dir = Path(LIGHTRAG_WORKING_DIR)

    # LightRAG lưu entities trong file graph_chunk_entity_relation.json hoặc tương tự
    # Thử các file phổ biến
    candidate_files = list(data_dir.glob("*.json")) + list(data_dir.glob("**/*.json"))

    entity_file = None
    for f in candidate_files:
        if "entity" in f.name.lower() or "graph" in f.name.lower() or "kv" in f.name.lower():
            entity_file = f
            break

    if not entity_file:
        # Thử đọc từ graph storage trực tiếp
        await _print_entities_from_rag(args)
        return

    print(f"\nĐọc từ: {entity_file}")
    _print_json_entities(entity_file, args.filter, args.limit)


async def _print_entities_from_rag(args):
    """Đọc entities qua LightRAG storage API."""
    rag = await _get_rag()

    # LightRAG expose chunk_entity_relation_graph
    try:
        graph = rag.chunk_entity_relation_graph
        nodes = list(graph._graph.nodes(data=True)) if hasattr(graph, '_graph') else []
        if not nodes:
            # Thử attr khác
            nodes = []
            async for node_id, node_data in graph.nodes():
                nodes.append((node_id, node_data))
    except Exception as e:
        print(f"Không đọc được graph qua API: {e}")
        await _fallback_read_files(args, mode="entities")
        return

    keyword = args.filter.lower() if args.filter else None
    count = 0
    print(f"\n{'─'*60}")
    for node_id, data in nodes:
        if keyword and keyword not in str(node_id).lower() and keyword not in str(data).lower():
            continue
        print(f"Entity: {node_id}")
        if data:
            for k, v in data.items():
                print(f"  {k}: {str(v)[:120]}")
        print()
        count += 1
        if count >= args.limit:
            break

    print(f"{'─'*60}")
    print(f"Tổng hiển thị: {count} entities (--limit {args.limit})")


def _print_json_entities(json_file: Path, keyword: str | None, limit: int):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)

    kw = keyword.lower() if keyword else None
    count = 0
    print(f"\n{'─'*60}")

    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = enumerate(data)
    else:
        print(data)
        return

    for k, v in items:
        text = f"{k} {json.dumps(v, ensure_ascii=False)}"
        if kw and kw not in text.lower():
            continue
        print(f"Key: {k}")
        print(f"  {json.dumps(v, ensure_ascii=False)[:200]}")
        print()
        count += 1
        if count >= limit:
            break

    print(f"{'─'*60}")
    print(f"Tổng hiển thị: {count}")


async def _fallback_read_files(args, mode: str):
    """Fallback: đọc trực tiếp từ file storage của LightRAG."""
    data_dir = Path(LIGHTRAG_WORKING_DIR)
    print(f"\nFallback: quét files trong {data_dir}")

    kw = args.filter.lower() if args.filter else None

    # LightRAG thường lưu dạng .json hoặc thư mục con
    all_json = sorted(data_dir.rglob("*.json"))
    print(f"Tìm thấy {len(all_json)} file JSON\n")

    count = 0
    for jf in all_json:
        try:
            with open(jf, encoding='utf-8') as f:
                content = f.read()
            if kw and kw not in content.lower():
                continue
            print(f"📄 {jf.relative_to(data_dir)}")
            # In preview
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    for k, v in list(parsed.items())[:3]:
                        print(f"   {k}: {str(v)[:100]}")
                elif isinstance(parsed, list):
                    for item in parsed[:2]:
                        print(f"   {str(item)[:100]}")
            except Exception:
                print(f"   {content[:200]}")
            print()
            count += 1
            if count >= args.limit:
                break
        except Exception as e:
            continue

    if count == 0:
        print("Không tìm thấy kết quả phù hợp.")


async def cmd_relations(args):
    """Liệt kê relations (edges) trong graph."""
    rag = await _get_rag()
    kw = args.filter.lower() if args.filter else None

    try:
        graph = rag.chunk_entity_relation_graph
        edges = []
        async for src, tgt, data in graph.edges():
            edges.append((src, tgt, data))
    except Exception:
        await _fallback_read_files(args, mode="relations")
        return

    count = 0
    print(f"\n{'─'*60}")
    for src, tgt, data in edges:
        text = f"{src} {tgt} {data}"
        if kw and kw not in text.lower():
            continue
        rel_type = data.get("relationship", data.get("type", "?")) if data else "?"
        weight   = data.get("weight", "") if data else ""
        print(f"{src}  →[{rel_type}]→  {tgt}  (weight={weight})")
        count += 1
        if count >= args.limit:
            break

    print(f"{'─'*60}")
    print(f"Tổng hiển thị: {count} relations")


async def cmd_chunks(args):
    """Xem raw text chunks đã được index."""
    data_dir = Path(LIGHTRAG_WORKING_DIR)
    kw = args.filter.lower() if args.filter else None

    # LightRAG thường lưu chunks trong kv store
    chunk_files = (
        list(data_dir.glob("*chunk*"))
        + list(data_dir.glob("*text*"))
        + list(data_dir.glob("*.json"))
    )

    count = 0
    print(f"\n{'─'*60}")
    for cf in chunk_files:
        try:
            with open(cf, encoding='utf-8') as f:
                raw = f.read()
            if kw and kw not in raw.lower():
                continue

            parsed = json.loads(raw)
            items = parsed.items() if isinstance(parsed, dict) else enumerate(parsed)
            for k, v in items:
                text = str(v)
                if kw and kw not in text.lower():
                    continue
                print(f"File : {cf.name}")
                print(f"Key  : {k}")
                print(f"Text : {text[:500]}")
                print()
                count += 1
                if count >= args.limit:
                    return
        except Exception:
            continue

    if count == 0:
        print("Không tìm thấy chunks phù hợp.")
    print(f"{'─'*60}")
    print(f"Tổng hiển thị: {count} chunks")


async def cmd_stats(args):
    """Thống kê tổng quan về LightRAG data."""
    data_dir = Path(LIGHTRAG_WORKING_DIR)

    print(f"\n{'='*60}")
    print(f"LightRAG Working Dir: {data_dir.resolve()}")
    print(f"{'='*60}")

    if not data_dir.exists():
        print("❌ Thư mục không tồn tại!")
        return

    # File stats
    all_files = list(data_dir.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)

    print(f"\n📁 Files: {len(files)}")
    print(f"💾 Tổng dung lượng: {total_size / 1024 / 1024:.1f} MB")

    # Breakdown by extension
    from collections import Counter
    ext_count = Counter(f.suffix for f in files)
    print(f"\nPhân loại file:")
    for ext, cnt in ext_count.most_common():
        print(f"  {ext or '(no ext)'}: {cnt} files")

    # List files
    print(f"\nDanh sách files:")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.relative_to(data_dir):<50} {size_kb:>8.1f} KB")

    # Try to get graph stats
    print(f"\nGraph stats:")
    try:
        rag = await _get_rag()
        graph = rag.chunk_entity_relation_graph

        n_nodes = 0
        async for _ in graph.nodes():
            n_nodes += 1

        n_edges = 0
        async for _ in graph.edges():
            n_edges += 1

        print(f"  Entities (nodes): {n_nodes}")
        print(f"  Relations (edges): {n_edges}")
    except Exception as e:
        print(f"  Không đọc được graph stats: {e}")

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LightRAG Debug Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--working-dir", default=None,
        help="Override LIGHTRAG_WORKING_DIR"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # query
    p_query = sub.add_parser("query", help="Test query trực tiếp")
    p_query.add_argument("query", help="Query string")
    p_query.add_argument("--mode", default="hybrid",
                         choices=["naive", "local", "global", "hybrid", "mix"],
                         help="LightRAG query mode (default: hybrid)")
    p_query.add_argument("--top-k", type=int, default=10,
                         help="Top-K results (default: 10)")

    # entities
    p_ent = sub.add_parser("entities", help="Liệt kê entities trong graph")
    p_ent.add_argument("--filter", default=None, help="Lọc theo keyword")
    p_ent.add_argument("--limit", type=int, default=20, help="Số entities tối đa")

    # relations
    p_rel = sub.add_parser("relations", help="Liệt kê relations trong graph")
    p_rel.add_argument("--filter", default=None, help="Lọc theo keyword")
    p_rel.add_argument("--limit", type=int, default=30, help="Số relations tối đa")

    # chunks
    p_chunks = sub.add_parser("chunks", help="Xem raw text chunks")
    p_chunks.add_argument("--filter", default=None, help="Lọc theo keyword")
    p_chunks.add_argument("--limit", type=int, default=10, help="Số chunks tối đa")

    # stats
    sub.add_parser("stats", help="Thống kê tổng quan")

    args = parser.parse_args()

    if args.working_dir:
        global LIGHTRAG_WORKING_DIR
        LIGHTRAG_WORKING_DIR = args.working_dir

    dispatch = {
        "query":     cmd_query,
        "entities":  cmd_entities,
        "relations": cmd_relations,
        "chunks":    cmd_chunks,
        "stats":     cmd_stats,
    }

    asyncio.run(dispatch[args.command](args))


if __name__ == "__main__":
    main()