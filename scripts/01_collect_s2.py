"""
01_collect_s2.py
================
Semantic Scholar API로 학술 논문 제목/초록을 수집하는 스크립트.
- 분야(Law)에서 아동 관련 용어별로 논문을 검색
- API 키 없이도 동작 (속도 제한: ~1req/sec)
- 키가 있으면 환경변수 S2_API_KEY로 전달

사용법:
    python scripts/01_collect_s2.py
    python scripts/01_collect_s2.py --field Psychology --max-per-term 300
    S2_API_KEY=your_key python scripts/01_collect_s2.py
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── 설정 ──────────────────────────────────────────────

CHILD_TERMS = [
    "child",
    "children",
    "childhood",
    "youth",
    "young people",
    "adolescent",
    "adolescence",
    "juvenile",
    "minor",
    "minors",
    "teenage",
    "teenager",
    "young adult",
]

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,fieldsOfStudy,journal,citationCount,externalIds,url"

# API 키가 없으면 ~1 req/sec, 있으면 ~10 req/sec
API_KEY = os.environ.get("S2_API_KEY", None)
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}
SLEEP = 1.1 if not API_KEY else 0.15  # rate limit 존중


# ── 함수 ──────────────────────────────────────────────

def search_semantic_scholar(query: str, field: str, offset: int = 0, limit: int = 100) -> dict:
    """Semantic Scholar Paper Search API 호출."""
    params = {
        "query": query,
        "fieldsOfStudy": field,
        "offset": offset,
        "limit": min(limit, 100),  # API 최대 100
        "fields": S2_FIELDS,
    }
    resp = requests.get(S2_SEARCH_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_papers(field: str, max_per_term: int, output_dir: Path) -> pd.DataFrame:
    """모든 검색어에 대해 논문을 수집하고 DataFrame으로 반환."""
    all_papers = []

    for term in CHILD_TERMS:
        offset = 0
        term_papers = []
        pbar = tqdm(total=max_per_term, desc=f"'{term}'", unit="papers", leave=True)

        while offset < max_per_term:
            try:
                result = search_semantic_scholar(term, field, offset=offset)
                papers = result.get("data", [])
                total_available = result.get("total", 0)

                if not papers:
                    break

                for p in papers:
                    p["search_term"] = term

                term_papers.extend(papers)
                pbar.update(len(papers))
                offset += 100
                time.sleep(SLEEP)

                # 더 이상 없으면 중단
                if offset >= total_available:
                    break

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"\n  ⚠ Rate limited. Waiting 30s...")
                    time.sleep(30)
                    continue
                else:
                    print(f"\n  ✗ HTTP {e.response.status_code} at offset {offset}")
                    break
            except Exception as e:
                print(f"\n  ✗ Error: {e}")
                time.sleep(5)
                break

        pbar.close()
        all_papers.extend(term_papers)
        print(f"  → '{term}': {len(term_papers)} papers (available: {result.get('total', '?')})")

    print(f"\n{'='*50}")
    print(f"Total raw papers: {len(all_papers)}")

    # ── DataFrame 변환 ──
    df = pd.DataFrame(all_papers)
    if df.empty:
        print("⚠ No papers collected!")
        return df

    # ── 중복 제거 (같은 논문이 여러 검색어에 걸릴 수 있음) ──
    # 매칭된 검색어 목록은 보존
    df_terms = df.groupby("paperId")["search_term"].apply(list).reset_index()
    df_terms.columns = ["paperId", "matched_terms"]

    df_unique = df.drop_duplicates(subset="paperId").drop(columns=["search_term"])
    df_merged = df_unique.merge(df_terms, on="paperId")

    # ── 초록 없는 논문 분리 ──
    has_abstract = df_merged["abstract"].notna() & (df_merged["abstract"].str.strip() != "")
    df_with = df_merged[has_abstract].copy()
    df_without = df_merged[~has_abstract].copy()

    print(f"Unique papers: {len(df_merged)}")
    print(f"  With abstract: {len(df_with)} ({len(df_with)/len(df_merged)*100:.1f}%)")
    print(f"  Without abstract: {len(df_without)} ({len(df_without)/len(df_merged)*100:.1f}%)")

    return df_with


def print_summary(df: pd.DataFrame):
    """수집 결과 요약 출력."""
    print(f"\n{'='*50}")
    print(f"📊 SUMMARY")
    print(f"{'='*50}")
    print(f"Total papers (with abstract): {len(df)}")

    if df.empty:
        return

    # 연도 분포
    print(f"\n📅 Year range: {df['year'].min()} – {df['year'].max()}")
    print(f"   Median year: {df['year'].median():.0f}")

    # 검색어별 빈도
    print(f"\n🏷 Papers per search term:")
    term_counts = Counter()
    for terms in df["matched_terms"]:
        for t in terms:
            term_counts[t] += 1
    for term, count in term_counts.most_common():
        bar = "█" * (count // 20) + "░" * max(0, 20 - count // 20)
        print(f"   {term:18s} {count:5d}  {bar}")

    # 초록 길이
    abs_len = df["abstract"].str.len()
    print(f"\n📝 Abstract length (chars):")
    print(f"   Mean: {abs_len.mean():.0f}  |  Median: {abs_len.median():.0f}  |  "
          f"Min: {abs_len.min():.0f}  |  Max: {abs_len.max():.0f}")

    # 샘플
    print(f"\n📄 Random sample (3 papers):")
    for _, row in df.sample(min(3, len(df))).iterrows():
        print(f"\n   [{row.get('year', '?')}] {row['title']}")
        print(f"   Terms: {row['matched_terms']}")
        abs_preview = row["abstract"][:150].replace("\n", " ")
        print(f"   Abstract: {abs_preview}...")


# ── 메인 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect papers from Semantic Scholar")
    parser.add_argument("--field", default="Law", help="Field of study (default: Law)")
    parser.add_argument("--max-per-term", type=int, default=200,
                        help="Max papers per search term (default: 200, pilot size)")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*50}")
    print(f"📚 Semantic Scholar Paper Collection")
    print(f"{'='*50}")
    print(f"Field: {args.field}")
    print(f"Max per term: {args.max_per_term}")
    print(f"Search terms: {len(CHILD_TERMS)}")
    print(f"API key: {'✓ (fast mode)' if API_KEY else '✗ (slow mode, ~1 req/sec)'}")
    print(f"{'='*50}\n")

    # ── 수집 ──
    df = collect_papers(args.field, args.max_per_term, output_dir)

    if df.empty:
        print("No data collected. Exiting.")
        return

    # ── 요약 ──
    print_summary(df)

    # ── 저장 ──
    field_slug = args.field.lower().replace(" ", "_")
    csv_path = output_dir / f"pilot_{field_slug}_child.csv"
    pkl_path = output_dir / f"pilot_{field_slug}_child.pkl"
    json_path = output_dir / f"pilot_{field_slug}_meta.json"

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    # 메타데이터 저장 (재현성)
    meta = {
        "field": args.field,
        "max_per_term": args.max_per_term,
        "search_terms": CHILD_TERMS,
        "total_collected": len(df),
        "collection_date": pd.Timestamp.now().isoformat(),
        "api_key_used": bool(API_KEY),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved:")
    print(f"   CSV:  {csv_path}")
    print(f"   PKL:  {pkl_path}")
    print(f"   META: {json_path}")
    print(f"\n✅ Done! Next step: python scripts/02_tfidf_quick.py")


if __name__ == "__main__":
    main()
