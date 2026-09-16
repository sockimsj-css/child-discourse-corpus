"""
02_tfidf_quick.py  (v2)
=======================
수집된 코퍼스에 대해 TF-IDF 분석 수행.
- 학술 논문 특화 불용어 제거
- 그룹별 상위 특징어 + Distinctive Terms 를 하나의 CSV에 시트(탭) 구분으로 저장
- 분야별로 돌릴 수 있게 --input 인자 지원

사용법:
    python scripts/02_tfidf_quick.py
    python scripts/02_tfidf_quick.py --input data/raw/pilot_psychology_child.pkl
    python scripts/02_tfidf_quick.py --input data/raw/pilot_law_child.pkl --top-n 30
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 설정 ──────────────────────────────────────────────

TERM_GROUPS = {
    "child/children": ["child", "children", "childhood"],
    "adolescent": ["adolescent", "adolescence"],
    "youth": ["youth", "young people", "young adult"],
    "juvenile": ["juvenile"],
    "minor": ["minor", "minors"],
    "teenage": ["teenage", "teenager"],
}

# 학술 논문 특화 불용어 — 모든 분야에서 흔히 나오지만 담론적 의미가 없는 단어
ACADEMIC_STOPWORDS = [
    # 연구 방법/구조
    "study", "studies", "research", "results", "findings", "paper", "article",
    "analysis", "data", "participants", "sample", "method", "methods",
    "approach", "review", "literature", "examined", "explored", "discussed",
    "investigated", "aimed", "objective", "purpose", "conclusion", "conclusions",
    "implications", "limitation", "limitations", "future", "conducted",
    "design", "procedure", "measures", "measure", "instrument",
    # 통계/수치 표현
    "significant", "significantly", "associated", "association", "relationship",
    "effect", "effects", "outcome", "outcomes", "factor", "factors",
    "variable", "variables", "prevalence", "rate", "rates", "percent",
    "ci", "95", "odds", "ratio", "mean", "average", "sd", "se",
    "p", "n", "total", "number", "higher", "lower", "increased", "decreased",
    "compared", "comparison", "difference", "differences", "similar",
    "positive", "negative", "correlation", "regression", "model",
    "statistically", "multivariate", "logistic", "adjusted", "unadjusted",
    # 범용 명사/동사
    "cases", "case", "people", "group", "groups", "level", "levels",
    "type", "types", "form", "forms", "use", "used", "using", "based",
    "include", "included", "including", "provides", "provide", "provided",
    "suggest", "suggests", "suggested", "indicate", "indicates", "indicated",
    "report", "reported", "reports", "found", "showed", "shown", "show",
    "impact", "role", "need", "needs", "important", "particular",
    "particularly", "specific", "specifically", "current", "present",
    "recent", "new", "different", "various", "common", "likely",
    "potential", "possible", "related", "relevant", "key", "major",
    "general", "overall", "given", "certain", "well", "also",
    "however", "although", "moreover", "furthermore", "therefore",
    "thus", "addition", "example", "context", "process", "issue", "issues",
    # 지역/국가 (분야 무관하게 등장)
    "united", "states", "united states", "country", "countries", "national",
    "international", "global",
    # 시간 표현
    "year", "years", "period", "time", "age", "ages",  # 'age'는 모든 그룹에 나오므로
    # 기타 고빈도 무의미어
    "et", "al", "doi", "https", "http", "vol", "pp", "isbn",
    "university", "journal", "press", "published", "copyright",
]


def assign_group(matched_terms: list) -> str:
    """논문의 매칭된 검색어를 기준으로 대표 그룹 배정."""
    for group_name, group_terms in TERM_GROUPS.items():
        if any(t in matched_terms for t in group_terms):
            return group_name
    return "other"


def run_tfidf(df: pd.DataFrame, output_dir: Path, top_n: int = 25) -> tuple:
    """그룹별 TF-IDF + Distinctive Terms 분석."""

    df = df.copy()
    df["term_group"] = df["matched_terms"].apply(assign_group)

    # 그룹별 문서 수
    group_counts = df["term_group"].value_counts()
    print(f"\n📊 Group sizes:")
    for group, count in group_counts.items():
        print(f"   {group:20s} {count:5d} papers")

    valid_groups = group_counts[group_counts >= 10].index.tolist()
    df_valid = df[df["term_group"].isin(valid_groups)]

    # 그룹별 초록 합침
    group_texts = {}
    for group in valid_groups:
        texts = df_valid[df_valid["term_group"] == group]["abstract"].tolist()
        group_texts[group] = " ".join(texts)

    if len(group_texts) < 2:
        print("⚠ Need at least 2 groups. Exiting.")
        return None, None

    # 불용어 = sklearn 기본 영어 + 학술 특화
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    all_stopwords = list(ENGLISH_STOP_WORDS) + ACADEMIC_STOPWORDS

    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words=all_stopwords,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
    )
    tfidf_matrix = vectorizer.fit_transform(group_texts.values())
    features = vectorizer.get_feature_names_out()

    group_list = list(group_texts.keys())

    # ── Part 1: 그룹별 상위 특징어 ──
    tfidf_rows = []
    print(f"\n{'='*60}")
    print(f"🏷  TF-IDF Top {top_n} Terms per Group (stopwords filtered)")
    print(f"{'='*60}")

    for i, group in enumerate(group_list):
        scores = tfidf_matrix[i].toarray().flatten()
        top_idx = scores.argsort()[-top_n:][::-1]

        print(f"\n▶ {group} ({group_counts[group]} papers)")
        print(f"  {'Rank':>4s}  {'Term':25s}  {'TF-IDF':>8s}")
        print(f"  {'─'*42}")

        for rank, idx in enumerate(top_idx, 1):
            term = features[idx]
            score = scores[idx]
            tfidf_rows.append({
                "sheet": "tfidf_top",
                "group": group,
                "rank": rank,
                "term": term,
                "tfidf_score": round(score, 5),
            })
            bar = "█" * int(score * 80)
            print(f"  {rank:4d}  {term:25s}  {score:.5f}  {bar}")

    # ── Part 2: Distinctive Terms (이 그룹에서만 높은 단어) ──
    dist_rows = []
    print(f"\n{'='*60}")
    print(f"🔍 Distinctive Terms (high here, low elsewhere)")
    print(f"{'='*60}")

    for i, group in enumerate(group_list):
        scores_i = tfidf_matrix[i].toarray().flatten()
        others = [tfidf_matrix[j].toarray().flatten()
                  for j in range(len(group_list)) if j != i]
        avg_others = np.mean(others, axis=0)
        diff = scores_i - avg_others
        top_diff_idx = diff.argsort()[-20:][::-1]

        print(f"\n▶ {group} — distinctive terms:")
        rank = 0
        for idx in top_diff_idx:
            if diff[idx] > 0.005:  # 최소 차이 임계값
                rank += 1
                term = features[idx]
                dist_rows.append({
                    "sheet": "distinctive",
                    "group": group,
                    "rank": rank,
                    "term": term,
                    "this_group": round(scores_i[idx], 5),
                    "others_avg": round(avg_others[idx], 5),
                    "diff": round(diff[idx], 5),
                })
                print(f"  {rank:3d}. {term:25s}  "
                      f"diff={diff[idx]:.4f}  "
                      f"(this={scores_i[idx]:.4f}, others={avg_others[idx]:.4f})")

    # ── 저장 ──
    output_dir.mkdir(parents=True, exist_ok=True)

    df_tfidf = pd.DataFrame(tfidf_rows)
    df_dist = pd.DataFrame(dist_rows)

    # 하나의 Excel 파일에 시트 2개로 저장
    excel_path = output_dir / f"{output_dir.name}_tfidf_analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_tfidf.to_excel(writer, sheet_name="TF-IDF Top Terms", index=False)
        df_dist.to_excel(writer, sheet_name="Distinctive Terms", index=False)

    # CSV도 별도 저장 (Excel 없는 환경 대비)
    df_tfidf.to_csv(output_dir / f"{output_dir.name}_tfidf_top_terms.csv", index=False)
    df_dist.to_csv(output_dir / f"{output_dir.name}_tfidf_distinctive.csv", index=False)

    print(f"\n💾 Saved:")
    print(f"   Excel (2 sheets): {excel_path}")
    print(f"   CSV (top terms):  {output_dir / 'tfidf_top_terms.csv'}")
    print(f"   CSV (distinctive): {output_dir / 'tfidf_distinctive.csv'}")

    return df_tfidf, df_dist


# ── 메인 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TF-IDF analysis with academic stopwords")
    parser.add_argument("--input", default="data/raw/pilot_law_child.pkl",
                        help="Input pickle file from 01_collect_s2.py")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--top-n", type=int, default=25, help="Top N terms per group")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ File not found: {input_path}")
        print(f"  Run 01_collect_s2.py first!")
        return

    # 입력 파일명에서 분야 추출 (출력 파일명에 반영)
    field_tag = input_path.stem.replace("pilot_", "").replace("_child", "")

    print(f"{'='*60}")
    print(f"📊 TF-IDF Analysis (v2 — academic stopwords filtered)")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Field: {field_tag}")

    df = pd.read_pickle(input_path)
    print(f"Papers loaded: {len(df)}")

    output_dir = Path(args.output_dir) / field_tag
    run_tfidf(df, output_dir, top_n=args.top_n)

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
