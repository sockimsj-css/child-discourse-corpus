"""
03_cross_field_compare.py
=========================
여러 분야의 TF-IDF 결과를 합쳐서 비교 시각화.
- 분야 × 용어그룹 히트맵
- 용어그룹별 분야 간 바 차트
- 파일명에 분야 조합 태그 자동 부착 (e.g. law_psy_edu_heatmap.png)

사용법:
    py scripts/03_cross_field_compare.py
    py scripts/03_cross_field_compare.py --fields law psychology education
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ── 한글 폰트 설정 ──

def setup_font():
    korean_fonts = ["Malgun Gothic", "NanumGothic", "AppleGothic", "NotoSansCJK"]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in korean_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            print(f"  Font: {font}")
            return
    print("  Font: default (Korean may not render)")


TERM_GROUPS_ORDER = ["child/children", "adolescent", "youth", "juvenile", "minor", "teenage"]
COLORS = {
    "child/children": "#7B68EE",
    "adolescent":     "#4ECDC4",
    "youth":          "#45B7D1",
    "juvenile":       "#FF6B6B",
    "minor":          "#FFA07A",
    "teenage":        "#FFD93D",
}


# ── 데이터 로드 ──

def load_distinctive(field, outputs_base):
    candidates = [
        outputs_base / field / f"{field}_tfidf_distinctive.csv",
        outputs_base / field / "tfidf_distinctive.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["field"] = field
            return df
    print(f"  Warning: not found for '{field}': tried {[str(c) for c in candidates]}")
    return pd.DataFrame()


def load_top_terms(field, outputs_base):
    candidates = [
        outputs_base / field / f"{field}_tfidf_top_terms.csv",
        outputs_base / field / "tfidf_top_terms.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["field"] = field
            return df
    print(f"  Warning: not found for '{field}': tried {[str(c) for c in candidates]}")
    return pd.DataFrame()


# ── 시각화 ──

def plot_heatmap(all_dist, output_dir, tag):
    fields = sorted(all_dist["field"].unique())
    groups = [g for g in TERM_GROUPS_ORDER if g in all_dist["group"].unique()]

    fig, ax = plt.subplots(figsize=(max(len(fields) * 4, 10), max(len(groups) * 1.2, 6)))

    cell_texts = []
    cell_scores = []
    for group in groups:
        row_texts = []
        row_scores = []
        for field in fields:
            subset = all_dist[(all_dist["field"] == field) & (all_dist["group"] == group)]
            subset = subset.sort_values("diff", ascending=False).head(4)
            if len(subset) > 0:
                terms = subset["term"].tolist()
                avg_diff = subset["diff"].mean()
                row_texts.append("\n".join(terms))
                row_scores.append(avg_diff)
            else:
                row_texts.append("-")
                row_scores.append(0)
        cell_texts.append(row_texts)
        cell_scores.append(row_scores)

    scores_array = np.array(cell_scores)
    im = ax.imshow(scores_array, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels([f.upper() for f in fields], fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=12)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    for i in range(len(groups)):
        for j in range(len(fields)):
            text_color = "white" if scores_array[i, j] > scores_array.mean() else "black"
            ax.text(j, i, cell_texts[i][j],
                    ha="center", va="center", fontsize=8,
                    color=text_color, linespacing=1.4)

    ax.set_title("Distinctive Terms by Label x Field\n(top 4 terms per cell, color = avg TF-IDF diff)\n",
                 fontsize=14, fontweight="bold", pad=20)
    plt.colorbar(im, ax=ax, label="Mean TF-IDF Difference", shrink=0.8)
    plt.tight_layout()

    path = output_dir / f"{tag}_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def plot_term_bars(all_dist, output_dir, tag):
    fields = sorted(all_dist["field"].unique())
    groups = [g for g in TERM_GROUPS_ORDER if g in all_dist["group"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, group in enumerate(groups):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break

        group_data = all_dist[all_dist["group"] == group]

        for fi, field in enumerate(fields):
            fd = group_data[group_data["field"] == field].sort_values("diff", ascending=False).head(5)
            if fd.empty:
                continue
            y_pos = np.arange(len(fd)) + fi * 0.25
            ax.barh(y_pos, fd["diff"], height=0.22, label=field.upper(), alpha=0.85)
            for yp, (_, row) in zip(y_pos, fd.iterrows()):
                ax.text(row["diff"] + 0.005, yp, row["term"], va="center", fontsize=7)

        ax.set_title(f'"{group}"', fontsize=12, fontweight="bold",
                     color=COLORS.get(group, "black"))
        ax.set_xlabel("TF-IDF Diff", fontsize=9)
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, max(group_data["diff"].max() * 1.4, 0.1))

    for idx in range(len(groups), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle("How Each Label Is Framed Differently Across Fields",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = output_dir / f"{tag}_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def generate_summary_table(all_dist, output_dir, tag):
    fields = sorted(all_dist["field"].unique())
    groups = [g for g in TERM_GROUPS_ORDER if g in all_dist["group"].unique()]

    rows = []
    print(f"\n{'='*80}")
    print(f"CROSS-FIELD SUMMARY: Top 3 Distinctive Terms per Cell")
    print(f"{'='*80}")

    header = f"{'Label':<18s}" + "".join(f"{f.upper():<30s}" for f in fields)
    print(header)
    print("-" * len(header))

    for group in groups:
        row = {"label": group}
        line = f"{group:<18s}"
        for field in fields:
            subset = all_dist[(all_dist["field"] == field) & (all_dist["group"] == group)]
            top3 = subset.sort_values("diff", ascending=False).head(3)["term"].tolist()
            cell = ", ".join(top3) if top3 else "-"
            row[field] = cell
            line += f"{cell:<30s}"
        rows.append(row)
        print(line)

    df_summary = pd.DataFrame(rows)
    path = output_dir / f"{tag}_summary.csv"
    df_summary.to_csv(path, index=False)
    print(f"\n  Saved: {path}")


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="Cross-field TF-IDF comparison")
    parser.add_argument("--fields", nargs="+", default=["law", "psychology", "education"],
                        help="Fields to compare")
    parser.add_argument("--outputs-base", default="outputs", help="Base outputs directory")
    args = parser.parse_args()

    outputs_base = Path(args.outputs_base)
    output_dir = outputs_base / "compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 분야 태그 생성 (파일명용)
    tag = "_".join(f[:3] for f in args.fields)

    print(f"{'='*60}")
    print(f"Cross-Field Comparison")
    print(f"{'='*60}")
    print(f"Fields: {args.fields}")
    print(f"Tag: {tag}")

    setup_font()

    # 데이터 로드
    all_dist = []
    all_top = []
    for field in args.fields:
        df_d = load_distinctive(field, outputs_base)
        df_t = load_top_terms(field, outputs_base)
        if not df_d.empty:
            all_dist.append(df_d)
            print(f"  Loaded {field}: {len(df_d)} distinctive terms")
        if not df_t.empty:
            all_top.append(df_t)

    if not all_dist:
        print("No data loaded. Check outputs/ directories.")
        return

    all_dist = pd.concat(all_dist, ignore_index=True)
    all_top = pd.concat(all_top, ignore_index=True) if all_top else pd.DataFrame()

    # 시각화 (tag를 함수에 전달)
    print(f"\nGenerating visualizations...")
    plot_heatmap(all_dist, output_dir, tag)
    plot_term_bars(all_dist, output_dir, tag)
    generate_summary_table(all_dist, output_dir, tag)

    # 전체 데이터 저장
    all_dist.to_csv(output_dir / f"{tag}_distinctive.csv", index=False)
    if not all_top.empty:
        all_top.to_csv(output_dir / f"{tag}_top_terms.csv", index=False)

    print(f"\nAll outputs in: {output_dir}/")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
