"""
04_cooccurrence_network.py
==========================
초록 코퍼스에서 동시출현 네트워크를 구성하고 시각화.
- 분야별 또는 용어그룹별로 네트워크 생성
- Louvain 커뮤니티 탐지 → 색 부여
- 인터랙티브 HTML (pyvis) + 정적 PNG (matplotlib)

사용법:
    py scripts/04_cooccurrence_network.py
    py scripts/04_cooccurrence_network.py --input data/raw/pilot_law_child.pkl
    py scripts/04_cooccurrence_network.py --input data/raw/pilot_law_child.pkl --by-group
    py scripts/04_cooccurrence_network.py --all-fields
"""

import argparse
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    import networkx as nx
except ImportError:
    print("networkx 필요: pip install networkx")
    exit(1)

try:
    from community import community_louvain
except ImportError:
    print("python-louvain 필요: pip install python-louvain")
    exit(1)

try:
    from pyvis.network import Network as PyvisNetwork
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("  (pyvis 없음 — HTML 인터랙티브 출력 스킵. pip install pyvis)")


# ── 설정 ──

STOPWORDS = set("""
the a an and or but in on at to for of is are was were be been being
have has had do does did will would shall should may might can could
this that these those it its they them their he she his her we our
which what who whom how where when why not no nor all each every
any some such than too very just also more most other another
between through during before after above below from up down
about into over again further then once here there
study studies research results findings paper article analysis data
participants sample method methods approach review literature
examined explored discussed investigated aimed objective purpose
conclusion conclusions implications limitation limitations future
conducted design procedure measures measure instrument
significant significantly associated association relationship
effect effects outcome outcomes factor factors variable variables
prevalence rate rates percent ci odds ratio mean average
compared comparison difference differences similar positive negative
correlation regression model statistically multivariate logistic
adjusted group groups level levels type types form forms
use used using based include included including provides provide
suggest suggests indicate indicates report reported found showed show
impact role need needs important particular particularly specific
current present recent new different various common likely
potential possible related relevant key major general overall
however although moreover furthermore therefore thus addition
example context process issue issues et al doi https http
united states country countries national international global
year years period time age ages number total higher lower
increased decreased children child youth young adolescent juvenile
minor minors teenage teenager adolescence childhood people
""".split())

COMMUNITY_COLORS = [
    "#7B68EE", "#FF6B6B", "#4ECDC4", "#FFD93D", "#45B7D1",
    "#96CEB4", "#FF8C94", "#A8D8EA", "#DDA0DD", "#98D8C8",
    "#F7DC6F", "#BB8FCE", "#85C1E9", "#F0B27A", "#AED6F1",
]

TERM_GROUPS = {
    "child/children": ["child", "children", "childhood"],
    "adolescent": ["adolescent", "adolescence"],
    "youth": ["youth", "young people", "young adult"],
    "juvenile": ["juvenile"],
    "minor": ["minor", "minors"],
    "teenage": ["teenage", "teenager"],
}


def setup_font():
    korean_fonts = ["Malgun Gothic", "NanumGothic", "AppleGothic"]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in korean_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return


def preprocess(text):
    """초록 텍스트를 전처리 → 단어 리스트."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    return words


def build_cooccurrence(abstracts, window="sentence", min_count=3, top_n_words=200):
    """
    초록 리스트에서 동시출현 매트릭스 구축.
    window="sentence": 문장 단위로 동시출현 카운트
    window="abstract": 초록 전체 단위
    """
    # 전체 단어 빈도
    word_freq = Counter()
    for abstract in abstracts:
        words = preprocess(abstract)
        word_freq.update(words)

    # 상위 N개 단어만 사용
    top_words = set(w for w, _ in word_freq.most_common(top_n_words))

    # 동시출현 카운트
    cooccur = Counter()
    for abstract in abstracts:
        if window == "sentence":
            # 문장 단위
            sentences = re.split(r"[.!?]", abstract)
            units = sentences
        else:
            # 초록 전체 단위
            units = [abstract]

        for unit in units:
            words = preprocess(unit)
            words = [w for w in words if w in top_words]
            words = list(set(words))  # 중복 제거 (한 문장 내)
            for w1, w2 in combinations(sorted(words), 2):
                cooccur[(w1, w2)] += 1

    # 최소 빈도 필터
    cooccur = {pair: count for pair, count in cooccur.items() if count >= min_count}

    return cooccur, word_freq


def build_network(cooccur, word_freq, max_edges=300):
    """동시출현 데이터에서 NetworkX 그래프 생성."""
    G = nx.Graph()

    # 상위 엣지만 사용
    sorted_edges = sorted(cooccur.items(), key=lambda x: x[1], reverse=True)[:max_edges]

    for (w1, w2), weight in sorted_edges:
        G.add_edge(w1, w2, weight=weight)

    # 노드 크기 = 단어 빈도
    for node in G.nodes():
        G.nodes[node]["freq"] = word_freq.get(node, 1)

    # 고립 노드 제거
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G


def detect_communities(G):
    """Louvain 커뮤니티 탐지."""
    partition = community_louvain.best_partition(G, random_state=42)

    # 커뮤니티별 색상 배정
    communities = set(partition.values())
    color_map = {}
    for i, comm in enumerate(sorted(communities)):
        color_map[comm] = COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]

    for node in G.nodes():
        comm = partition[node]
        G.nodes[node]["community"] = comm
        G.nodes[node]["color"] = color_map[comm]

    return partition, color_map


def plot_network_static(G, partition, title, output_path, figsize=(16, 12)):
    """정적 PNG 네트워크 시각화."""
    fig, ax = plt.subplots(figsize=figsize)

    # 레이아웃
    pos = nx.spring_layout(G, k=1.5, iterations=80, seed=42, weight="weight")

    # 노드 크기 = 빈도 기반
    freqs = [G.nodes[n].get("freq", 1) for n in G.nodes()]
    max_freq = max(freqs) if freqs else 1
    node_sizes = [300 + (f / max_freq) * 2500 for f in freqs]

    # 노드 색상 = 커뮤니티
    node_colors = [G.nodes[n].get("color", "#999999") for n in G.nodes()]

    # 엣지 굵기 = 가중치
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.3 + (w / max_w) * 3.0 for w in weights]

    # 그리기
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           alpha=0.25, edge_color="#888888")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85, edgecolors="white", linewidths=0.5)

    # 라벨 (상위 노드만)
    top_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n].get("freq", 0), reverse=True)
    label_nodes = top_nodes[:min(60, len(top_nodes))]
    labels = {n: n for n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")

    # 커뮤니티 범례
    comm_nodes = defaultdict(list)
    for node, comm in partition.items():
        comm_nodes[comm].append(node)

    legend_handles = []
    for comm in sorted(comm_nodes.keys()):
        top_in_comm = sorted(comm_nodes[comm],
                             key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:3]
        label = ", ".join(top_in_comm)
        color = G.nodes[top_in_comm[0]].get("color", "#999")
        handle = plt.scatter([], [], c=color, s=100, label=f"C{comm}: {label}")
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc="lower left", fontsize=7,
              framealpha=0.9, title="Communities (top 3 words)")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def export_pyvis(G, title, output_path):
    """인터랙티브 HTML 네트워크 (pyvis)."""
    if not HAS_PYVIS:
        return

    net = PyvisNetwork(height="800px", width="100%", bgcolor="#ffffff",
                       font_color="#333333", notebook=False)
    net.heading = title
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=150)

    freqs = {n: G.nodes[n].get("freq", 1) for n in G.nodes()}
    max_freq = max(freqs.values()) if freqs else 1

    for node in G.nodes():
        size = 10 + (freqs[node] / max_freq) * 50
        color = G.nodes[node].get("color", "#999999")
        net.add_node(node, label=node, size=size, color=color,
                     title=f"{node}\nfreq: {freqs[node]}\ncommunity: {G.nodes[node].get('community', '?')}")

    for u, v in G.edges():
        w = G[u][v]["weight"]
        net.add_edge(u, v, value=w, title=f"co-occurrence: {w}")

    net.save_graph(str(output_path))
    print(f"  Saved: {output_path}")


def analyze_communities(G, partition, word_freq):
    """커뮤니티별 핵심 단어 출력."""
    comm_nodes = defaultdict(list)
    for node, comm in partition.items():
        comm_nodes[comm].append(node)

    print(f"\n  Communities detected: {len(comm_nodes)}")
    for comm in sorted(comm_nodes.keys()):
        nodes = comm_nodes[comm]
        top = sorted(nodes, key=lambda n: word_freq.get(n, 0), reverse=True)[:8]
        color = G.nodes[top[0]].get("color", "?") if top else "?"
        print(f"  C{comm} ({len(nodes)} words, {color}): {', '.join(top)}")


def assign_group(matched_terms):
    for group_name, group_terms in TERM_GROUPS.items():
        if any(t in matched_terms for t in group_terms):
            return group_name
    return "other"


def run_for_field(df, field_name, output_dir, by_group=False):
    """한 분야에 대해 네트워크 분석 실행."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = field_name[:3]

    if not by_group:
        # 분야 전체 네트워크
        print(f"\n{'='*50}")
        print(f"Building network: {field_name} (all papers)")
        print(f"{'='*50}")

        abstracts = df["abstract"].tolist()
        cooccur, word_freq = build_cooccurrence(abstracts, window="sentence", min_count=3, top_n_words=150)
        G = build_network(cooccur, word_freq, max_edges=250)
        partition, _ = detect_communities(G)

        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        analyze_communities(G, partition, word_freq)

        title = f"Co-occurrence Network: {field_name.upper()} ({len(df)} papers)"
        plot_network_static(G, partition, title,
                            output_dir / f"{tag}_network_all.png")
        export_pyvis(G, title, output_dir / f"{tag}_network_all.html")

        # 엣지 리스트 저장
        edges = []
        for u, v in G.edges():
            edges.append({"source": u, "target": v, "weight": G[u][v]["weight"]})
        pd.DataFrame(edges).to_csv(output_dir / f"{tag}_edges_all.csv", index=False)

    else:
        # 용어그룹별 네트워크
        df["term_group"] = df["matched_terms"].apply(assign_group)

        for group in TERM_GROUPS.keys():
            subset = df[df["term_group"] == group]
            if len(subset) < 20:
                print(f"\n  Skipping {group}: only {len(subset)} papers")
                continue

            print(f"\n{'='*50}")
            print(f"Building network: {field_name} / {group} ({len(subset)} papers)")
            print(f"{'='*50}")

            abstracts = subset["abstract"].tolist()
            cooccur, word_freq = build_cooccurrence(
                abstracts, window="sentence",
                min_count=max(2, len(subset) // 50),
                top_n_words=100
            )
            G = build_network(cooccur, word_freq, max_edges=150)

            if G.number_of_nodes() < 5:
                print(f"  Too few nodes ({G.number_of_nodes()}). Skipping.")
                continue

            partition, _ = detect_communities(G)

            print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            analyze_communities(G, partition, word_freq)

            group_slug = group.replace("/", "_")
            title = f"Co-occurrence: {field_name.upper()} / \"{group}\" ({len(subset)} papers)"
            plot_network_static(G, partition, title,
                                output_dir / f"{tag}_network_{group_slug}.png",
                                figsize=(14, 10))
            export_pyvis(G, title, output_dir / f"{tag}_network_{group_slug}.html")


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="Co-occurrence network from corpus")
    parser.add_argument("--input", default=None,
                        help="Single pkl file (e.g. data/raw/pilot_law_child.pkl)")
    parser.add_argument("--all-fields", action="store_true",
                        help="Process all pkl files in data/raw/")
    parser.add_argument("--by-group", action="store_true",
                        help="Build separate networks per term group")
    parser.add_argument("--output-dir", default="outputs", help="Base output directory")
    args = parser.parse_args()

    setup_font()

    if args.all_fields:
        pkl_files = sorted(Path("data/raw").glob("pilot_*_child.pkl"))
        if not pkl_files:
            print("No pkl files found in data/raw/")
            return
        print(f"Found {len(pkl_files)} field files: {[p.stem for p in pkl_files]}")
    elif args.input:
        pkl_files = [Path(args.input)]
    else:
        # 기본: data/raw/에서 찾기
        pkl_files = sorted(Path("data/raw").glob("pilot_*_child.pkl"))
        if not pkl_files:
            print("No input specified and no pkl files found. Use --input or --all-fields")
            return

    for pkl_path in pkl_files:
        if not pkl_path.exists():
            print(f"File not found: {pkl_path}")
            continue

        field_name = pkl_path.stem.replace("pilot_", "").replace("_child", "")
        df = pd.read_pickle(pkl_path)
        print(f"\nLoaded {field_name}: {len(df)} papers")

        output_dir = Path(args.output_dir) / field_name
        run_for_field(df, field_name, output_dir, by_group=args.by_group)

    print(f"\nAll done!")


if __name__ == "__main__":
    main()
