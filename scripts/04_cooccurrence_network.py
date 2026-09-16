"""
04_cooccurrence_network.py  (v3)
=================================
Co-occurrence network from academic paper abstracts.
- Enhanced academic stopwords
- Node size/edge width visually exaggerated for clarity
- Node stats CSV with degree_centrality, betweenness_centrality
- Top degree/betweenness nodes annotated on the figure
- All output in English

Usage:
    py scripts/04_cooccurrence_network.py --all-fields
    py scripts/04_cooccurrence_network.py --all-fields --by-group
    py scripts/04_cooccurrence_network.py --input data/raw/pilot_law_child.pkl
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
    print("Required: pip install networkx"); exit(1)
try:
    from community import community_louvain
except ImportError:
    print("Required: pip install python-louvain"); exit(1)
try:
    from pyvis.network import Network as PyvisNetwork
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False


# -- Stopwords (enhanced) --

STOPWORDS = set("""
the a an and or but in on at to for of is are was were be been being
have has had do does did will would shall should may might can could
this that these those it its they them their he she his her we our you
which what who whom how where when why not no nor all each every
any some such than too very just also more most other another
between through during before after above below from up down
about into over again further then once here there
with within without among both upon along across against
study studies research results findings paper article analysis data
participants sample method methods approach review literature
examined explored discussed investigated aimed objective purpose
conclusion conclusions implications limitation limitations future
conducted design procedure measures measure instrument
significant significantly associated association relationship
effect effects outcome outcomes factor factors variable variables
prevalence rate rates percent ci odds ratio mean average sd se
compared comparison difference differences similar positive negative
correlation regression model statistically multivariate logistic
adjusted group groups level levels type types form forms
use used using based include included including provides provide
suggest suggests indicate indicates report reported found showed show
impact role need needs important particular particularly specific
current present recent new different various common likely
potential possible related relevant key major general overall
given certain well however although moreover furthermore therefore
thus addition example context process issue issues
et al doi https http vol pp isbn university journal press published
united states country countries national international global
year years period time age ages number total higher lower
increased decreased whether two three one first may often many
several still making made make per since
children child youth young adolescent adolescence juvenile
minor minors teenage teenager childhood people
abstract purpose objective aims conclusions background aim
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
    for font in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            return


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    return words


def build_cooccurrence(abstracts, min_count=3, top_n_words=150):
    word_freq = Counter()
    for abstract in abstracts:
        word_freq.update(preprocess(abstract))
    top_words = set(w for w, _ in word_freq.most_common(top_n_words))
    cooccur = Counter()
    for abstract in abstracts:
        for sent in re.split(r"[.!?]", abstract):
            words = list(set(w for w in preprocess(sent) if w in top_words))
            for w1, w2 in combinations(sorted(words), 2):
                cooccur[(w1, w2)] += 1
    cooccur = {p: c for p, c in cooccur.items() if c >= min_count}
    return cooccur, word_freq


def build_network(cooccur, word_freq, max_edges=300):
    G = nx.Graph()
    for (w1, w2), weight in sorted(cooccur.items(), key=lambda x: x[1], reverse=True)[:max_edges]:
        G.add_edge(w1, w2, weight=weight)
    for node in G.nodes():
        G.nodes[node]["freq"] = word_freq.get(node, 1)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def detect_communities(G):
    partition = community_louvain.best_partition(G, random_state=42)
    color_map = {comm: COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
                 for i, comm in enumerate(sorted(set(partition.values())))}
    for node in G.nodes():
        G.nodes[node]["community"] = partition[node]
        G.nodes[node]["color"] = color_map[partition[node]]
    return partition, color_map


def compute_node_stats(G, partition, word_freq):
    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, weight="weight")
    rows = []
    for node in G.nodes():
        rows.append({
            "node": node,
            "degree_centrality": round(deg[node], 4),
            "degree_count": G.degree(node),
            "betweenness_centrality": round(btw[node], 4),
            "community": partition.get(node, -1),
            "community_color": G.nodes[node].get("color", ""),
            "freq": word_freq.get(node, 0),
            "weighted_degree": sum(G[node][nbr]["weight"] for nbr in G.neighbors(node)),
        })
    return pd.DataFrame(rows).sort_values("betweenness_centrality", ascending=False)


def plot_network_static(G, partition, word_freq, title, output_path, figsize=(18, 13)):
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    ax_table = axes[1]

    pos = nx.spring_layout(G, k=2.0, iterations=120, seed=42, weight="weight")

    # -- Node sizes: sqrt scale, wide range --
    freqs = np.array([G.nodes[n].get("freq", 1) for n in G.nodes()])
    freq_norm = np.sqrt(freqs / max(freqs.max(), 1))
    node_sizes = 80 + freq_norm * 4500

    node_colors = [G.nodes[n].get("color", "#999") for n in G.nodes()]

    # -- Edge widths: exaggerated --
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    if len(weights) > 0:
        w_norm = weights / max(weights.max(), 1)
        edge_widths = 0.15 + w_norm * 7.0
        edge_alphas = 0.06 + w_norm * 0.45
    else:
        edge_widths, edge_alphas = [1], [0.2]

    # Draw edges individually (for per-edge alpha)
    for idx, (u, v) in enumerate(G.edges()):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#888888", linewidth=edge_widths[idx],
                alpha=float(edge_alphas[idx]))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.88,
                           edgecolors="white", linewidths=0.8)

    # Labels: top nodes, font size proportional to freq
    nodes_sorted = sorted(G.nodes(), key=lambda n: G.nodes[n].get("freq", 0), reverse=True)
    label_count = min(50, len(nodes_sorted))
    for i, node in enumerate(nodes_sorted[:label_count]):
        x, y = pos[node]
        fs = max(6, min(15, 6 + freq_norm[list(G.nodes()).index(node)] * 11))
        fw = "bold" if i < 8 else "normal"
        ax.text(x, y, node, ha="center", va="center", fontsize=fs, fontweight=fw)

    # Community legend
    comm_nodes = defaultdict(list)
    for node, comm in partition.items():
        comm_nodes[comm].append(node)
    legend_handles = []
    for comm in sorted(comm_nodes.keys()):
        top = sorted(comm_nodes[comm], key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:3]
        color = G.nodes[top[0]].get("color", "#999")
        legend_handles.append(plt.scatter([], [], c=color, s=120, label=f"C{comm}: {', '.join(top)}",
                                          edgecolors="white", linewidths=0.5))
    ax.legend(handles=legend_handles, loc="lower left", fontsize=6.5,
              framealpha=0.92, title="Communities (top 3 words)", title_fontsize=7)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")

    # -- Right panel: stats table --
    stats = compute_node_stats(G, partition, word_freq)

    # Top 7 by degree_centrality
    top_deg = stats.sort_values("degree_centrality", ascending=False).head(7)
    # Top 7 by betweenness_centrality
    top_btw = stats.sort_values("betweenness_centrality", ascending=False).head(7)

    ax_table.axis("off")
    y = 0.95

    ax_table.text(0.0, y, "Top 7 — Degree Centrality", fontsize=10, fontweight="bold",
                  transform=ax_table.transAxes, va="top")
    y -= 0.04
    ax_table.text(0.0, y, f"{'Node':15s} {'Deg':>4s} {'Comm':>5s}",
                  fontsize=8, fontfamily="monospace", transform=ax_table.transAxes, va="top",
                  color="#555555")
    y -= 0.03
    for _, r in top_deg.iterrows():
        color = r["community_color"]
        ax_table.text(0.0, y, f"{r['node']:15s} {r['degree_count']:4d}   C{r['community']}",
                      fontsize=8.5, fontfamily="monospace", transform=ax_table.transAxes, va="top",
                      color=color, fontweight="bold")
        y -= 0.03

    y -= 0.04
    ax_table.text(0.0, y, "Top 7 — Betweenness Centrality", fontsize=10, fontweight="bold",
                  transform=ax_table.transAxes, va="top")
    y -= 0.04
    ax_table.text(0.0, y, f"{'Node':15s} {'Btw':>7s} {'Comm':>5s}",
                  fontsize=8, fontfamily="monospace", transform=ax_table.transAxes, va="top",
                  color="#555555")
    y -= 0.03
    for _, r in top_btw.iterrows():
        color = r["community_color"]
        ax_table.text(0.0, y, f"{r['node']:15s} {r['betweenness_centrality']:.4f}   C{r['community']}",
                      fontsize=8.5, fontfamily="monospace", transform=ax_table.transAxes, va="top",
                      color=color, fontweight="bold")
        y -= 0.03

    # Summary stats
    y -= 0.05
    ax_table.text(0.0, y, f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}",
                  fontsize=8, transform=ax_table.transAxes, va="top", color="#888888")
    y -= 0.03
    ax_table.text(0.0, y, f"Communities: {len(set(partition.values()))}",
                  fontsize=8, transform=ax_table.transAxes, va="top", color="#888888")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  PNG: {output_path}")


def export_pyvis(G, title, output_path):
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
        color = G.nodes[node].get("color", "#999")
        net.add_node(node, label=node, size=size, color=color,
                     title=f"{node}\nfreq: {freqs[node]}\ncommunity: {G.nodes[node].get('community','?')}")
    for u, v in G.edges():
        net.add_edge(u, v, value=G[u][v]["weight"], title=f"co-occurrence: {G[u][v]['weight']}")
    net.save_graph(str(output_path))
    print(f"  HTML: {output_path}")


def run_for_field(df, field_name, output_base, by_group=False):
    tag = field_name[:3]

    if not by_group:
        out_fig = output_base / "figures"
        out_html = output_base / "interactive"
        out_data = output_base / "data"
        for d in [out_fig, out_html, out_data]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Network: {field_name} (all, {len(df)} papers)")
        print(f"{'='*50}")

        abstracts = df["abstract"].tolist()
        cooccur, word_freq = build_cooccurrence(abstracts, min_count=3, top_n_words=150)
        G = build_network(cooccur, word_freq, max_edges=250)
        partition, _ = detect_communities(G)
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        title = f"Co-occurrence: {field_name.upper()} ({len(df)} papers)"
        plot_network_static(G, partition, word_freq, title,
                            out_fig / f"{tag}_network_all.png")
        export_pyvis(G, title, out_html / f"{tag}_network_all.html")

        stats = compute_node_stats(G, partition, word_freq)
        stats.to_csv(out_data / f"{tag}_node_stats_all.csv", index=False)

        edges = [{"source": u, "target": v, "weight": G[u][v]["weight"]}
                 for u, v in G.edges()]
        pd.DataFrame(edges).sort_values("weight", ascending=False).to_csv(
            out_data / f"{tag}_edges_all.csv", index=False)

        print(f"\n  Top 5 Degree Centrality:")
        for _, r in stats.sort_values("degree_centrality", ascending=False).head(5).iterrows():
            print(f"    {r['node']:20s}  deg={r['degree_count']:3d}  C{r['community']}")
        print(f"  Top 5 Betweenness Centrality:")
        for _, r in stats.sort_values("betweenness_centrality", ascending=False).head(5).iterrows():
            print(f"    {r['node']:20s}  btw={r['betweenness_centrality']:.4f}  C{r['community']}")

    else:
        df = df.copy()
        df["term_group"] = df["matched_terms"].apply(
            lambda terms: next((gn for gn, gt in TERM_GROUPS.items()
                                if any(t in terms for t in gt)), "other"))

        for group in TERM_GROUPS.keys():
            subset = df[df["term_group"] == group]
            if len(subset) < 20:
                print(f"\n  Skip {group}: {len(subset)} papers")
                continue

            out_fig = output_base / "figures"
            out_html = output_base / "interactive"
            out_data = output_base / "data"
            for d in [out_fig, out_html, out_data]:
                d.mkdir(parents=True, exist_ok=True)

            group_slug = group.replace("/", "_")

            print(f"\n{'='*50}")
            print(f"Network: {field_name} / \"{group}\" ({len(subset)} papers)")
            print(f"{'='*50}")

            abstracts = subset["abstract"].tolist()
            cooccur, word_freq = build_cooccurrence(
                abstracts, min_count=max(2, len(subset)//50), top_n_words=100)
            G = build_network(cooccur, word_freq, max_edges=150)

            if G.number_of_nodes() < 5:
                print(f"  Too few nodes. Skip.")
                continue

            partition, _ = detect_communities(G)
            print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

            title = f"Co-occurrence: {field_name.upper()} / \"{group}\" ({len(subset)} papers)"
            plot_network_static(G, partition, word_freq, title,
                                out_fig / f"{tag}_network_{group_slug}.png",
                                figsize=(18, 11))
            export_pyvis(G, title, out_html / f"{tag}_network_{group_slug}.html")

            stats = compute_node_stats(G, partition, word_freq)
            stats.to_csv(out_data / f"{tag}_node_stats_{group_slug}.csv", index=False)

            print(f"  Top 5 Degree Centrality:")
            for _, r in stats.sort_values("degree_centrality", ascending=False).head(5).iterrows():
                print(f"    {r['node']:20s}  deg={r['degree_count']:3d}  C{r['community']}")
            print(f"  Top 5 Betweenness Centrality:")
            for _, r in stats.sort_values("betweenness_centrality", ascending=False).head(5).iterrows():
                print(f"    {r['node']:20s}  btw={r['betweenness_centrality']:.4f}  C{r['community']}")


def main():
    parser = argparse.ArgumentParser(description="Co-occurrence network (v3)")
    parser.add_argument("--input", default=None)
    parser.add_argument("--all-fields", action="store_true")
    parser.add_argument("--by-group", action="store_true")
    parser.add_argument("--output-dir", default="outputs/03_network")
    args = parser.parse_args()

    setup_font()

    if args.all_fields:
        pkl_files = sorted(Path("data/raw").glob("pilot_*_child.pkl"))
    elif args.input:
        pkl_files = [Path(args.input)]
    else:
        pkl_files = sorted(Path("data/raw").glob("pilot_*_child.pkl"))

    if not pkl_files:
        print("No input files found."); return

    for pkl_path in pkl_files:
        if not pkl_path.exists():
            continue
        field_name = pkl_path.stem.replace("pilot_", "").replace("_child", "")
        df = pd.read_pickle(pkl_path)
        print(f"\nLoaded {field_name}: {len(df)} papers")

        if args.by_group:
            output_base = Path(args.output_dir) / "by_group" / field_name
        else:
            output_base = Path(args.output_dir) / "by_field" / field_name
        run_for_field(df, field_name, output_base, by_group=args.by_group)

    print(f"\nAll done!")


if __name__ == "__main__":
    main()
