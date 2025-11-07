import numpy as np
import networkx as nx
from networkx.algorithms import efficiency_measures
from community import community_louvain

def compute_graph_metrics(con_matrix, n_rand=10):

    #  Clean and prepare matrix 
    mat = np.nan_to_num(con_matrix, nan=0.0)
    mat[mat < 0] = 0.0
    np.fill_diagonal(mat, 0)

    threshold = np.percentile(mat[mat > 0], 75)
    mat[mat < threshold] = 0

    G = nx.from_numpy_array(mat)

    # Core metrics 
    mean_degree = np.mean([v for _, v in G.degree(weight="weight")])
    clustering = nx.average_clustering(G, weight="weight")

    try:
        path_length = nx.average_shortest_path_length(
            G, weight=lambda u, v, d: 1.0 / max(d.get("weight", 1e-5), 1e-5)
        )
    except (nx.NetworkXError, ZeroDivisionError):
        path_length = np.nan

    try:
        global_eff = efficiency_measures.global_efficiency(
            nx.Graph([(u, v, {"weight": 1.0 / max(w, 1e-5)}) for u, v, w in G.edges(data="weight")])
        )
    except ZeroDivisionError:
        global_eff = np.nan

    # Local efficiency
    try:
        local_eff = efficiency_measures.local_efficiency(G)
    except Exception:
        local_eff = np.nan

    # Community structure (modularity + participation coefficient) 
    try:
        partition = community_louvain.best_partition(G, weight="weight")
        Q = community_louvain.modularity(partition, G, weight="weight")
        communities = {}
        for node, comm in partition.items():
            communities.setdefault(comm, []).append(node)

        # Participation coefficient
        node_strength = dict(G.degree(weight="weight"))
        part_coeffs = []
        for i in G.nodes():
            k_i = node_strength[i]
            if k_i == 0:
                part_coeffs.append(0)
                continue
            sum_sq = 0
            for c_nodes in communities.values():
                w_ic = sum(G[i][j]["weight"] for j in c_nodes if G.has_edge(i, j))
                sum_sq += (w_ic / k_i) ** 2
            part_coeffs.append(1 - sum_sq)
        participation_coeff = np.mean(part_coeffs)
    except Exception:
        Q, participation_coeff = np.nan, np.nan

    # Random graph comparison (for small-worldness) 
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    weights = np.array([d["weight"] for (_, _, d) in G.edges(data=True)])

    C_rand_list, L_rand_list = [], []
    for _ in range(n_rand):
        G_rand = nx.gnm_random_graph(n_nodes, n_edges)
        for (u, v) in G_rand.edges():
            G_rand[u][v]["weight"] = float(np.random.choice(weights))
        C_rand_list.append(nx.average_clustering(G_rand, weight="weight"))
        try:
            L_rand_list.append(
                nx.average_shortest_path_length(
                    G_rand, weight=lambda u, v, d: 1.0 / max(d.get("weight", 1e-5), 1e-5)
                )
            )
        except nx.NetworkXError:
            L_rand_list.append(np.nan)

    C_rand, L_rand = np.nanmean(C_rand_list), np.nanmean(L_rand_list)

    if C_rand > 0 and L_rand > 0 and not np.isnan(path_length):
        small_worldness = (clustering / C_rand) / (path_length / L_rand)
    else:
        small_worldness = np.nan

    # Return metrics
    return {
        "mean_degree": mean_degree,
        "clustering": clustering,
        "path_length": path_length,
        "global_efficiency": global_eff,
        "local_efficiency": local_eff,
        "modularity": Q,
        "participation_coefficient": participation_coeff,
        "small_worldness": small_worldness,
    }
