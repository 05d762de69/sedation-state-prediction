import numpy as np
import networkx as nx
from networkx.algorithms import efficiency_measures
from community import community_louvain
import brainconn
from brainconn.degree import degrees_dir
from brainconn.clustering import clustering_coef_wd
from brainconn.distance import charpath, efficiency_wei
from brainconn.modularity import modularity_louvain_und

def compute_graph_metrics(con_matrix, n_rand=10):

    #  Clean and prepare matrix 
    mat = np.nan_to_num(con_matrix, nan=0.0)
    mat[mat < 0] = 0.0
    np.fill_diagonal(mat, 0)

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


def safe_charpath(D, **kwargs):
    """Handle variable number of outputs from charpath."""
    result = charpath(D, **kwargs)
    return result[0] if isinstance(result, tuple) else result


def compute_graph_metrics_epochs(con_matrix, n_rand=3, eps=1e-5):
    """
    Compute graph-theoretical metrics for a weighted (typically undirected) connectivity matrix
    using brainconn (FIU version) + networkx fallback for small-worldness.
    """

    # Clean & symmetrize 
    W = np.nan_to_num(con_matrix, nan=0.0)
    W[W < 0] = 0.0
    np.fill_diagonal(W, 0)
    W = (W + W.T) / 2  # enforce symmetry

    n_nodes = W.shape[0]
    directed = not np.allclose(W, W.T, atol=1e-10)

    #  Mean strength (weighted degree)
    total_strength = np.sum(W, axis=1)
    mean_strength = np.mean(total_strength)

    #  Clustering 
    clustering = np.nanmean(clustering_coef_wd(W))

    #  Path length 
    D = 1.0 / (W + eps)
    D[W == 0] = np.inf
    path_length = safe_charpath(D, include_diagonal=False, include_infinite=False)

    #  Efficiency 
    global_eff = efficiency_wei(W, local=False)
    local_eff = efficiency_wei(W, local=True)

    # Modularity (undirected Louvain)
    try:
        Ci, Q = modularity_louvain_und(W)
    except Exception:
        Ci, Q = np.full(W.shape[0], np.nan), np.nan

    # Participation coefficient (undirected)
    participation = np.zeros(n_nodes)
    for i in range(n_nodes):
        if total_strength[i] == 0:
            continue
        for c in np.unique(Ci):
            nodes_c = np.where(Ci == c)[0]
            w_ic = np.sum(W[i, nodes_c])
            participation[i] += (w_ic / total_strength[i]) ** 2
        participation[i] = 1 - participation[i]
    participation_coeff = np.mean(participation)

    # Random graph comparison (small-worldness)
    n_edges = np.count_nonzero(W)
    weights = W[W > 0]
    C_rand_list, L_rand_list = [], []

    for _ in range(n_rand):
        G_rand = nx.gnm_random_graph(n_nodes, n_edges, directed=False)
        for (u, v) in G_rand.edges():
            G_rand[u][v]["weight"] = float(np.random.choice(weights))
        C_rand_list.append(np.nanmean(list(nx.clustering(G_rand, weight="weight").values())))
        try:
            L_rand_list.append(nx.average_shortest_path_length(
                G_rand,
                weight=lambda u, v, d: 1.0 / max(d.get("weight", eps), eps),
            ))
        except nx.NetworkXError:
            L_rand_list.append(np.nan)

    C_rand, L_rand = np.nanmean(C_rand_list), np.nanmean(L_rand_list)
    small_worldness = (
        (clustering / C_rand) / (path_length / L_rand)
        if C_rand > 0 and L_rand > 0 and not np.isnan(path_length)
        else np.nan
    )

    # Scalars only 
    global_eff = np.nanmean(global_eff)
    local_eff = np.nanmean(local_eff)
    clustering = np.nanmean(clustering)
    participation_coeff = np.nanmean(participation_coeff)

    #  Return summary
    return {
        "mean_strength": mean_strength,
        "clustering": clustering,
        "path_length": path_length,
        "global_efficiency": global_eff,
        "local_efficiency": local_eff,
        "modularity": Q,
        "participation_coefficient": participation_coeff,
        "small_worldness": small_worldness,
    }
