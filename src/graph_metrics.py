def compute_graph_metrics(con_matrix, n_rand=10):
    #  Clean matrix 
    mat = np.nan_to_num(con_matrix, nan=0.0)
    mat[mat < 0] = 0.0
    np.fill_diagonal(mat, 0)

    #  Build weighted graph 
    G = nx.from_numpy_array(mat)

    # Metrics 
    mean_degree = np.mean([v for _, v in G.degree(weight="weight")])
    clustering = nx.average_clustering(G, weight="weight")

    try:
        path_length = nx.average_shortest_path_length(
            G, weight=lambda u, v, d: 1.0 / max(d.get("weight", 1e-5), 1e-5)
        )
    except nx.NetworkXError:
        path_length = np.nan

    #  Random graph comparison for small worldness
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    weights = np.array([d["weight"] for (_, _, d) in G.edges(data=True)])

    C_rand_list = []
    L_rand_list = []

    for _ in range(n_rand):
        # Generate random graph with same number of nodes/edges
        G_rand = nx.gnm_random_graph(n_nodes, n_edges)
        
        # Assign random weights
        for (u, v) in G_rand.edges():
            G_rand[u][v]["weight"] = float(np.random.choice(weights))

        # Compute metrics (lambda to avoid division by zero)
        C_rand_list.append(nx.average_clustering(G_rand, weight="weight"))
        try:
            L_rand_list.append(nx.average_shortest_path_length(
                G_rand, weight=lambda u, v, d: 1.0 / max(d.get("weight", 1e-5), 1e-5)
            ))
        except nx.NetworkXError:
            L_rand_list.append(np.nan)

    #  Average random graph metrics 
    C_rand = np.nanmean(C_rand_list)
    L_rand = np.nanmean(L_rand_list)

    # small worldness 
    if C_rand > 0 and L_rand > 0 and not np.isnan(path_length):
        small_worldness = (clustering / C_rand) / (path_length / L_rand)
    else:
        small_worldness = np.nan

    return {
        "mean_degree": mean_degree,
        "clustering": clustering,
        "path_length": path_length,
        "small_worldness": small_worldness,
    }
