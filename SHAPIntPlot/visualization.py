import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 1

def get_color(value, max_abs_shap):
    normalized = abs(value) / max_abs_shap
    light_blue_cyan = (0, 1, 1)
    pink = (1, 0.75, 0.8)

    if value < 0:
        color = (light_blue_cyan[0] * normalized, light_blue_cyan[1] * normalized, light_blue_cyan[2], 1)
    else:
        color = (pink[0], pink[1] * normalized, pink[2] * normalized, 1)

    return color

def create_final_network_graph(node_attributes, edge_attributes, max_node_size=1000, max_edge_width=5, 
                               spread=1.2, text_size=10, min_node_size=100, figsize=(8, 10), iterations=100, 
                               scale=1, use_average_degree=False):
    G = nx.Graph()
    max_interaction_strength = max(edge_attr['strength'] for edge_attr in edge_attributes.values())
    strengths = [attr['strength'] for attr in edge_attributes.values()]
    bipolar_scores = [attr['bipolar_score'] for attr in edge_attributes.values()]
    min_strength, max_strength = min(strengths), max(strengths)

    edge_colors = [
        (0, 0, 0, 1) if -0.05 < attr['bipolar_score'] < 0.05 else
        (0, 0, abs(attr['bipolar_score']), normalize(attr['strength'], min_strength, max_strength)) if attr['bipolar_score'] < 0 else
        (attr['bipolar_score'], 0, 0, normalize(attr['strength'], min_strength, max_strength))
        for attr in edge_attributes.values()
    ]

    scaled_edge_widths = [max_edge_width * (edge_attr['strength'] / max_interaction_strength) for edge_attr in edge_attributes.values()]
    max_importance = max(importance for importance, _ in node_attributes.values())
    scaled_node_sizes = {
        node: max(min_node_size, max_node_size * (importance / max_importance))
        for node, (importance, _) in node_attributes.items()
    }

    shap_values = [pair[1] for pair in node_attributes.values()]
    max_abs_shap = np.nanmax(np.abs(shap_values)) or 1

    node_colors = [
        (0.5, 0.5, 0.5, 1) if np.isnan(value) else get_color(value, max_abs_shap) for value in shap_values
    ]

    for node, size in scaled_node_sizes.items():
        G.add_node(node, size=size)
    for (source, target), width in zip(edge_attributes.keys(), scaled_edge_widths):
        G.add_edge(source, target, weight=width)

    node_density_based_k = spread * (1 / np.sqrt(len(node_attributes)))
    average_degree_based_k = node_density_based_k
    if use_average_degree:
        average_degree = np.mean([degree for _, degree in G.degree()])
        average_degree_based_k = spread * (1 / np.sqrt(average_degree * len(node_attributes)))

    k = average_degree_based_k if use_average_degree else node_density_based_k
    positions = nx.spring_layout(G, k=k, iterations=iterations, seed=42, scale=3)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, positions, node_size=[scaled_node_sizes[node] for node in G.nodes()], node_color=node_colors)
    nx.draw_networkx_edges(G, positions, width=scaled_edge_widths, edge_color=edge_colors)

    labels = {node: node for node in node_attributes.keys()}
    normal_labels = {node: labels[node] for node, size in scaled_node_sizes.items() if size > min_node_size}
    small_labels = {node: labels[node] for node, size in scaled_node_sizes.items() if size <= min_node_size}

    nx.draw_networkx_labels(G, positions, labels=normal_labels, font_size=text_size, font_color='black')
    nx.draw_networkx_labels(G, positions, labels=small_labels, font_size=text_size, font_color='red')

    plt.title('Final Feature Interaction Graph with SHAP Values')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
