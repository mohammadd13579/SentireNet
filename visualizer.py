import plotly.graph_objects as go
import networkx as nx
import numpy as np

def draw_graph(G):
    """
    Generates an interactive 3D or 2D Plotly graph visualization.
    """
    if not G.nodes:
        return go.Figure(layout_title_text="No graph to display. Please run analysis.")

    # Get a 2D layout
    pos = nx.spring_layout(G, dim=2)

    # Prepare data for Plotly
    edge_x = []
    edge_y = []
    edge_weights = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = G.edges[edge]['semantic_similarity']
        edge_weights.append(weight)

    # Create the edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        data = G.nodes[node]
        polarity = data['polarity']
        count = data['count']
        
        # Set text, color, and size
        node_text.append(f"{node}<br>Count: {count}<br>Polarity: {polarity:.2f}")
        node_color.append(polarity)
        node_size.append(5 + (count * 1.5))

    # Create the node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Polarity',
                    side='right'
                ),
                xanchor='left'
            ),
            line_width=2,
            cmin=-1,
            cmax=1
        ))


    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text='SentireNet: Semantic & Emotional Web',
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[ dict(
                text="Node size = frequency, Node color = sentiment polarity",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
 
    return fig