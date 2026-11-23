import plotly.graph_objects as go

DARK_BACKGROUND = '#0d1117'
DARK_GRID = '#30363d'
DARK_TEXT = '#e6edf3'
DARK_ZERO_LINE = '#6e7681'

def apply_dark_3d_style(fig, title, zaxis_title, width=1100, height=800):
    axis_style = dict(
        backgroundcolor=DARK_BACKGROUND,
        gridcolor=DARK_GRID,
        zerolinecolor=DARK_ZERO_LINE,
        color=DARK_TEXT,
        showbackground=True
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        width=width,
        height=height,
        template='plotly_dark',
        paper_bgcolor=DARK_BACKGROUND,
        plot_bgcolor=DARK_BACKGROUND,
        font=dict(color=DARK_TEXT),
        hoverlabel=dict(bgcolor='#161b22', font_color=DARK_TEXT),
        scene=dict(
            bgcolor=DARK_BACKGROUND,
            xaxis=dict(title='Log-moneyness', **axis_style),
            yaxis=dict(title='Maturity, years', **axis_style),
            zaxis=dict(title=zaxis_title, **axis_style),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))
        ),
        legend=dict(
            bgcolor='rgba(13, 17, 23, 0.75)',
            bordercolor=DARK_GRID,
            borderwidth=1
        ),
        margin=dict(l=10, r=10, b=10, t=60)
    )

    return fig


def dark_3d_scatter(x, y, z, title, zaxis_title, name, colorscale='Viridis'):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        name=name,
        marker=dict(
            size=2.5,
            color=z,
            colorscale=colorscale,
            opacity=0.75,
            showscale=True,
            colorbar=dict(title=zaxis_title, len=0.65, thickness=14)
        ),
        hovertemplate='k: %{x:.3f}<br>T: %{y:.3f}<br>Volatility: %{z:.2f}%<extra></extra>'
    ))

    return apply_dark_3d_style(fig, title, zaxis_title)


def dark_3d_surface(x, y, z, title, zaxis_title, name, colorscale='Viridis'):
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=z,
        name=name,
        colorscale=colorscale,
        opacity=0.9,
        showscale=True,
        colorbar=dict(title=zaxis_title, len=0.65, thickness=14),
        lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.9, specular=0.2),
        hovertemplate='k: %{x:.3f}<br>T: %{y:.3f}<br>Volatility: %{z:.2f}%<extra></extra>'
    ))

    return apply_dark_3d_style(fig, title, zaxis_title)