import pickle

import pandas as pd
import numpy as np
import numpy.typing as npt

from pathlib import Path
from fire import Fire
from tqdm.cli import tqdm
from thefuzz import fuzz
from itertools import combinations, product
from typing import Callable, Any, List, Optional


def get_from_tmp(tmp_fn: Path, fn: Callable, *args, **kwargs) -> Any:
    if tmp_fn.exists():
        print(f"You are using cached data from tmp file: {tmp_fn}, debug only!")
        with open(tmp_fn, "rb") as pf:
            obj = pickle.load(pf)
    else:
        obj = fn(*args, **kwargs)
        with open(tmp_fn, "wb") as pf:
            pickle.dump(obj, pf)
    return obj


def calc_msg_dist(df: pd.DataFrame) -> npt.NDArray:
    df["text"] = df["text"].apply(lambda m: m.lower())

    n_msg = len(df)
    msg_dist = np.full((n_msg, n_msg), 0.0)
    # msg_dist = np.full((n_msg, n_msg), np.inf)

    for msg_i, msg_j in tqdm(list(combinations(range(n_msg), 2))):
        if df.channel_name.iloc[msg_i] != df.channel_name.iloc[msg_j]:
            dist = fuzz.ratio(df.text.iloc[msg_i], df.text.iloc[msg_j])
            msg_dist[msg_i, msg_j] = dist
            msg_dist[msg_j, msg_i] = dist
    return msg_dist


def get_msg_groups(
    msg_dist: npt.NDArray,
    dist_threshold: float,
) -> List[List[int]]:

    msg_is_close = msg_dist >= dist_threshold

    n_msg = msg_dist.shape[0]
    used = [False for _ in range(n_msg)]

    groups = []

    def _add_to_group(m, g):
        # print(m)
        if not used[m]:
            used[m] = True
            g.append(m)

            for mm in np.flatnonzero(msg_is_close[m]):
                _add_to_group(mm, g)

    for msg_idx in range(n_msg):
        # print(f"Start group from {msg_idx}")
        group = []
        _add_to_group(msg_idx, group)

        if group:
            groups.append(group)
    return groups


def filter_by_group_size(groups: List[List[int]], min_group_len: int) -> List[List[int]]:
    return [g for g in groups if len(g) >= min_group_len]

def remove_duplicates_inside_channel(
    groups: List[List[int]],
    df: pd.DataFrame
) -> List[List[int]]:

    for group_idx in range(len(groups)):
        group = groups[group_idx]
        remove = [False for _ in range(len(group))]
        for i, j in combinations(range(len(group)), 2):
            msg_i_idx = group[i]
            msg_j_idx = group[j]
            if (not remove[i] and not remove[j]) and \
               (df.iloc[msg_i_idx].channel_name == df.iloc[msg_j_idx].channel_name):
                if df.iloc[msg_i_idx].date <= df.iloc[msg_j_idx].date:
                    remove[j] = True
                else:
                    remove[i] = True
        groups[group_idx] = [m for m, r in zip(group, remove) if not r]
    return groups


def print_groups(groups: List[List[int]], df: pd.DataFrame, screen_width: int = 88) -> None:
    for g in groups:
        print(f"Message group len: {len(g)}")
        print("*" * screen_width)
        for msg in g:
            print(f"Channel name: {df.channel_name.iloc[msg]}")
            print(f"Date: {df.date.iloc[msg]}")
            print(f"Message idx: {msg}")

            print()
            print(df.text.iloc[msg])
            print("- " * (screen_width // 2))
        print("*" * screen_width)
        print()
        print()


def main(data_folder: str, ignore_subfolders: Optional[List[str]] = None):
    data_folder = Path(data_folder)
    dfs = {fn: pd.read_csv(fn, parse_dates=["date"]) for fn in data_folder.glob("**/*.csv")}

    if ignore_subfolders is not None:
        print(f"Ignore subfolders: {ignore_subfolders}")
        dfs = {k: v for k, v in dfs.items()
               if all([ig != k.parent.name for ig in ignore_subfolders])}
 

    for channel_folder in dfs:
        dfs[channel_folder]["channel_name"] = channel_folder.name

    df = pd.concat(list(dfs.values()))
    df = df[df["text"].notna()]
    df = df.reset_index(drop=True)
    df = df[["text", "channel_name", "date"]]


    msg_dist_save_fn = f"msg_dist_{data_folder.stem}.pkl" \
                       if ignore_subfolders is None else \
                       f"msg_dist_{data_folder.stem}_without_{'_'.join(ignore_subfolders)}.pkl"
    msg_dist_save_fn = Path(msg_dist_save_fn)
    msg_dist = get_from_tmp(
        msg_dist_save_fn,
        calc_msg_dist, df,
    )

    dist_threshold = 90
    min_group_len = 5

    groups = get_msg_groups(msg_dist, dist_threshold)

    # Remove duplicates inside channel
    groups = remove_duplicates_inside_channel(groups, df)

    # Remove small groups
    groups = filter_by_group_size(groups, min_group_len)

    # screen_width = 88
    screen_width = 193
    print_groups(groups, df, screen_width)
    print(f"Num of groups founded in {data_folder}: {len(groups)}")
    print(f"Num of channels: {len(df.channel_name.unique())}")
    print(f"Message distance treshold: {dist_threshold}")
    print(f"Min elements in group: {min_group_len}")


    # Create and draw graph
    import networkx as nx
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    g = nx.Graph()
    for msg_group in groups:
        channels = [df.iloc[msg_idx].channel_name.replace(".csv", "") for msg_idx in msg_group]
        for ch_i, ch_j in combinations(channels, 2):
            if g.has_edge(ch_i, ch_j):
                g[ch_i][ch_j]["weight"] += 1
            else:
                g.add_edge(ch_i, ch_j, weight=1)

    print(g)

    pos = nx.kamada_kawai_layout(g)
    
    edge_traces = []
    edge_text = []
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = g[edge[0]][edge[1]]["weight"]
        # print(weight)
        edge_trace = go.Scatter(
            x=[x0, x1], 
            y=[y0, y1],
            line=dict(width=weight * 0.2, color = 'cornflowerblue'),
            hoverinfo="text",
            # text = ([text]),
            mode='lines',
        )
        # edge_trace.text = f"Duplicates: {/weight}"

        edge_traces.append(edge_trace)


  #запасати в hoverinfo кількість зʼєднань



    node_x = []
    node_y = []

    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text = [],
        mode='markers+text',
        textposition = "top center",
        hoverinfo='text',
        hovertemplate = [],
        line_width=7,
        marker = dict(
            color=[],
            size =[],
            line = None,
            )
        )

    node_adjacencies = []
    node_text = []

    node_names = list(g.nodes())
    for node, adjacencies in enumerate(g.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_trace['marker']['color'] += tuple(['cornflowerblue'])
        node_text.append('<b>'+f"{node_names[node]}"+'</b>')


    node_trace.marker.size = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[*edge_traces, node_trace],
                layout=go.Layout(
                    paper_bgcolor='rgba(255,255,255,0.9)',
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    title='Мережа звʼязків телеграм-каналів Запорізької і Херсонської областей',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    # hovertemplate = ('# of connections: '+str(len(adjacencies[1]))),
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.update_layout(showlegend = False)
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    fig.show()
    fig.write_html("pivden.html")



if __name__ == "__main__":
    Fire(main)
