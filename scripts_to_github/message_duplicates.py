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
    min_group_len = 2

    groups = get_msg_groups(msg_dist, dist_threshold)

    group_len = 0
    for group in filter_by_group_size(groups, 2):
        group_len = group_len + len(group)


    pr_dubl = group_len / len(df) 

    print(pr_dubl)
   

    # Remove duplicates inside channel
    groups = remove_duplicates_inside_channel(groups, df)



    # Remove small groups
    groups = filter_by_group_size(groups, min_group_len)

    df['groups_num'] = None 
    for group_id, group in enumerate(groups):
        for item in group:
            df.at[df.index[item], "groups_num"] = group_id

    top_n = 20
    grouped = df.groupby("channel_name") \
                .apply(lambda d: d["groups_num"].notna().sum())
    
    grouped = pd.DataFrame({"num_of_duplicates": grouped})
    grouped = grouped.sort_values(by="num_of_duplicates", ascending=False)
    grouped = grouped.iloc[:top_n]

    
    print(f"Top {top_n} channels based on num of duplicats:")
    for _, item in grouped.iterrows():
        # print(type(item.name))
        channel_link = item.name.replace(".csv", "")
        num_of_spaces = 25 - len(channel_link)
        print(
            f"\t- https://t.me/" \
            + item.name.replace(".csv", "") \
            + (" " * num_of_spaces) + str(item.num_of_duplicates)
        )

    
    screen_width = 193
    print_groups(groups, df, screen_width)
    print(pr_dubl)
    print(f"Num of groups founded in {data_folder}: {len(groups)}")
    print(f"Num of channels: {len(df.channel_name.unique())}")
    print(f"Message distance treshold: {dist_threshold}")
    print(f"Min elements in group: {min_group_len}")

if __name__ == "__main__":
    Fire(main)
