import pickle

import pandas as pd
import numpy as np

from pathlib import Path
from fire import Fire
from tqdm.cli import tqdm
# from thefuzz import fuzz
from itertools import combinations

# def folder_loop(folder):
# 	return list(Path(folder).rglob("*.csv"))

def folder_loop(folder):
	for path in Path(folder).iterdir():
		subfolders = [subfolder for subfolder in folder.iterdir() if subfolder.is_dir()]
		return subfolders


def subfolders_proces(subfolder: str): 
	dfs = {fn.name: pd.read_csv(fn) for fn in subfolder.glob("*.csv")}
	for channel_name in dfs:
		dfs[channel_name]["channel_name"] = channel_name
	df = pd.concat(list(dfs.values()))
	df = df.reset_index(drop=True)
	df = df[["text", "channel_name"]]
	counts_ER = df['text'].str.count('Единая Россия').sum()
	counts_SR = df['text'].str.count('Справедливая Россия').sum()
	counts_KPRF = df['text'].str.count('КПРФ').sum()
	counts_NL = df['text'].str.count('Новые люди').sum()
	counts_LDPR = df['text'].str.count('ЛДПР').sum()

	counts_together = {
		'Единая Россия': counts_ER, 
		'Справедливая Россия': counts_SR, 
		'КПРФ': counts_KPRF,
		'Новые люди': counts_NL,
		'ЛДПР' : counts_LDPR,
	}

	counts_together = {k: int(v) for k, v in counts_together.items()}
	return counts_together


folder = Path('sorted')

all_folders = folder_loop(folder)
print(all_folders)

region = {}
for item in all_folders:
	mentions = subfolders_proces(item)
	region[item.stem] = mentions

# print(region)

# for item in region:
# 	print(f'{item.stem} : {str(mentions)}')

del region["out_of_context"]
del region["interesting"]
del region["parties"]

import json
print(json.dumps(region, indent=4, ensure_ascii=False))

region_df = pd.DataFrame.from_dict(region)


region_df = region_df.reset_index()
region_df = region_df.rename(columns={'index': 'Партія'})

print(region_df)




# exit(0)

import plotly.express as px
import matplotlib.pyplot as plt
from plotly.figure_factory import create_table 


fig_l = px.pie(
	region_df, 
	names = 'Партія', 
	values = 'luhansk', 
	title=f'Згадки партій у Телеграмі щодо псевдовиборів у Луганській області',
	labels={'luhansk':'Кількість згадок'},
)

fig_l.update_traces(
	textposition='auto', 
	textinfo='percent+label',
	insidetextorientation='radial',
	rotation=90,
					)
# fig_l.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig_l.show()
fig_l.write_html('plotly_pie_charts/luhansk.html')

# exit(0)

fig_p = px.pie(
	region_df, 
	names = 'Партія', 
	values = 'pivden', 
	title=f'Згадки партій у Телеграмі щодо псевдовиборів у Запорізькій та Херсонській областях',
	labels={'pivden':'Кількість згадок'})
fig_p.update_traces(
	textposition='auto', 
	textinfo='percent+label',
	insidetextorientation='radial',
	rotation=90,
	)
fig_p.show()
fig_p.write_html('plotly_pie_charts/pivden.html')

fig_d = px.pie(
	region_df, 
	names = 'Партія', 
	values = 'donetsk', 
	title=f'Згадки партій у Телеграмі щодо псевдовиборів у Донецькій області',
	labels={'donetsk':'Кількість згадок'},
	)
fig_d.update_traces(
	textposition='inside', 
	textinfo='percent+label',
	rotation=90
	# insidetextorientation='radial',
	)

fig_d.show()
fig_d.write_html('plotly_pie_charts/donetsk.html')


