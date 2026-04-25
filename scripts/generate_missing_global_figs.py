"""
Generate cross-language figures missing from the report:
  1. Global POS distribution of interveners (all 40 languages)
  2. POS distribution grouped by typology (SOV / SVO / VSO / Free)
  3. Global arity distribution (all 40 languages, by typology)
  4. Complexity by POS type (all 40 languages)

All in academic / Big-4 style: clean white background, muted palette,
clear labels, no chart junk.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import os, sys

# ── paths ──────────────────────────────────────────────────────────────────
BASE   = '/Users/kritnandan/Library/CloudStorage/OneDrive-IITKanpur/CGS410'
FEAT   = os.path.join(BASE, 'final_outputs/all_features.csv')
OUT    = os.path.join(BASE, 'plots/global')
os.makedirs(OUT, exist_ok=True)

# ── typology map ────────────────────────────────────────────────────────────
TYPOLOGY = {
    'en':'SVO','fr':'SVO','es':'SVO','pt':'SVO','it':'SVO','de':'SVO',
    'nl':'SVO','sv':'SVO','da':'SVO','no':'SVO','zh':'SVO','id':'SVO',
    'vi':'SVO','th':'SVO','yo':'SVO','wo':'SVO','ca':'SVO','gl':'SVO',
    'hi':'SOV','ja':'SOV','ko':'SOV','tr':'SOV','fa':'SOV','bn':'SOV',
    'ta':'SOV','te':'SOV','mr':'SOV','gu':'SOV','eu':'SOV',
    'ar':'VSO','ga':'VSO','cy':'VSO','tl':'VSO',
    'ru':'Free','pl':'Free','cs':'Free','hu':'Free','fi':'Free','et':'Free',
    'ml':'SOV',
}

# ── style ───────────────────────────────────────────────────────────────────
PALETTE  = {'SVO':'#2E75B6','SOV':'#C00000','VSO':'#70AD47','Free':'#ED7D31'}
GRAY     = '#595959'
LGRAY    = '#D9D9D9'
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   GRAY,
    'xtick.color':      GRAY,
    'ytick.color':      GRAY,
    'figure.dpi':       150,
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.15,
})

# ── load data (sample to keep memory reasonable) ────────────────────────────
print('Loading features …', flush=True)
COLS = ['language','intervener_upos','arity','complexity_score']
df = pd.read_csv(FEAT, usecols=COLS, dtype={'language':str,'intervener_upos':str,
                                             'arity':float,'complexity_score':float})
df['typology'] = df['language'].map(TYPOLOGY).fillna('SVO')
print(f'  Loaded {len(df):,} rows', flush=True)

# canonical POS order
POS_ORDER = ['NOUN','VERB','ADJ','ADV','ADP','DET','PRON','NUM','PROPN',
             'PART','CCONJ','SCONJ','AUX','PUNCT','X','SYM','INTJ','_']

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Global POS distribution of interveners (all 40 languages)
# ─────────────────────────────────────────────────────────────────────────────
print('Figure 1: Global POS distribution …', flush=True)
pos_counts = df['intervener_upos'].value_counts()
pos_pct    = (pos_counts / pos_counts.sum() * 100)
# keep only known POS in canonical order
present = [p for p in POS_ORDER if p in pos_pct.index]
pos_pct = pos_pct.loc[present]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(pos_pct.index, pos_pct.values, color='#2E75B6', edgecolor='white',
              linewidth=0.5, zorder=3)
ax.set_ylabel('Share of all interveners (%)', color=GRAY, fontsize=10)
ax.set_title('Global Intervener POS Distribution — All 40 Languages (24.3 M tokens)',
             fontsize=11, fontweight='bold', color='#1F3864', pad=10)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color=LGRAY, zorder=0)
ax.set_axisbelow(True)
# value labels
for bar in bars:
    h = bar.get_height()
    if h > 0.5:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5, color=GRAY)
ax.set_xlabel('Universal POS Tag', color=GRAY, fontsize=10)
ax.tick_params(axis='x', rotation=30)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'global_pos_distribution.png'))
plt.close(fig)
print('  saved global_pos_distribution.png', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — POS distribution by typology (grouped bars)
# ─────────────────────────────────────────────────────────────────────────────
print('Figure 2: POS by typology …', flush=True)
top_pos = [p for p in POS_ORDER if p in pos_counts.index][:10]  # top 10 only
typo_pos = (df[df['intervener_upos'].isin(top_pos)]
            .groupby(['typology','intervener_upos'])
            .size().reset_index(name='n'))
totals = typo_pos.groupby('typology')['n'].transform('sum')
typo_pos['pct'] = typo_pos['n'] / totals * 100

fig, ax = plt.subplots(figsize=(10, 4.5))
x     = np.arange(len(top_pos))
width = 0.18
typos = ['SVO','SOV','VSO','Free']
for i, typ in enumerate(typos):
    sub = typo_pos[typo_pos['typology'] == typ].set_index('intervener_upos')
    vals = [sub.loc[p,'pct'] if p in sub.index else 0 for p in top_pos]
    ax.bar(x + i*width - 1.5*width, vals, width, label=typ,
           color=PALETTE[typ], edgecolor='white', linewidth=0.4, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(top_pos, rotation=20)
ax.set_ylabel('Share within typology group (%)', color=GRAY, fontsize=10)
ax.set_title('Intervener POS Distribution by Word-Order Typology',
             fontsize=11, fontweight='bold', color='#1F3864', pad=10)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color=LGRAY, zorder=0)
ax.set_axisbelow(True)
ax.legend(title='Typology', framealpha=0.9, fontsize=9, title_fontsize=9)
ax.set_xlabel('Universal POS Tag', color=GRAY, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'pos_by_typology.png'))
plt.close(fig)
print('  saved pos_by_typology.png', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Global arity distribution by typology (stacked histogram)
# ─────────────────────────────────────────────────────────────────────────────
print('Figure 3: Arity by typology …', flush=True)
# cap arity at 6 for readability
df['arity_cap'] = df['arity'].clip(0, 6)

fig, axes = plt.subplots(1, 4, figsize=(11, 3.8), sharey=True)
for ax, typ in zip(axes, typos):
    sub = df[df['typology'] == typ]['arity_cap']
    counts = sub.value_counts().sort_index()
    pct    = counts / counts.sum() * 100
    ax.bar(pct.index, pct.values, color=PALETTE[typ], edgecolor='white',
           linewidth=0.5, zorder=3)
    ax.set_title(typ, fontsize=11, fontweight='bold', color=PALETTE[typ])
    ax.set_xlabel('Arity (# direct dependents)', fontsize=9, color=GRAY)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color=LGRAY, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=8)
    # annotate zero
    if 0 in pct.index:
        ax.text(0, pct[0]+0.5, f'{pct[0]:.0f}%', ha='center', va='bottom',
                fontsize=8, color=GRAY)

axes[0].set_ylabel('Share of interveners (%)', fontsize=9, color=GRAY)
fig.suptitle('Arity Distribution of Interveners by Typological Group — 40 Languages',
             fontsize=11, fontweight='bold', color='#1F3864', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'arity_by_typology.png'))
plt.close(fig)
print('  saved arity_by_typology.png', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Complexity by POS type (all 40 languages, box plot)
# ─────────────────────────────────────────────────────────────────────────────
print('Figure 4: Complexity by POS …', flush=True)
top8 = [p for p in POS_ORDER if p in pos_counts.index][:8]
data_by_pos = [df.loc[df['intervener_upos']==p, 'complexity_score'].dropna().sample(
                min(50000, (df['intervener_upos']==p).sum()), random_state=42).values
               for p in top8]

fig, ax = plt.subplots(figsize=(10, 4))
bp = ax.boxplot(data_by_pos, labels=top8, patch_artist=True,
                medianprops=dict(color='white', linewidth=1.8),
                whiskerprops=dict(color=GRAY, linewidth=0.8),
                capprops=dict(color=GRAY, linewidth=0.8),
                flierprops=dict(marker='.', color=LGRAY, markersize=2, alpha=0.3),
                boxprops=dict(linewidth=0.8))
colors = ['#2E75B6','#C00000','#70AD47','#ED7D31','#7030A0',
          '#00B0F0','#FF0000','#92D050']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_ylabel('Complexity Score C(w)', color=GRAY, fontsize=10)
ax.set_title('Intervener Complexity by POS Type — All 40 Languages',
             fontsize=11, fontweight='bold', color='#1F3864', pad=10)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color=LGRAY, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel('Universal POS Tag', color=GRAY, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'complexity_by_pos_global.png'))
plt.close(fig)
print('  saved complexity_by_pos_global.png', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Arity per language (bar chart, sorted)
# ─────────────────────────────────────────────────────────────────────────────
print('Figure 5: Avg arity per language …', flush=True)
lang_arity = (df.groupby('language')['arity']
                .mean()
                .sort_values()
                .reset_index())
lang_arity['typology'] = lang_arity['language'].map(TYPOLOGY).fillna('SVO')

fig, ax = plt.subplots(figsize=(12, 4))
bar_colors = [PALETTE[t] for t in lang_arity['typology']]
ax.bar(lang_arity['language'], lang_arity['arity'], color=bar_colors,
       edgecolor='white', linewidth=0.4, zorder=3)
ax.set_ylabel('Mean Arity', color=GRAY, fontsize=10)
ax.set_title('Mean Intervener Arity per Language (All 40 Languages)',
             fontsize=11, fontweight='bold', color='#1F3864', pad=10)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color=LGRAY, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis='x', rotation=60, labelsize=8)
legend_handles = [mpatches.Patch(color=PALETTE[t], label=t) for t in typos]
ax.legend(handles=legend_handles, title='Typology', fontsize=9, title_fontsize=9,
          framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'arity_per_language.png'))
plt.close(fig)
print('  saved arity_per_language.png', flush=True)

print('\nAll figures saved to:', OUT)
