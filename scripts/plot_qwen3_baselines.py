"""Plot the published Qwen3 time/throughput tables without accessing local raw evidence."""
from pathlib import Path
import argparse,csv,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rl_xy.qwen3_paths import output_path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=ROOT/'experiments/2026-10-29-qwen3')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();OUT=output_path(ROOT,args.output);OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
MODELS=['4B','8B','14B'];PLATFORMS=['gpu','wafer']
def load(name):
 with (args.data/name).open() as f:rows=list(csv.DictReader(f))
 for r in rows:
  for k,v in list(r.items()):
   if k not in ['model','platform','window']:
    try:r[k]=float(v)
    except ValueError:pass
 return rows
summary=load('summary.csv');window_rows=load('window_metrics.csv')
assert len(summary)==6 and len(window_rows)==24

# Publication-sized, editable-text exports. All data are real frozen simulation outputs.
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'axes.linewidth': 0.8, 'lines.linewidth': 1.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'legend.frameon': False, 'savefig.facecolor': 'white',
    'hatch.linewidth': 0.45,
})

def row(model, platform):
    return next(x for x in summary if x['model'] == 'Qwen3-' + model and x['platform'] == platform)

def save_fig(fig, stem):
    fig.savefig(OUT / (stem + '.svg'))
    fig.savefig(OUT / (stem + '.pdf'))
    fig.savefig(OUT / (stem + '.png'), dpi=300)

bar_colors = ['#35CFC5', '#D99DDB', '#FFD8B2']
line_colors = ['#B67F00', '#008C99', '#3963D6']
fig = plt.figure(figsize=(183 / 25.4, 94 / 25.4))
ax = fig.add_axes([0.105, 0.305, 0.785, 0.465]); twin = ax.twinx()
for axis in (ax, twin):
    axis.set_ylim(0, 1.40); axis.set_yticks(np.arange(0, 1.41, 0.2))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    axis.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False); twin.spines['left'].set_visible(False)
ax.set_ylabel('Normalized device-step time ↓', labelpad=5)
twin.set_ylabel('Normalized effective throughput ↑', labelpad=6)
ax.axhline(1, color='#BBC1C8', linewidth=0.7, linestyle=(0, (3, 3)), zorder=0)
handles = []
xall = []
for i, model in enumerate(MODELS):
    xs = np.array([i * 2.2, i * 2.2 + 0.72]); xall.extend(xs)
    records = [row(model, p) for p in PLATFORMS]
    times = [r['time_ratio_to_gpu'] for r in records]
    rates = [r['throughput_ratio_to_gpu'] for r in records]
    bars = ax.bar(xs, times, width=0.48, color=bar_colors[i], edgecolor='#6F7377', linewidth=0.4, zorder=2)
    bars[0].set_hatch('//')
    terr = np.array([[v - r['paired_time_ratio_min'] for v, r in zip(times, records)],
                     [r['paired_time_ratio_max'] - v for v, r in zip(times, records)]])
    qerr = np.array([[v - r['paired_throughput_ratio_min'] for v, r in zip(rates, records)],
                     [r['paired_throughput_ratio_max'] - v for v, r in zip(rates, records)]])
    assert np.all(terr >= -1e-12) and np.all(qerr >= -1e-12)
    ax.errorbar(xs, times, yerr=np.maximum(terr, 0), fmt='none', color='#333333', capsize=2.5, elinewidth=0.7, zorder=4)
    twin.errorbar(xs, rates, yerr=np.maximum(qerr, 0), fmt='-o', color=line_colors[i], capsize=3,
                  markersize=3.5, linewidth=1.6, elinewidth=0.7, barsabove=True, zorder=6)
    for x, value in zip(xs, times):
        ax.text(x, value - 0.10, f'{value:.3f}', ha='center', va='center', fontsize=7.5, fontweight='bold')
    top = records[1]['paired_throughput_ratio_max']
    twin.text(xs[1], top + 0.045, f'{rates[1]:.3f}', ha='center', va='bottom', fontsize=7.5, color=line_colors[i], fontweight='bold')
    ax.text(xs.mean(), -0.19, 'Qwen3-' + model, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8.8, fontweight='bold')
    handles.extend([Patch(facecolor=bar_colors[i], edgecolor='none', label=model + ' time'),
                    Line2D([], [], color=line_colors[i], marker='o', markersize=4, label=model + ' throughput')])
ax.set_xlim(-0.5, 5.62); ax.set_xticks(xall)
ax.set_xticklabels(['GPU', 'Wafer'] * 3)
ax.tick_params(axis='x', length=0, pad=6)
fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.915), columnspacing=2.4,
           handletextpad=0.5, labelspacing=0.55, handlelength=1.7)
fig.text(0.04, 0.974, 'Current Qwen3 baseline comparison', fontsize=9, fontweight='bold', va='top')
fig.text(0.96, 0.974, 'Uncalibrated simulation', fontsize=8, color='#925237', ha='right', va='top')
fig.text(0.5, 0.108, 'GPU = 1 within each model. Bars: time; lines: effective token throughput.', ha='center', fontsize=7)
fig.text(0.5, 0.050, 'Whiskers: range of 4 paired windows, not hardware-model uncertainty.', ha='center', fontsize=7)
save_fig(fig, 'baseline_normalized'); plt.close(fig)

# Absolute values: latency spread and all workload-window throughput observations.
platform_colors = {'gpu': '#74A7D1', 'wafer': '#E7AD76'}
fig = plt.figure(figsize=(183 / 25.4, 90 / 25.4))
axs = [fig.add_axes([0.105, 0.265, 0.355, 0.50]), fig.add_axes([0.595, 0.265, 0.355, 0.50])]
for a in axs:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.set_xticks(range(3)); a.set_xticklabels(['Qwen3-' + m for m in MODELS], fontsize=7.8)
    a.tick_params(axis='x', length=0, pad=7); a.set_xlim(-0.6, 2.6)
    a.yaxis.grid(True, color='#E5E7EB', linewidth=0.55, zorder=0)
axs[0].set_ylim(0, 540); axs[0].set_yticks([0, 100, 200, 300, 400, 500]); axs[0].set_ylabel('Device-step time (s) ↓')
axs[1].set_ylim(0, 30); axs[1].set_yticks([0, 5, 10, 15, 20, 25, 30]); axs[1].set_ylabel('Effective throughput (10³ token/s) ↑')
offsets = {'gpu': -0.19, 'wafer': 0.19}
for mi, model in enumerate(MODELS):
    for pi, platform in enumerate(PLATFORMS):
        r = row(model, platform); x = mi + offsets[platform]
        vals = sorted([v for v in window_rows if v['model'] == 'Qwen3-' + model and v['platform'] == platform], key=lambda v: v['window'])
        time_values = [v['device_step_s'] for v in vals]
        throughput_values = [v['effective_tokens_per_s'] / 1000 for v in vals]
        centers = [r['step_mean_s'], r['pooled_effective_tokens_per_s'] / 1000]
        for ai, a in enumerate(axs):
            a.bar([x], [centers[ai]], width=0.32, color=platform_colors[platform], edgecolor='#61666A', linewidth=0.4, zorder=2, hatch='//' if platform == 'gpu' else None)
            raw_values = time_values if ai == 0 else throughput_values
            a.scatter(x + np.linspace(-0.055, 0.055, 4), raw_values, s=9, facecolors='white', edgecolors='#444444', linewidths=0.5, zorder=4)
            if ai == 0:
                a.errorbar([x], [centers[ai]], yerr=[r['step_sample_sd_s']], fmt='none', color='#222222', capsize=2.4, linewidth=0.7, zorder=5)
                label_y = max(max(raw_values), centers[ai] + r['step_sample_sd_s']) + 11
                label = f'{centers[ai]:.1f}'
            else:
                label_y = max(max(raw_values), centers[ai]) + 0.8
                label = f'{centers[ai]:.2f}'
            a.text(x, label_y, label, ha='center', va='bottom', fontsize=7)
    # Draw paired window differences, never connect different models.
    gr = sorted([v for v in window_rows if v['model'] == 'Qwen3-' + model and v['platform'] == 'gpu'], key=lambda v: v['window'])
    wr = sorted([v for v in window_rows if v['model'] == 'Qwen3-' + model and v['platform'] == 'wafer'], key=lambda v: v['window'])
    for wi, (g, w) in enumerate(zip(gr, wr)):
        shift = np.linspace(-0.055, 0.055, 4)[wi]
        xs = [mi - 0.19 + shift, mi + 0.19 + shift]
        axs[0].plot(xs, [g['device_step_s'], w['device_step_s']], color='#737B83', alpha=0.42, linewidth=0.45, zorder=3)
        axs[1].plot(xs, [g['effective_tokens_per_s'] / 1000, w['effective_tokens_per_s'] / 1000], color='#737B83', alpha=0.42, linewidth=0.45, zorder=3)
axs[0].text(-0.22, 1.07, 'a', transform=axs[0].transAxes, fontsize=10, fontweight='bold')
axs[1].text(-0.22, 1.07, 'b', transform=axs[1].transAxes, fontsize=10, fontweight='bold')
axs[0].set_title('Absolute latency', fontsize=9, loc='left', pad=12)
axs[1].set_title('Absolute throughput', fontsize=9, loc='left', pad=12)
fig.legend(handles=[Patch(facecolor=platform_colors['gpu'], edgecolor='#61666A', hatch='//', label='GPU baseline'), Patch(facecolor=platform_colors['wafer'], edgecolor='#61666A', label='Wafer migration')], loc='upper center', bbox_to_anchor=(0.5, 0.963), ncol=2, columnspacing=2.5, fontsize=8)
fig.text(0.5, 0.135, 'Time bars: mean ± SD. Throughput bars: total tokens / total time. Points: 4 matched windows.', ha='center', fontsize=6.9)
fig.text(0.5, 0.075, 'Effective tokens = prompt + response, counted once per trajectory. 192 prompts × 8 answers per step.', ha='center', fontsize=6.9)
fig.text(0.5, 0.025, 'Current uncalibrated simulation; window variation is not a hardware-model error bound.', ha='center', fontsize=6.9, color='#925237')
save_fig(fig, 'baseline_absolute'); plt.close(fig)

print("Figures written to",OUT)
