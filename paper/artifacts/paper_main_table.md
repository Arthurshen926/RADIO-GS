# RADIO-GS: Main Results Table (LERF-OVS)
*Auto-generated. Last updated: 2026-04-23 14:47*

## Main Table: LocAcc (n=3 seeds, mean ± std)

| Scene | noFDH | FDH-WS240 (Ours) | Δ |
|---|---|---|---|
| Figurines | 0.7381 ± 0.0372 | **0.7738 ± 0.0273** | +0.0357 |
| Ramen | 0.8592 ± 0.0373 | **0.8779 ± 0.0081** | +0.0188 |
| Teatime | 0.8701 ± 0.0196 | **0.8757 ± 0.0259** | +0.0056 |
| Waldo Kitchen | 0.6515 ± 0.1461 | **0.7121 ± 0.1050** | +0.0606 |

**Macro Average (4 scenes):** noFDH = 0.7797 | FDH-WS240 = **0.8099** | Δ = +0.0302

## Per-Seed Breakdown

### Figurines

| Method | Seed 42 | Seed 7 | Seed 123 | Mean | Std | N |
|---|---|---|---|---|---|---|
| noFDH | 0.7500 | 0.6964 | 0.7679 | 0.7381 | 0.0372 | 3/3 |
| FDH-WS240 | 0.8036 | 0.7679 | 0.7500 | 0.7738 | 0.0273 | 3/3 |

### Ramen

| Method | Seed 42 | Seed 7 | Seed 123 | Mean | Std | N |
|---|---|---|---|---|---|---|
| noFDH | 0.9014 | 0.8451 | 0.8310 | 0.8592 | 0.0373 | 3/3 |
| FDH-WS240 | 0.8732 | 0.8873 | 0.8732 | 0.8779 | 0.0081 | 3/3 |

### Teatime

| Method | Seed 42 | Seed 7 | Seed 123 | Mean | Std | N |
|---|---|---|---|---|---|---|
| noFDH | 0.8814 | 0.8475 | 0.8814 | 0.8701 | 0.0196 | 3/3 |
| FDH-WS240 | 0.8814 | 0.8983 | 0.8475 | 0.8757 | 0.0259 | 3/3 |

### Waldo Kitchen

| Method | Seed 42 | Seed 7 | Seed 123 | Mean | Std | N |
|---|---|---|---|---|---|---|
| noFDH | 0.5909 | 0.8182 | 0.5455 | 0.6515 | 0.1461 | 3/3 |
| FDH-WS240 | 0.7727 | 0.5909 | 0.7727 | 0.7121 | 0.1050 | 3/3 |

## LaTeX Table (for paper)

```latex
\begin{table}[t]
\centering
\caption{LERF-OVS Localization Accuracy (LocAcc). Results are mean $\pm$ std over $n=3$ random seeds.}
\label{tab:main_results}
\begin{tabular}{lcc}
\toprule
Scene & noFDH & FDH-WS240 (Ours) \\
\midrule
Figurines & 0.738 $\pm$ 0.037 & \textbf{0.774 $\pm$ 0.027} \\
Ramen & 0.859 $\pm$ 0.037 & \textbf{0.878 $\pm$ 0.008} \\
Teatime & 0.870 $\pm$ 0.020 & \textbf{0.876 $\pm$ 0.026} \\
Waldo Kitchen & 0.652 $\pm$ 0.146 & \textbf{0.712 $\pm$ 0.105} \\
\midrule
Macro Avg & 0.780 & \textbf{0.810} \\
\bottomrule
\end{tabular}
\end{table}
```

## Completeness: 24/24 runs have eval results

- ✅ figurines/nofdh: seeds done=[42, 7, 123], pending=[]
- ✅ figurines/fdh_ws240: seeds done=[42, 7, 123], pending=[]
- ✅ ramen/nofdh: seeds done=[42, 7, 123], pending=[]
- ✅ ramen/fdh_ws240: seeds done=[42, 7, 123], pending=[]
- ✅ teatime/nofdh: seeds done=[42, 7, 123], pending=[]
- ✅ teatime/fdh_ws240: seeds done=[42, 7, 123], pending=[]
- ✅ waldo_kitchen/nofdh: seeds done=[42, 7, 123], pending=[]
- ✅ waldo_kitchen/fdh_ws240: seeds done=[42, 7, 123], pending=[]
