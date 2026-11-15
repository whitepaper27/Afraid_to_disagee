import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = sys.argv[1]; O = sys.argv[2]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"neutral":"Neutral","threat":"Threat","reward":"Reward","urgency":"Urgency","praise":"Praise",
         "control_nosuggest":"No suggestion\n(threat only)","authority":"Authority*","boss_deadline":"Boss+deadline*"}
ORDER = ["control_nosuggest","neutral","threat","reward","urgency","boss_deadline","praise","authority"]

# Fig 1: behaviour
r = pd.read_csv(f"{T}/rates_none.csv").set_index("condition").loc[ORDER]
x = np.arange(len(r)); fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.bar(x-0.2, r.bad_flip, 0.4, yerr=[r.bad_flip-r.bad_lo, r.bad_hi-r.bad_flip], capsize=3, color="#c0392b", label="Abandons a correct answer (bad flip)")
ax.bar(x+0.2, r.good_update, 0.4, yerr=[r.good_update-r.good_lo, r.good_hi-r.good_update], capsize=3, color="#2e86c1", label="Fixes a wrong answer (good update)")
ax.axhline(r.loc["neutral","bad_flip"], color="#c0392b", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels([NAMES[c] for c in r.index], fontsize=8.5)
ax.set_ylabel("Rate (95% bootstrap CI)"); ax.set_ylim(0,1.12); ax.set_yticks([0,.2,.4,.6,.8,1]); ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
ax.set_title("Pushback outcomes by pressure type (Qwen2.5-3B, 596 MMLU items)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{O}/fig1_behaviour.png", dpi=220); plt.close(fig)

# Fig 2: the dial (removal effects on bad flips AND good updates)
m = pd.read_csv(f"{T}/mediation.csv")
conds = ["threat","reward","urgency","praise","boss_deadline"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, kind, title in ((axes[0],"bad","Reduction in abandoning correct answers"),(axes[1],"good","Reduction in fixing wrong answers")):
    d = m[(m.kind==kind)].set_index("condition").loc[conds]; xx = np.arange(len(conds)); w=0.27
    for k,(col,lab,colr) in enumerate((("d_emo","Matched emotion removed","#8e44ad"),("d_mis","Opposite-valence emotion removed","#f39c12"),("d_rand","Random direction removed (10 seeds)","#7f8c8d"))):
        y=d[col]*100; lo=(d[col]-d[f"{col}_lo"])*100; hi=(d[f"{col}_hi"]-d[col])*100
        ax.bar(xx+(k-1)*w, y, w, yerr=[lo,hi], capsize=2, color=colr, label=lab)
    ax.axhline(0,color="k",lw=0.8); ax.set_xticks(xx); ax.set_xticklabels([NAMES[c].replace("*","") for c in conds], fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
axes[0].set_ylabel("Percentage points (vs. no intervention)"); axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("Removing the matched emotion direction makes the model harder to move in BOTH directions", fontsize=10)
fig.tight_layout(); fig.savefig(f"{O}/fig2_dial.png", dpi=220); plt.close(fig)

# Fig 3: addition vs random-direction control (neutral pushback)
c = pd.read_csv(f"{T}/controls_extra.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, kind, title in ((axes[0],"bad","Change in abandoning correct answers"),(axes[1],"good","Change in fixing wrong answers")):
    cc = c[c.kind==kind]
    rnd = cc[cc.spec.str.startswith("addrand_")].copy()
    rnd["alpha"] = rnd.spec.str.extract(r"_a([0-9.]+)$").astype(float)
    for a, g in rnd.groupby("alpha"):
        ax.scatter([a]*len(g), g.delta*100, color="#7f8c8d", marker="x", s=30, zorder=3, label="Random directions (5 seeds)" if a==0.05 else None)
    emo = cc[cc.spec.str.startswith("add_")].copy()
    emo["emotion"] = emo.spec.str.split("_").str[1]; emo["alpha"] = emo.spec.str.extract(r"_a([0-9.]+)$").astype(float)
    for lab, g in emo.groupby("emotion"):
        g = g.sort_values("alpha"); ax.plot(g.alpha, g.delta*100, marker="o", label=lab)
    ax.axhline(0,color="k",lw=0.8); ax.set_xlabel("Steering strength α (fraction of activation norm)"); ax.set_title(title, fontsize=9.5)
    ax.set_xticks([0.02,0.05,0.1])
axes[0].set_ylabel("Percentage points (vs. neutral pushback)"); h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=8,loc="lower center",ncol=6,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Adding emotion directions to neutral pushback vs. adding random directions", fontsize=10)
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig(f"{O}/fig3_addition.png", dpi=220); plt.close(fig)

# summary numbers for the text
rnd = c[c.spec.str.startswith("addrand_")].copy(); rnd["alpha"]=rnd.spec.str.extract(r"_a([0-9.]+)$").astype(float)
print(rnd.groupby(["alpha","kind"]).delta.agg(["mean","min","max"]).round(3))
