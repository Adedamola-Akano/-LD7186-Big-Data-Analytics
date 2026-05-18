"""
================================================================
LD7186 Big Data Analytics — Section 1
Connecticut Accidental Drug-Related Deaths, 2012-2024

Student : Damola
Approach: Object-oriented surveillance pipeline.
          One OverdoseSurveillance class encapsulates each
          analytical step (load, clean, describe, plot, test).
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Damola colour palette — earthy greens / teals / coral / mustard
# ----------------------------------------------------------------
PALETTE = {
    "primary"   : "#2E7D5B",
    "accent"    : "#E07856",
    "highlight" : "#D4A24C",
    "deep"      : "#1B4332",
    "soft"      : "#A8DADC",
    "male"      : "#52796F",
    "female"    : "#E29578",
}

plt.rcParams["font.family"]  = "DejaVu Serif"
plt.rcParams["figure.dpi"]   = 110
plt.rcParams["savefig.dpi"]  = 150
plt.rcParams["savefig.bbox"] = "tight"
sns.set_style("white")


# ================================================================
#  Surveillance pipeline class
# ================================================================
class OverdoseSurveillance:
    """End-to-end surveillance analysis for the CT OCME dataset."""

    SUBSTANCES = [
        "Heroin", "Cocaine", "Fentanyl", "Fentanyl Analogue",
        "Oxycodone", "Oxymorphone", "Ethanol", "Hydrocodone",
        "Benzodiazepine", "Methadone", "Meth/Amphetamine",
        "Amphet", "Tramad", "Hydromorphone",
        "Morphine (Not Heroin)", "Xylazine", "Gabapentin",
        "Opiate NOS", "Heroin/Morph/Codeine",
        "Other Opioid", "Any Opioid",
    ]

    AGE_BANDS  = [0, 25, 35, 45, 55, 65, 100]
    AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

    # ------------------------------------------------------------
    def __init__(self, csv_path, out_dir="damola_figs", alpha=0.05):
        self.csv_path   = csv_path
        self.out_dir    = out_dir
        self.alpha      = alpha
        self.raw        = None
        self.df         = None
        self.crosstab   = None
        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------
    def load(self):
        print("=" * 70)
        print(" STEP 1 — load OCME data ".center(70, "="))
        print("=" * 70)
        self.raw = pd.read_csv(self.csv_path)
        print(f"  records  : {self.raw.shape[0]:,}")
        print(f"  variables: {self.raw.shape[1]}")
        return self

    # ------------------------------------------------------------
    # 2. EDA
    # ------------------------------------------------------------
    def explore(self):
        print("\n" + "=" * 70)
        print(" STEP 2 — exploratory screen ".center(70, "="))
        print("=" * 70)
        print("\n[head]")
        print(self.raw.head())
        print("\n[dtypes]")
        print(self.raw.dtypes)
        print("\n[describe]")
        print(self.raw.describe())
        print("\n[missingness top-15]")
        print(self.raw.isnull().sum()
                  .sort_values(ascending=False).head(15))
        return self

    # ------------------------------------------------------------
    # 3. Pre-processing
    # ------------------------------------------------------------
    def preprocess(self):
        print("\n" + "=" * 70)
        print(" STEP 3 — pre-process ".center(70, "="))
        print("=" * 70)

        df = self.raw.copy()
        df["Date"]      = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"]      = df["Date"].dt.year
        df["Month"]     = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.day_name()

        for sub in self.SUBSTANCES:
            if sub in df.columns:
                df[sub] = (df[sub].astype(str).str.strip()
                              .str.upper() == "Y").astype(int)

        df["Age"]  = pd.to_numeric(df["Age"], errors="coerce")
        df["Sex"]  = df["Sex"].astype(str).str.strip().str.title()
        df.loc[~df["Sex"].isin(["Male", "Female"]), "Sex"] = np.nan
        df["Race"] = df["Race"].astype(str).str.strip()

        before = len(df)
        df = df.dropna(subset=["Year", "Age", "Sex"]).copy()
        df["Year"] = df["Year"].astype(int)

        df["NumSubstances"] = df[[s for s in self.SUBSTANCES
                                  if s != "Any Opioid"]].sum(axis=1)
        df["AgeBand"] = pd.cut(df["Age"], bins=self.AGE_BANDS,
                                labels=self.AGE_LABELS, right=False)

        print(f"  before clean: {before:,}")
        print(f"  after  clean: {len(df):,}")
        print(f"  excluded    : {before-len(df):,} "
              f"({100*(before-len(df))/before:.2f}%)")

        self.df = df
        return self

    # ------------------------------------------------------------
    def _save(self, name):
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, name))
        plt.close()

    # ------------------------------------------------------------
    # 4. Descriptive summary
    # ------------------------------------------------------------
    def describe(self):
        print("\n" + "=" * 70)
        print(" STEP 4 — descriptive summary ".center(70, "="))
        print("=" * 70)
        df = self.df
        print(f"  mean age : {df['Age'].mean():.2f}")
        print(f"  median   : {df['Age'].median():.2f}")
        print(f"  std dev  : {df['Age'].std():.2f}")
        print(f"  range    : {df['Age'].min()}-{df['Age'].max()}")
        print("\n  sex split:")
        print(df["Sex"].value_counts())
        print(f"\n  polysubstance (>=2): "
              f"{(df['NumSubstances']>=2).sum():,} "
              f"({(df['NumSubstances']>=2).mean()*100:.1f}%)")
        return self

    # ============================================================
    # RQ1 — annual volume + year-on-year %
    # ============================================================
    def rq1_annual_trend(self):
        print("\n--- RQ1: annual volume + growth rate -------------")
        yearly = self.df.groupby("Year").size()
        pct    = yearly.pct_change() * 100

        fig, ax1 = plt.subplots(figsize=(11, 5.5))
        ax1.bar(yearly.index, yearly.values,
                color=PALETTE["primary"], alpha=0.7,
                edgecolor=PALETTE["deep"], linewidth=1.2,
                label="Total deaths")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Number of deaths", color=PALETTE["deep"])
        ax1.tick_params(axis="y", labelcolor=PALETTE["deep"])
        ax1.set_xticks(yearly.index)
        ax1.spines["top"].set_visible(False)
        for i, v in enumerate(yearly.values):
            ax1.text(yearly.index[i], v + 25, str(v),
                     ha="center", fontsize=8, color=PALETTE["deep"])

        ax2 = ax1.twinx()
        ax2.plot(pct.index, pct.values,
                 color=PALETTE["accent"], marker="D",
                 markersize=8, linewidth=2.5,
                 label="Year-on-year % change")
        ax2.axhline(0, color="grey", linestyle=":", linewidth=1)
        ax2.set_ylabel("Year-on-year change (%)",
                       color=PALETTE["accent"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["accent"])
        ax2.spines["top"].set_visible(False)
        for x, v in zip(pct.index, pct.values):
            if not np.isnan(v):
                ax2.annotate(f"{v:+.0f}%", xy=(x, v), xytext=(0, 8),
                             textcoords="offset points",
                             ha="center", fontsize=8,
                             color=PALETTE["accent"],
                             fontweight="bold")
        ax1.set_title("Annual Mortality Volume and Growth Rate, "
                      "2012-2024",
                      fontsize=13, fontweight="bold",
                      color=PALETTE["deep"], pad=15)
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.95),
                   frameon=True, edgecolor=PALETTE["deep"])
        self._save("01_deaths_by_year.png")
        return yearly, pct

    # ============================================================
    # RQ2a — fentanyl vs non-fentanyl stacked
    # ============================================================
    def rq2_fentanyl_share(self):
        print("\n--- RQ2: fentanyl share over time ----------------")
        split = (self.df.groupby(["Year", "Fentanyl"])
                     .size().unstack(fill_value=0))
        split.columns = ["No fentanyl", "Fentanyl-involved"]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        split.plot(kind="bar", stacked=True, ax=ax,
                   color=[PALETTE["soft"], PALETTE["accent"]],
                   edgecolor=PALETTE["deep"], linewidth=0.8,
                   width=0.75)
        ax.set_title("Fentanyl-Involved vs Non-Fentanyl Deaths, "
                     "2012-2024",
                     fontsize=13, fontweight="bold",
                     color=PALETTE["deep"], pad=15)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of deaths")
        ax.legend(loc="upper left", frameon=True,
                  edgecolor=PALETTE["deep"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=0)

        totals = split.sum(axis=1)
        share  = split["Fentanyl-involved"] / totals * 100
        for i, y in enumerate(split.index):
            ax.text(i, totals[y] + 30, f"{share[y]:.0f}%",
                    ha="center", fontsize=8,
                    color=PALETTE["deep"], fontweight="bold")
        self._save("02_top_substances.png")
        return split

    # ============================================================
    # RQ2b — polysubstance count distribution
    # ============================================================
    def rq2_polysubstance(self):
        print("\n--- RQ2: polysubstance distribution --------------")
        poly = self.df["NumSubstances"].value_counts().sort_index()
        print(poly)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [PALETTE["soft"]      if n <= 1
                  else PALETTE["highlight"] if n <= 3
                  else PALETTE["accent"]
                  for n in poly.index]
        ax.bar(poly.index, poly.values, color=colors,
               edgecolor=PALETTE["deep"], linewidth=1.2)
        for n, v in zip(poly.index, poly.values):
            ax.text(n, v + 50, f"{v:,}",
                    ha="center", fontsize=9,
                    color=PALETTE["deep"], fontweight="bold")
        ax.set_title("Polysubstance Profile — Substances Detected "
                     "per Death",
                     fontsize=13, fontweight="bold",
                     color=PALETTE["deep"], pad=15)
        ax.set_xlabel("Substances detected per death")
        ax.set_ylabel("Number of deaths")
        ax.set_xticks(poly.index)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=[
            Patch(facecolor=PALETTE["soft"],
                  edgecolor=PALETTE["deep"],
                  label="1 substance (mono)"),
            Patch(facecolor=PALETTE["highlight"],
                  edgecolor=PALETTE["deep"],
                  label="2-3 substances"),
            Patch(facecolor=PALETTE["accent"],
                  edgecolor=PALETTE["deep"],
                  label="4+ substances (high-risk poly)")
        ], loc="upper right", frameon=True,
           edgecolor=PALETTE["deep"])
        self._save("03_top3_trend.png")
        return poly

    # ============================================================
    # RQ3 — age density + mean-age drift + race + age band sex
    # ============================================================
    def rq3_age_profile(self):
        print("\n--- RQ3: age profile + age drift -----------------")
        df = self.df

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for sex, c in [("Male", PALETTE["male"]),
                       ("Female", PALETTE["female"])]:
            ages = df.loc[df["Sex"] == sex, "Age"].dropna()
            xs   = np.linspace(15, 90, 300)
            kde  = gaussian_kde(ages)
            axes[0].fill_between(xs, kde(xs), color=c,
                                  alpha=0.4, label=sex)
            axes[0].plot(xs, kde(xs), color=c, linewidth=2)
        axes[0].set_title("Age Density by Sex",
                          fontweight="bold", color=PALETTE["deep"])
        axes[0].set_xlabel("Age")
        axes[0].set_ylabel("Density")
        axes[0].legend(frameon=True, edgecolor=PALETTE["deep"])
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        agg = df.groupby("Year")["Age"].agg(["mean","median","std"])
        axes[1].plot(agg.index, agg["mean"],
                      color=PALETTE["deep"], marker="o",
                      linewidth=2.5, markersize=8,
                      label="Mean age")
        axes[1].plot(agg.index, agg["median"],
                      color=PALETTE["accent"], marker="s",
                      linewidth=2, linestyle="--",
                      label="Median age")
        axes[1].fill_between(agg.index,
                              agg["mean"] - agg["std"],
                              agg["mean"] + agg["std"],
                              color=PALETTE["primary"],
                              alpha=0.15, label="\u00b11 SD")
        axes[1].set_title("Has the Typical Victim Aged?",
                          fontweight="bold", color=PALETTE["deep"])
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Age")
        axes[1].set_xticks(agg.index)
        axes[1].tick_params(axis="x", labelrotation=45)
        axes[1].legend(frameon=True, edgecolor=PALETTE["deep"])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        self._save("04_age_sex.png")

        # Race × year stacked %
        df_r = df.copy()
        df_r["RaceGroup"] = df_r["Race"].replace({
            "Black or African American": "Black",
            "Black":                     "Black",
            "white":                     "White",
        })
        df_r.loc[~df_r["RaceGroup"].isin(
            ["White", "Black"]), "RaceGroup"] = "Other"
        race_year = (pd.crosstab(df_r["Year"], df_r["RaceGroup"],
                                  normalize="index")
                       * 100)[["White","Black","Other"]]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        race_year.plot(kind="bar", stacked=True, ax=ax,
                        color=[PALETTE["primary"],
                               PALETTE["accent"],
                               PALETTE["soft"]],
                        edgecolor=PALETTE["deep"],
                        linewidth=0.6, width=0.85)
        ax.set_title("Racial Composition of Decedents Over "
                      "Time (% by year)",
                      fontsize=13, fontweight="bold",
                      color=PALETTE["deep"], pad=15)
        ax.set_xlabel("Year")
        ax.set_ylabel("Share of deaths (%)")
        ax.legend(title="Race group", loc="upper center",
                  bbox_to_anchor=(0.5, -0.15), ncol=3,
                  frameon=True, edgecolor=PALETTE["deep"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=0)
        self._save("08_race.png")

        # Age band × sex
        age_sex     = pd.crosstab(df["AgeBand"], df["Sex"])
        age_sex_pct = age_sex.div(age_sex.sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        age_sex.plot(kind="barh", stacked=True, ax=axes[0],
                     color=[PALETTE["female"], PALETTE["male"]],
                     edgecolor=PALETTE["deep"], linewidth=1.2)
        axes[0].invert_yaxis()
        axes[0].set_title("Counts by Age Band and Sex",
                          fontweight="bold", color=PALETTE["deep"])
        axes[0].set_xlabel("Number of deaths")
        axes[0].set_ylabel("Age Band")
        axes[0].legend(loc="lower right", frameon=True,
                        edgecolor=PALETTE["deep"])
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        age_sex_pct.plot(kind="barh", stacked=True, ax=axes[1],
                          color=[PALETTE["female"],
                                 PALETTE["male"]],
                          edgecolor=PALETTE["deep"], linewidth=1.2)
        axes[1].invert_yaxis()
        axes[1].set_title("Sex Composition Within Each Age Band (%)",
                          fontweight="bold", color=PALETTE["deep"])
        axes[1].set_xlabel("Share within age band (%)")
        axes[1].set_ylabel("")
        axes[1].set_xlim(0, 100)
        axes[1].legend(loc="lower right", frameon=True,
                        edgecolor=PALETTE["deep"])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        for i, ab in enumerate(age_sex_pct.index):
            f_pct = age_sex_pct.loc[ab, "Female"]
            m_pct = age_sex_pct.loc[ab, "Male"]
            axes[1].text(f_pct/2, i, f"{f_pct:.0f}%",
                          ha="center", va="center", color="white",
                          fontweight="bold", fontsize=10)
            axes[1].text(f_pct + m_pct/2, i, f"{m_pct:.0f}%",
                          ha="center", va="center", color="white",
                          fontweight="bold", fontsize=10)
        self._save("09_age_groups.png")
        return self

    # ============================================================
    # RQ4 — fentanyl involvement by age band
    # ============================================================
    def rq4_fentanyl_by_age(self):
        print("\n--- RQ4: fentanyl by age band --------------------")
        fa = self.df.groupby("AgeBand")["Fentanyl"].agg(
            ["sum","count"])
        fa["pct"]    = fa["sum"] / fa["count"] * 100
        fa["nofent"] = fa["count"] - fa["sum"]
        print(fa)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(fa))
        ax.bar(x, fa["nofent"], color=PALETTE["soft"],
                edgecolor=PALETTE["deep"], linewidth=1.2,
                label="No fentanyl")
        ax.bar(x, fa["sum"], bottom=fa["nofent"],
                color=PALETTE["accent"],
                edgecolor=PALETTE["deep"], linewidth=1.2,
                label="Fentanyl-involved")
        for i, (lab, row) in enumerate(fa.iterrows()):
            ax.text(i, row["count"] + 30,
                    f"{row['pct']:.0f}%", ha="center",
                    fontsize=10, color=PALETTE["deep"],
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(fa.index)
        ax.set_title("Fentanyl Involvement by Age Band "
                      "(with share annotated)",
                      fontsize=13, fontweight="bold",
                      color=PALETTE["deep"], pad=15)
        ax.set_xlabel("Age Band")
        ax.set_ylabel("Number of deaths")
        ax.legend(loc="upper right", frameon=True,
                  edgecolor=PALETTE["deep"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._save("05_fentanyl_sex.png")

        self.crosstab = pd.crosstab(self.df["Sex"],
                                     self.df["Fentanyl"])
        return fa

    # ============================================================
    # Substances co-detected with fentanyl
    # ============================================================
    def co_occurrence(self):
        print("\n--- co-occurrence with fentanyl ------------------")
        fent_deaths = self.df[self.df["Fentanyl"] == 1]
        cols   = [s for s in self.SUBSTANCES
                   if s not in ("Fentanyl", "Any Opioid")]
        counts = (fent_deaths[cols].sum()
                      .sort_values(ascending=False).head(12))
        pct    = counts / len(fent_deaths) * 100
        print(pct.head())

        fig, ax = plt.subplots(figsize=(10, 6.5))
        bars = ax.barh(pct.index, pct.values,
                       color=PALETTE["accent"],
                       edgecolor=PALETTE["deep"], linewidth=1.2)
        for i in range(3):
            bars[i].set_color(PALETTE["deep"])
        ax.invert_yaxis()
        for i, n in enumerate(pct.index):
            ax.text(pct[n] + 0.5, i,
                    f"{pct[n]:.1f}%  (n={counts[n]:,})",
                    va="center", fontsize=9,
                    color=PALETTE["deep"], fontweight="bold")
        ax.set_title(f"Substances Co-Detected with Fentanyl "
                      f"({len(fent_deaths):,} fentanyl deaths)",
                      fontsize=13, fontweight="bold",
                      color=PALETTE["deep"], pad=15)
        ax.set_xlabel("% of fentanyl deaths where this "
                       "substance also present")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")
        self._save("06_corr_heatmap.png")
        return pct

    # ============================================================
    # Day-of-week
    # ============================================================
    def day_of_week(self):
        print("\n--- day-of-week ----------------------------------")
        order = ["Monday","Tuesday","Wednesday","Thursday",
                 "Friday","Saturday","Sunday"]
        dow     = self.df["DayOfWeek"].value_counts().reindex(order)
        dow_pct = dow / dow.sum() * 100
        print(dow)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        colors = [PALETTE["primary"]
                   if d not in ("Saturday","Sunday")
                   else PALETTE["accent"] for d in order]
        ax.bar(dow.index, dow.values, color=colors,
                edgecolor=PALETTE["deep"], linewidth=1.2)
        for d in dow.index:
            ax.text(d, dow[d] + 30,
                    f"{dow[d]:,}\n({dow_pct[d]:.1f}%)",
                    ha="center", fontsize=9,
                    color=PALETTE["deep"], fontweight="bold")
        mean_per_day = dow.mean()
        ax.axhline(mean_per_day, color=PALETTE["deep"],
                    linestyle="--", linewidth=1)
        ax.set_title("Deaths by Day of Week — "
                      "Are Weekends Higher-Risk?",
                      fontsize=13, fontweight="bold",
                      color=PALETTE["deep"], pad=15)
        ax.set_xlabel("Day of week")
        ax.set_ylabel("Number of deaths")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=[
            Patch(facecolor=PALETTE["primary"],
                  edgecolor=PALETTE["deep"], label="Weekday"),
            Patch(facecolor=PALETTE["accent"],
                  edgecolor=PALETTE["deep"], label="Weekend"),
            plt.Line2D([0],[0], color=PALETTE["deep"],
                        linestyle="--",
                        label=f"Weekly mean ({mean_per_day:.0f})")
        ], loc="upper center", bbox_to_anchor=(0.5, -0.15),
            ncol=3, frameon=True, edgecolor=PALETTE["deep"])
        self._save("07_year_month_heatmap.png")
        return dow

    # ============================================================
    # H1 — Welch's t-test on age by sex
    # ============================================================
    def hypothesis_age_by_sex(self):
        print("\n" + "=" * 70)
        print(" H1 — Welch's t-test, mean age vs sex ".center(70, "="))
        print("=" * 70)
        print("  H0: no difference   |  H1: difference exists")

        m = self.df.loc[self.df["Sex"] == "Male",   "Age"].dropna()
        f = self.df.loc[self.df["Sex"] == "Female", "Age"].dropna()
        print(f"  Male  : n={len(m):,}, "
              f"mean={m.mean():.2f}, sd={m.std():.2f}")
        print(f"  Female: n={len(f):,}, "
              f"mean={f.mean():.2f}, sd={f.std():.2f}")

        t, p = stats.ttest_ind(m, f, equal_var=False)
        decision = ("REJECT H0" if p < self.alpha
                    else "FAIL to reject H0")
        print(f"  t={t:.4f}, p={p:.6f} -> {decision}")
        return {"t": t, "p": p, "decision": decision}

    # ============================================================
    # H2 — chi-square sex × fentanyl
    # ============================================================
    def hypothesis_chi_square(self):
        print("\n" + "=" * 70)
        print(" H2 — chi-square, sex x fentanyl ".center(70, "="))
        print("=" * 70)
        print("  H0: independent     |  H1: associated")

        chi2, p, dof, _ = stats.chi2_contingency(self.crosstab)
        n = self.crosstab.values.sum()
        v = np.sqrt(chi2 / (n * (min(self.crosstab.shape) - 1)))
        decision = ("REJECT H0" if p < self.alpha
                    else "FAIL to reject H0")
        print(f"  chi-square = {chi2:.4f}")
        print(f"  df         = {dof}")
        print(f"  p-value    = {p:.6f}")
        print(f"  Cramer's V = {v:.4f}")
        print(f"  decision   : {decision}")
        return {"chi2": chi2, "p": p, "dof": dof,
                "cramers_v": v, "decision": decision}


# ================================================================
#  Run the surveillance pipeline
# ================================================================
if __name__ == "__main__":
    surveillance = (
        OverdoseSurveillance(
            csv_path="Accidental_Drug_Related_Deaths_2012-2024.csv",
            out_dir="damola_figs")
        .load()
        .explore()
        .preprocess()
        .describe()
    )

    surveillance.rq1_annual_trend()
    surveillance.rq2_fentanyl_share()
    surveillance.rq2_polysubstance()
    surveillance.rq3_age_profile()
    surveillance.rq4_fentanyl_by_age()
    surveillance.co_occurrence()
    surveillance.day_of_week()
    surveillance.hypothesis_age_by_sex()
    surveillance.hypothesis_chi_square()

    print("\n" + "=" * 70)
    print(" pipeline complete — see folder: damola_figs ".center(70, "="))
    print("=" * 70)
