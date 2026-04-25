"""
Hypothesis testing: ANOVA, Mann-Whitney U, Cohen's d, chi-square, z-scores.
Mixed-effects models via statsmodels.

AUDIT FIXES (2026-04-24):
- Added per_sentence_zscores_with_ci(): per-sentence z-scores with bootstrap 95% CI
  (Futrell et al. 2015 method — avoids corpus-level aggregation bias). Fix 13/17.
- Added ancova(): ANCOVA controlling for a covariate, for Dravidian word-order control. Fix 14.
- Added power_analysis_n40(): post-hoc power analysis for n=40 language study. Fix 18.
"""

from __future__ import annotations

import logging

import math

import warnings

from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

import pandas as pd

if TYPE_CHECKING:

    from src.baselines.random_trees import RandomTreeBaseline

    from src.data.loader import Sentence

logger = logging.getLogger(__name__)

def z_score(

    real_value: float,

    random_mean: float,

    random_std: float,

) -> float:

    if random_std == 0:

        return float("nan")

    return (real_value - random_mean) / random_std

def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:

    """Cohen's d effect size between two groups."""

    na, nb = len(a), len(b)

    if na < 2 or nb < 2:

        return float("nan")

    mean_a, mean_b = np.mean(a), np.mean(b)

    pooled_std = math.sqrt(

        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)

    )

    if pooled_std == 0:

        return float("nan")

    return (mean_a - mean_b) / pooled_std

class HypothesisTester:

    @staticmethod

    def anova(*groups: Sequence[float]) -> Dict:

        from scipy import stats

        groups_clean = [np.array(g, dtype=float) for g in groups if len(g) >= 2]

        if len(groups_clean) < 2:

            return {"f_stat": float("nan"), "p_value": float("nan")}

        try:

            f_stat, p_value = stats.f_oneway(*groups_clean)

            return {"f_stat": float(f_stat), "p_value": float(p_value)}

        except Exception as e:

            logger.warning("ANOVA failed: %s", e)

            return {"f_stat": float("nan"), "p_value": float("nan")}

    @staticmethod

    def mann_whitney(a: Sequence[float], b: Sequence[float]) -> Dict:

        from scipy import stats

        a_arr = np.array(a, dtype=float)

        b_arr = np.array(b, dtype=float)

        if len(a_arr) < 2 or len(b_arr) < 2:

            return {"u_stat": float("nan"), "p_value": float("nan")}

        try:

            u_stat, p_value = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")

            return {"u_stat": float(u_stat), "p_value": float(p_value)}

        except Exception as e:

            logger.warning("Mann-Whitney failed: %s", e)

            return {"u_stat": float("nan"), "p_value": float("nan")}

    @staticmethod

    def chi_square(observed: Dict, expected: Dict = None) -> Dict:

        """Chi-square test for POS distribution."""

        from scipy import stats

        vocab = sorted(observed.keys())

        obs = np.array([observed.get(k, 0) for k in vocab], dtype=float)

        if expected is None:



            exp = np.full_like(obs, obs.sum() / len(obs))

        else:

            exp = np.array([expected.get(k, 0) for k in vocab], dtype=float)



        mask = exp > 0

        obs, exp = obs[mask], exp[mask]

        if len(obs) < 2:

            return {"chi2": float("nan"), "p_value": float("nan"), "dof": 0}

        try:

            chi2, p = stats.chisquare(obs, f_exp=exp)

            return {"chi2": float(chi2), "p_value": float(p), "dof": len(obs) - 1}

        except Exception as e:

            logger.warning("Chi-square failed: %s", e)

            return {"chi2": float("nan"), "p_value": float("nan"), "dof": 0}

    @staticmethod

    def mixed_effects_model(df: pd.DataFrame, outcome: str, fixed: List[str]) -> Optional[object]:

        """
        Fit a mixed-effects model with 'language' as random effect.
        Returns the fitted model result or None on failure.
        """

        try:

            import statsmodels.formula.api as smf

            warnings.filterwarnings("ignore")

            clean = df[[outcome] + fixed + ["language"]].dropna()

            if len(clean) < 30:

                return None

            formula = f"{outcome} ~ " + " + ".join(fixed)

            model = smf.mixedlm(formula, clean, groups=clean["language"])

            result = model.fit(reml=False)

            return result

        except Exception as e:

            logger.warning("Mixed-effects model failed: %s", e)

            return None

    @staticmethod

    def compute_zscores(

        language: str,

        real_df: pd.DataFrame,

        random_stats: Dict,

    ) -> pd.DataFrame:

        """Build zscore_results.csv rows for a language."""

        rows = []

        metric_map = {

            "complexity_score": ("random_mean_complexity", "random_std_complexity"),

            "dependency_distance": ("random_mean_distance", "random_std_distance"),

        }

        for metric, (mean_key, std_key) in metric_map.items():

            if metric not in real_df.columns:

                continue

            real_val = real_df[metric].mean()

            r_mean = random_stats.get(mean_key, float("nan"))

            r_std = random_stats.get(std_key, float("nan"))

            z = z_score(real_val, r_mean, r_std)

            rows.append({

                "language": language,

                "metric": metric,

                "real_value": real_val,

                "random_mean": r_mean,

                "random_std": r_std,

                "z_score": z,

            })

        return pd.DataFrame(rows)

    @staticmethod

    def per_sentence_zscores_with_ci(

        language: str,

        sentences: List["Sentence"],

        features_df: pd.DataFrame,

        baseline: "RandomTreeBaseline",

        n_sentences_sample: int = 200,

        n_bootstrap: int = 1000,

        seed: int = 42,

    ) -> Dict[str, pd.DataFrame]:

        """Compute per-sentence z-scores with bootstrap 95% CI.

        AUDIT FIX (2026-04-24) — Fix 13/17:
        Futrell et al. (2015) compute per-sentence z-scores then aggregate with
        bootstrap CIs, preserving independence of sentences as the unit of analysis.
        Corpus-level aggregation (single mean vs single random mean) treats all tokens
        as one data point, inflating precision and masking sentence-level variance.

        Method:
          1. For each sampled sentence, compute real mean complexity/distance.
          2. Run baseline.compute_sentence_stats() to get sentence-level random mean/std.
          3. Compute sentence-level z-score = (real - random_mean) / random_std.
          4. Aggregate: mean z, std, and 95% bootstrap CI across all sampled sentences.

        Returns dict with keys:
          'per_sentence' — DataFrame, one row per sentence
          'summary'      — DataFrame, one row with mean z, CI, and support percentages
        """

        rng = np.random.default_rng(seed)



        if "sentence_id" not in features_df.columns:

            logger.warning("[%s] per_sentence_zscores: 'sentence_id' missing in features_df", language)

            return {}

        sent_real = (

            features_df.groupby("sentence_id")[["complexity_score", "dependency_distance"]]

            .mean()

            .rename(columns={"complexity_score": "real_complexity",

                             "dependency_distance": "real_distance"})

        )

        sample = sentences[:n_sentences_sample]

        rows = []

        for sent in sample:

            sid = sent.sent_id

            if sid not in sent_real.index:

                continue

            real_c = float(sent_real.loc[sid, "real_complexity"])

            real_d = float(sent_real.loc[sid, "real_distance"])

            stats = baseline.compute_sentence_stats(sent)

            b_mean_c = stats.get("mean_complexity", float("nan"))

            b_std_c  = stats.get("std_complexity",  float("nan"))

            b_mean_d = stats.get("mean_distance",    float("nan"))

            b_std_d  = stats.get("std_distance",     float("nan"))

            z_c = z_score(real_c, b_mean_c, b_std_c)

            z_d = z_score(real_d, b_mean_d, b_std_d)

            rows.append({

                "language": language,

                "sentence_id": sid,

                "real_complexity": real_c,

                "baseline_mean_complexity": b_mean_c,

                "baseline_std_complexity": b_std_c,

                "z_complexity": z_c,

                "real_distance": real_d,

                "baseline_mean_distance": b_mean_d,

                "baseline_std_distance": b_std_d,

                "z_distance": z_d,

                "supports_icm": z_c < 0,

                "supports_dlm": z_d > 0,

            })

        if not rows:

            logger.warning("[%s] per_sentence_zscores: no matching sentences found", language)

            return {}

        sent_df = pd.DataFrame(rows)

        z_c_arr = sent_df["z_complexity"].replace([np.inf, -np.inf], np.nan).dropna().values

        z_d_arr = sent_df["z_distance"].replace([np.inf, -np.inf], np.nan).dropna().values



        def bootstrap_ci(arr, n_boot):

            if len(arr) < 2:

                return float("nan"), float("nan")

            boot = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))

                    for _ in range(n_boot)]

            return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        ci_c_lo, ci_c_hi = bootstrap_ci(z_c_arr, n_bootstrap)

        ci_d_lo, ci_d_hi = bootstrap_ci(z_d_arr, n_bootstrap)

        pct_icm    = float((z_c_arr < 0).mean() * 100)

        pct_sig_icm = float((z_c_arr < -1.96).mean() * 100)

        pct_dlm    = float((z_d_arr > 0).mean() * 100)

        pct_sig_dlm = float((z_d_arr > 1.96).mean() * 100)

        summary_df = pd.DataFrame([{

            "language": language,

            "n_sentences_sampled": len(sent_df),

            "mean_z_complexity": float(z_c_arr.mean()) if len(z_c_arr) else float("nan"),

            "std_z_complexity":  float(z_c_arr.std())  if len(z_c_arr) else float("nan"),

            "ci_lower_z_complexity": ci_c_lo,

            "ci_upper_z_complexity": ci_c_hi,

            "ci_excludes_zero_icm": ci_c_hi < 0,

            "mean_z_distance": float(z_d_arr.mean()) if len(z_d_arr) else float("nan"),

            "std_z_distance":  float(z_d_arr.std())  if len(z_d_arr) else float("nan"),

            "ci_lower_z_distance": ci_d_lo,

            "ci_upper_z_distance": ci_d_hi,

            "ci_excludes_zero_dlm": ci_d_lo > 0,

            "pct_sentences_supporting_icm": pct_icm,

            "pct_sentences_significant_icm": pct_sig_icm,

            "pct_sentences_supporting_dlm": pct_dlm,

            "pct_sentences_significant_dlm": pct_sig_dlm,

            "n_bootstrap": n_bootstrap,

        }])

        logger.info(

            "[%s] Per-sentence ICM: mean_z=%.3f [%.3f, %.3f] | "

            "%.1f%% sentences support ICM, %.1f%% significant",

            language, float(z_c_arr.mean()), ci_c_lo, ci_c_hi, pct_icm, pct_sig_icm,

        )

        logger.info(

            "[%s] Per-sentence DLM: mean_z=%.3f [%.3f, %.3f] | "

            "%.1f%% sentences support DLM, %.1f%% significant",

            language, float(z_d_arr.mean()), ci_d_lo, ci_d_hi, pct_dlm, pct_sig_dlm,

        )

        return {"per_sentence": sent_df, "summary": summary_df}

    @staticmethod

    def ancova(

        df: pd.DataFrame,

        outcome: str,

        group: str,

        covariate: str,

    ) -> Dict[str, Any]:

        """ANCOVA: test group effect on outcome while controlling for a covariate.

        AUDIT FIX (2026-04-24) — Fix 14:
        Dravidian languages are all SOV. A plain Mann-Whitney comparison of Dravidian vs
        non-Dravidian confounds language-family effects with word-order effects.
        This ANCOVA isolates the family contribution after controlling for word order.

        Method: fit two OLS models:
          Reduced: outcome ~ C(covariate)
          Full:    outcome ~ C(group) + C(covariate)
        F-test the incremental variance explained by adding 'group'.

        Returns dict with f_stat, p_value, partial_eta_squared, n_obs.
        """

        try:

            import statsmodels.formula.api as smf

            from scipy import stats as sp_stats

            warnings.filterwarnings("ignore")

            clean = df[[outcome, group, covariate]].dropna().copy()

            if len(clean) < 10:

                return {"error": "insufficient data", "n_obs": len(clean)}

            formula_full    = f"{outcome} ~ C({group}) + C({covariate})"

            formula_reduced = f"{outcome} ~ C({covariate})"

            model_full    = smf.ols(formula_full,    clean).fit()

            model_reduced = smf.ols(formula_reduced, clean).fit()

            ss_group  = model_reduced.ssr - model_full.ssr

            ss_error  = model_full.ssr

            df_group  = model_full.df_model - model_reduced.df_model

            df_error  = model_full.df_resid

            if df_group > 0 and df_error > 0 and ss_error > 0:

                f_stat  = (ss_group / df_group) / (ss_error / df_error)

                p_value = float(1 - sp_stats.f.cdf(f_stat, df_group, df_error))

                partial_eta_sq = ss_group / (ss_group + ss_error)

            else:

                f_stat, p_value, partial_eta_sq = float("nan"), float("nan"), float("nan")

            result = {

                "f_stat": float(f_stat),

                "p_value": p_value,

                "partial_eta_squared": float(partial_eta_sq),

                "n_obs": len(clean),

                "group_variable": group,

                "covariate": covariate,

                "outcome": outcome,

                "r_squared_full": float(model_full.rsquared),

                "r_squared_reduced": float(model_reduced.rsquared),

            }

            logger.info(

                "ANCOVA %s ~ %s (controlling %s): F=%.3f p=%.4f partial_eta2=%.3f n=%d",

                outcome, group, covariate,

                f_stat, p_value, partial_eta_sq if not math.isnan(partial_eta_sq) else 0,

                len(clean),

            )

            return result

        except Exception as e:

            logger.warning("ANCOVA failed: %s", e)

            return {"error": str(e)}

    @staticmethod

    def power_analysis_n40(alpha: float = 0.05) -> Dict[str, Any]:

        """Post-hoc power analysis for one-way ANOVA with n=40 languages, 4 groups.

        AUDIT FIX (2026-04-24) — Fix 18:
        The study uses n=40 languages across 4 typological groups (~10 per group).
        Without a power analysis, we cannot know whether a non-significant result
        (e.g., complexity ANOVA p=0.152) reflects a true null or insufficient power.
        Reference: Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.

        Returns dict with power at small/medium/large effect sizes and n needed
        to achieve 80% power for a medium effect.
        """

        results: Dict[str, Any] = {

            "alpha": alpha,

            "k_groups": 4,

            "n_per_group_approx": 10,

            "total_languages": 40,

        }

        try:

            from statsmodels.stats.power import FTestAnovaPower

            analyzer = FTestAnovaPower()

            n_per_group = 10

            for label, f_effect in [("small", 0.10), ("medium", 0.25), ("large", 0.40)]:

                pwr = analyzer.power(

                    effect_size=f_effect,

                    nobs=n_per_group,

                    alpha=alpha,

                    k_groups=4,

                )

                results[f"power_{label}_effect_f{f_effect}"] = float(pwr)

                logger.info("Power (n=%d/group, f=%.2f %s effect): %.3f",

                            n_per_group, f_effect, label, pwr)



            n_needed = analyzer.solve_power(

                effect_size=0.25,

                alpha=alpha,

                power=0.80,

                k_groups=4,

            )

            results["n_per_group_needed_for_80pct_power_medium"] = float(n_needed)

            results["total_languages_needed"] = float(n_needed * 4)

            logger.info(

                "To detect medium effect (f=0.25) at 80%% power: need ~%d languages per group "

                "(~%d total). Current study has ~10/group (40 total).",

                int(n_needed) + 1, int(n_needed * 4) + 1,

            )

        except ImportError:

            logger.warning("statsmodels.stats.power not available — skipping power analysis")

            results["error"] = "statsmodels.stats.power not installed"

        except Exception as e:

            logger.warning("Power analysis failed: %s", e)

            results["error"] = str(e)

        return results

