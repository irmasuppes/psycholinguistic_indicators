# --- evaluate.py (patch) ---
import json, os
import pandas as pd
import numpy as np
from classifiers import train_indicators, evaluate_preds, train_tfidf, train_hybrid, train_bert, train_hybrid_bert

SHARED = ["sentiment_polarity", "sentiment_intensity",
          "social_reference_density_inv", "coping_strategy_num",
          "energy_level_inv", "rumination_level"]
DEP    = ["hopelessness_level", "cognitive_distortion_bin",
          "certainty_expression", "social_withdrawal"]
STR    = ["loss_of_control", "anger_hate", "physiological_state_inv", "help_seeking_inv"]

DEP_LABEL_MAP = {"minimum":0, "mild":1, "moderate":2, "severe":3}

SEED=42

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def to_numeric(df, cols):
    out = {}
    for c in cols:
        if c not in df: 
            continue
        s = df[c]
        if s.dtype == object:
            codes, _ = pd.factorize(s.astype(str), sort=True)
            out[c] = codes
        else:
            out[c] = pd.to_numeric(s, errors="coerce")
    return pd.DataFrame(out)

def round_dict(d, ndigits=3):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = round_dict(v, ndigits)
        elif isinstance(v, (float, int)) and v is not None:
            out[k] = round(float(v), ndigits)
        else:
            out[k] = v
    return out

def df_to_latex(df, caption, label, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False,
                            float_format=lambda x: f"{x:.3f}",
                            caption=caption, label=label))

def run_eval(ind_file, task="both", outdir="results", balanced=False):
    model_name = os.path.splitext(os.path.basename(ind_file))[0]
    print(f"Evaluating {model_name}")
    model_dir = os.path.join(outdir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    bert_model = 'AIMH/mental-roberta-large'
    df = load_jsonl(ind_file)

    shared, dep, strx = to_numeric(df, SHARED), to_numeric(df, DEP), to_numeric(df, STR)

    results = {}

    # STRESS (binary)
    if task in ["stress","both"]:
        print("\n=== Stress classification (binary) ===")
        y = df["label_stress"].astype(int).tolist()
        posts = df["text"].astype(str).tolist()

        rows = []

        print("\nTF-IDF baseline:")
        preds = train_tfidf(posts, y)
        metrics = evaluate_preds(y, preds, task="binary") 
        results["stress_tfidf"] = round_dict(metrics)
        rows.append({"Model": "TF–IDF", "Balanced Acc.": metrics["balanced_accuracy"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        print("\nIndicators-only (shared+stress):")
        ind = pd.concat([shared, strx], axis=1)
        preds = train_indicators(ind, y)
        metrics = evaluate_preds(y, preds, task="binary")
        results["stress_shared+stress"] = round_dict(metrics)
        rows.append({"Model": "Indicators (shared+stress)", "Balanced Acc.": metrics["balanced_accuracy"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        print("\nHybrid TF-IDF + indicators:")
        preds = train_hybrid(posts, ind, y)
        metrics = evaluate_preds(y, preds, task="binary")
        results["stress_hybrid"] = round_dict(metrics)
        rows.append({"Model": "Hybrid TF–IDF+Indicators", "Balanced Acc.": metrics["balanced_accuracy"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        print("\nBERT embeddings:")
        preds = train_bert(posts, y, model_name=bert_model)
        metrics = evaluate_preds(y, preds, task="binary")
        results["stress_bert"] = round_dict(metrics)
        rows.append({"Model": "Mental-RoBERTa", "Balanced Acc.": metrics["balanced_accuracy"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        print("\nHybrid TF-IDF + indicators + BERT:")
        preds = train_hybrid_bert(posts, ind, y, model_name=bert_model)
        metrics = evaluate_preds(y, preds, task="binary")
        results["stress_hybrid_bert"] = round_dict(metrics)
        rows.append({"Model": "Hybrid TF–IDF+Ind+BERT", "Balanced Acc.": metrics["balanced_accuracy"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        stress_df = pd.DataFrame(rows)
        df_to_latex(stress_df, caption="Comparison of feature representations for binary stress detection (5×CV).",
                    label="tab:stress_results", path=os.path.join(model_dir, "table_stress_results.tex"))

    # DEPRESSION (ordinal)
    if task in ["depression","both"]:
        print("\n=== Depression severity (ordinal) ===")
        mapped = df["label_depression"].map(DEP_LABEL_MAP)
        if mapped.isna().any():
            print("Warning: unknown/missing depression labels:",
                  df.loc[mapped.isna(), "label_depression"].unique())
            keep = mapped.notna()
            df = df.loc[keep].copy()
            shared, dep = shared.loc[keep], dep.loc[keep]
            mapped = mapped.loc[keep]

        posts = df["text"].astype(str).tolist()
        y = mapped.astype(int).tolist()

        rows = []

        print("\nTF-IDF baseline:")
        preds = train_tfidf(posts, y)
        metrics = evaluate_preds(y, preds, task="ordinal")
        results["depression_tfidf"] = round_dict(metrics)
        rows.append({"Model": "TF–IDF", "Balanced Acc.": metrics["balanced_accuracy"],
                     "QWK": metrics["quadratic_weighted_kappa"], "Spearman ρ": metrics["spearman_rho"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        print("\nIndicators-only (shared+depression):")
        ind = pd.concat([shared, dep], axis=1)
        preds = train_indicators(ind, y)
        metrics = evaluate_preds(y, preds, task="ordinal")
        results["depression_shared+dep"] = round_dict(metrics)
        rows.append({"Model": "Indicators (shared+dep)", "Balanced Acc.": metrics["balanced_accuracy"],
                     "QWK": metrics["quadratic_weighted_kappa"], "Spearman ρ": metrics["spearman_rho"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        print("\nHybrid TF-IDF + indicators:")
        preds = train_hybrid(posts, ind, y)
        metrics = evaluate_preds(y, preds, task="ordinal")
        results["depression_hybrid"] = round_dict(metrics)
        rows.append({"Model": "Hybrid TF–IDF+Indicators", "Balanced Acc.": metrics["balanced_accuracy"],
                     "QWK": metrics["quadratic_weighted_kappa"], "Spearman ρ": metrics["spearman_rho"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        print("\nBERT embeddings:")
        preds = train_bert(posts, y, model_name=bert_model)
        metrics = evaluate_preds(y, preds, task="ordinal")
        results["depression_bert"] = round_dict(metrics)
        rows.append({"Model": "Mental-RoBERTa", "Balanced Acc.": metrics["balanced_accuracy"],
                     "QWK": metrics["quadratic_weighted_kappa"], "Spearman ρ": metrics["spearman_rho"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        print("\nHybrid TF-IDF + indicators + BERT:")
        preds = train_hybrid_bert(posts, ind, y, model_name=bert_model)
        metrics = evaluate_preds(y, preds, task="ordinal")
        results["depression_hybrid_bert"] = round_dict(metrics)
        rows.append({"Model": "Hybrid TF–IDF+Ind+BERT", "Balanced Acc.": metrics["balanced_accuracy"],
                     "QWK": metrics["quadratic_weighted_kappa"], "Spearman ρ": metrics["spearman_rho"],
                     "Macro F1": metrics["macro avg"]["f1-score"]})

        dep_df = pd.DataFrame(rows)
        df_to_latex(dep_df, caption="Comparison of feature representations for depression-severity prediction (5×CV).",
                    label="tab:depression_results", path=os.path.join(model_dir, "table_depression_results.tex"))

    # Save rounded JSON
    outpath = os.path.join(model_dir, f"{model_name}_eval.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {outpath}")
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <ind_file> [task] [outdir] [if_balanced]")
        raise SystemExit(1)
    ind_file = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 else "both"
    outdir = sys.argv[3] if len(sys.argv) > 3 else "results"
    balanced = True if sys.argv[4]=="True" and len(sys.argv) > 4 else False
    run_eval(ind_file, task=task, outdir=outdir, balanced=balanced)
