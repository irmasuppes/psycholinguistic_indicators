import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from scipy.stats import spearmanr

import os, hashlib, torch
from transformers import AutoTokenizer, AutoModel

# Shared CV for all models 
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def quadratic_weighted_kappa(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    min_rating, max_rating = int(min(y_true.min(), y_pred.min())), int(max(y_true.max(), y_pred.max()))
    n = max_rating - min_rating + 1
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w[i, j] = ((i-j)**2)/((n-1)**2)
    O = np.zeros_like(w)
    for a, p in zip(y_true, y_pred):
        O[a-min_rating][p-min_rating] += 1
    act_hist = np.bincount(y_true-min_rating, minlength=n)
    pred_hist = np.bincount(y_pred-min_rating, minlength=n)
    E = np.outer(act_hist, pred_hist)/len(y_true)
    return float(1 - (np.sum(w*O)/np.sum(w*E)))

def train_tfidf(posts, labels):
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3, stop_words="english")
    X = vect.fit_transform(posts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    preds = cross_val_predict(clf, X, labels, cv=CV)
    return preds

def train_indicators(ind_df, labels):
    X = StandardScaler().fit_transform(ind_df.fillna(0))
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    preds = cross_val_predict(clf, X, labels, cv=CV)
    return preds

def train_hybrid(posts, indicators, y):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3, stop_words="english")
    X_text = tfidf.fit_transform(posts)
    X_ind = StandardScaler().fit_transform(indicators.fillna(0))
    X = hstack([X_text, csr_matrix(X_ind)])  # ensure sparse concat
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    preds = cross_val_predict(clf, X, y, cv=CV)
    return preds

def evaluate_preds(y_true, y_pred, task="binary"):
    out = {}
    rep = classification_report(y_true, y_pred, digits=3, output_dict=True)
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro avg"] = rep["macro avg"]
    out["weighted avg"] = rep["weighted avg"]
    if task != "binary":  # ordinal
        out["quadratic_weighted_kappa"] = quadratic_weighted_kappa(y_true, y_pred)
        out["spearman_rho"] = float(spearmanr(y_true, y_pred).correlation)
    return out

def _cache_key(texts, model_name):
    m = hashlib.md5()
    m.update(model_name.encode())
    first = texts[0][:100] if texts else ""
    last  = texts[-1][-100:] if texts else ""
    m.update((first + last + str(len(texts))).encode("utf-8"))
    return m.hexdigest()[:16]

def get_embeddings_hf(texts, model_name="AIMH/mental-roberta-large",
                      cache_dir="cache_bert", batch_size=8, max_length=512):
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(texts, model_name)
    cache_path = os.path.join(cache_dir, f"{model_name.replace('/', '_')}_{key}.npy")
    if os.path.exists(cache_path):
        X = np.load(cache_path)
        return X

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs.append(pooled.cpu().numpy())
    X = np.vstack(vecs)
    np.save(cache_path, X)
    return X

def train_bert(posts, labels, model_name="AIMH/mental-roberta-large"):
    X = StandardScaler().fit_transform(get_embeddings_hf(posts, model_name))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    preds = cross_val_predict(clf, X, labels, cv=CV)
    return preds

def train_hybrid_bert(posts, indicators, labels, model_name="AIMH/mental-roberta-large"):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3, stop_words="english")
    X_text = tfidf.fit_transform(posts)
    X_ind  = StandardScaler().fit_transform(indicators.fillna(0))
    X_bert = StandardScaler().fit_transform(get_embeddings_hf(posts, model_name))
    X_all  = hstack([X_text, csr_matrix(X_ind), csr_matrix(X_bert)])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    preds = cross_val_predict(clf, X_all, labels, cv=CV)
    return preds
