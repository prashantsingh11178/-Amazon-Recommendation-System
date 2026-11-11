"""
Recommender system comparison
-----------------------------
Trains a few different recommenders on the same interaction data, tests them on
the same held-out split and compares them with the same metrics.

Models
  Popularity     baseline, everybody gets the same "best sellers" list
  Item-kNN       collaborative filtering ("people who liked X also liked Y")
  MF-ALS         matrix factorisation (user/item biases + latent factors, trained with ALS)
  Content-based  TF-IDF on item text, only runs if an item metadata file exists
  Hybrid         a simple blend of the models above

Usage
  python recsys_compare.py --demo                      # quick run on fake data
  python recsys_compare.py --data path/to/folder_or_csv
  python recsys_compare.py --data ratings.csv --items items.csv \
         --user-col userId --item-col movieId --rating-col rating
"""
import argparse
import html
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

# >>> Put your dataset folder here (the folder that has Books.csv, Ratings.csv, Users.csv).
# With this set, you can simply run:  python recsys_compare.py
DATA_FOLDER = r"C:\Users\prash\OneDrive\Desktop\amazon recomendation system\books_data"

USER_NAMES = ["userid", "user", "customerid", "reviewerid", "visitorid", "uid"]
ITEM_NAMES = ["itemid", "item", "productid", "movieid", "asin", "product", "iid", "bookid", "isbn"]
RATING_NAMES = ["rating", "ratings", "score", "stars", "overall", "rate", "bookrating"]
TITLE_NAMES = ["title", "name", "productname", "itemname", "movietitle", "booktitle"]


# ----------------------------------------------------------------------------
# loading + cleaning
# ----------------------------------------------------------------------------
def clean(name):
    return str(name).lower().replace("_", "").replace("-", "").replace(" ", "")


def pick_column(df, candidates):
    lookup = {clean(c): c for c in df.columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def read_table(path, nrows=None):
    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False, encoding="utf-8", encoding_errors="replace")


def look_in_folder(folder):
    """If --data is a folder, guess which file is the ratings and which one is item info."""
    ratings, items = None, None
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".csv", ".tsv")):
            continue
        path = os.path.join(folder, fname)
        head = read_table(path, nrows=5)
        has_user = pick_column(head, USER_NAMES) is not None
        has_item = pick_column(head, ITEM_NAMES) is not None
        if has_user and has_item:
            # if there are several candidates keep the biggest file
            if ratings is None or os.path.getsize(path) > os.path.getsize(ratings):
                ratings = path
        elif has_item:
            items = path
    if ratings is None:
        raise SystemExit(f"Couldn't find a ratings/interactions file in {folder}. "
                         "Pass it with --data and the --user-col/--item-col flags.")
    return ratings, items


def make_demo_data(folder, n_users=600, n_items=400, seed=SEED):
    """Fake movie-style data so the pipeline can be tested without the real dataset."""
    rng = np.random.RandomState(seed)
    genres = ["action", "comedy", "drama", "romance", "scifi", "thriller", "animation", "horror"]
    k = 6
    P = rng.normal(0, 1, (n_users, k))
    Q = rng.normal(0, 1, (n_items, k))
    item_pop = rng.pareto(1.5, n_items) + 0.3
    item_pop = item_pop / item_pop.sum()
    item_genres = [rng.choice(genres, size=rng.randint(1, 3), replace=False) for _ in range(n_items)]

    rows = []
    for u in range(n_users):
        n = int(np.clip(rng.lognormal(3.3, 0.6), 8, 150))
        items = rng.choice(n_items, size=min(n, n_items), replace=False, p=item_pop)
        for i in items:
            raw = 3.4 + 0.55 * P[u] @ Q[i] + rng.normal(0, 0.5)
            rows.append((u, i, int(np.clip(np.round(raw), 1, 5))))
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame(rows, columns=["user_id", "item_id", "rating"]).to_csv(
        os.path.join(folder, "ratings.csv"), index=False)
    pd.DataFrame({
        "item_id": range(n_items),
        "title": [f"Movie {i}" for i in range(n_items)],
        "genres": [" ".join(g) for g in item_genres],
    }).to_csv(os.path.join(folder, "items.csv"), index=False)
    return folder


def load_data(args):
    data_path, items_path = args.data, args.items
    if os.path.isdir(data_path):
        data_path, found_items = look_in_folder(data_path)
        items_path = items_path or found_items

    raw = read_table(data_path)
    ucol = args.user_col or pick_column(raw, USER_NAMES)
    icol = args.item_col or pick_column(raw, ITEM_NAMES)
    rcol = args.rating_col or pick_column(raw, RATING_NAMES)
    if ucol is None or icol is None:
        raise SystemExit(f"Can't work out the user/item columns. Columns are: {list(raw.columns)}. "
                         "Use --user-col and --item-col.")

    df = pd.DataFrame({"user": raw[ucol].astype(str), "item": raw[icol].astype(str).str.strip().str.upper()})
    if rcol is None:
        print("No rating column found -> treating the data as implicit feedback (all = 1)")
        df["rating"] = 1.0
        explicit = False
    else:
        df["rating"] = pd.to_numeric(raw[rcol], errors="coerce")
        explicit = True
    df = df.dropna()
    n_before = len(df)
    df = df[df["rating"] > 0]      # in some datasets (e.g. Book-Crossing) 0 = "no rating given"
    if explicit and n_before != len(df):
        print(f"Dropped {n_before - len(df):,} rows with rating 0 (no explicit rating), kept {len(df):,}")
    df = df.drop_duplicates(["user", "item"], keep="last")

    meta = None
    if items_path:
        meta = read_table(items_path)
        mcol = pick_column(meta, ITEM_NAMES)
        if mcol is None:
            print("Item file has no item id column, ignoring it")
            meta = None
        else:
            meta[mcol] = meta[mcol].astype(str).str.strip().str.upper()
            meta = meta.drop_duplicates(mcol).set_index(mcol)
    print(f"Loaded {len(df):,} interactions from {data_path}")
    return df, meta, explicit


def filter_sparse(df, min_user, min_item):
    # keep dropping until nothing changes, otherwise removing items can push users below the limit
    while True:
        before = len(df)
        df = df[df.groupby("user")["user"].transform("size") >= min_user]
        df = df[df.groupby("item")["item"].transform("size") >= min_item]
        if len(df) == before:
            return df


def split_per_user(df, test_frac, seed=SEED):
    """Hold out a fraction of every user's interactions for testing."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    pos = df.groupby("user").cumcount()
    n = df.groupby("user")["user"].transform("size")
    n_test = np.minimum(np.maximum(1, np.round(n * test_frac)), n - 1)
    is_test = pos >= (n - n_test)
    return df[~is_test].copy(), df[is_test].copy()


def to_matrix(df, shape):
    return sp.csr_matrix((df["rating"].values, (df["u"].values, df["i"].values)), shape=shape)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
class Recommender:
    name = "base"
    predicts_ratings = True

    def set_bounds(self, R):
        self.lo, self.hi = R.data.min(), R.data.max()

    def clip(self, x):
        return np.clip(x, self.lo, self.hi)

    def rank_score(self, users):
        return self.score(users)


class Popularity(Recommender):
    name = "Popularity"

    def fit(self, R):
        self.set_bounds(R)
        counts = np.asarray((R > 0).sum(0)).ravel()
        sums = np.asarray(R.sum(0)).ravel()
        g = R.data.mean()
        m = 5  # damping so items with one 5-star rating don't win
        self.item_mean = (sums + m * g) / (counts + m)
        self.counts = counts.astype(float)
        return self

    def score(self, users):
        return np.tile(self.item_mean, (len(users), 1))

    def rank_score(self, users):
        return np.tile(self.counts, (len(users), 1))


class ItemKNN(Recommender):
    name = "Item-kNN"

    def __init__(self, k=50, shrink=10, damp=1.0):
        self.k, self.shrink, self.damp = k, shrink, damp

    def fit(self, R):
        R = R.tocsr()
        self.set_bounds(R)
        n_items = R.shape[1]
        counts = np.diff(R.indptr)
        sums = np.asarray(R.sum(1)).ravel()
        g = R.data.mean()
        self.user_mean = np.where(counts > 0, sums / np.maximum(counts, 1), g)

        # centre every rating on the user's own average
        Rc = R.copy().astype(np.float64)
        Rc.data = Rc.data - np.repeat(self.user_mean, counts)
        self.Rc = Rc
        self.B = (R > 0).astype(np.float64).tocsr()

        X = normalize(Rc.T.tocsr())          # items x users, unit length rows
        Bt = self.B.T.tocsr()                # items x users, 0/1
        rows, cols, vals = [], [], []
        block = 500
        kk = min(self.k, n_items - 1)
        for start in range(0, n_items, block):
            end = min(start + block, n_items)
            sim = (X[start:end] @ X.T).toarray()
            co = (Bt[start:end] @ Bt.T).toarray()
            sim = sim * co / (co + self.shrink)       # shrink sims backed by few common users
            sim[np.arange(end - start), np.arange(start, end)] = 0
            sim[sim < 0] = 0
            top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            r = np.repeat(np.arange(start, end), kk)
            c = top.ravel()
            v = sim[np.repeat(np.arange(end - start), kk), c]
            keep = v > 0
            rows.append(r[keep]); cols.append(c[keep]); vals.append(v[keep])
        self.S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n_items, n_items))
        self.ST = self.S.T.tocsr()
        return self

    def _num_den(self, users):
        num = (self.Rc[users] @ self.ST).toarray()
        den = (self.B[users] @ self.ST).toarray()
        return num, den

    def score(self, users):
        num, den = self._num_den(users)
        dev = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
        return self.clip(self.user_mean[users][:, None] + dev)

    def rank_score(self, users):
        num, den = self._num_den(users)
        return num / (den + self.damp)


class ALSRecommender(Recommender):
    """Matrix factorisation trained with alternating least squares.

    First we remove the easy stuff (global mean + item bias + user bias), then the
    latent factors only have to explain what's left over.
    """
    name = "MF-ALS"

    def __init__(self, factors=20, reg=0.1, iters=10, reg_item=25, reg_user=10):
        self.factors, self.reg, self.iters = factors, reg, iters
        self.reg_item, self.reg_user = reg_item, reg_user

    def _solve(self, M, other):
        """Update one side of the factorisation while the other side is held fixed."""
        k = self.factors
        out = np.zeros((M.shape[0], k))
        eye = np.eye(k)
        for r in range(M.shape[0]):
            s, e = M.indptr[r], M.indptr[r + 1]
            if s == e:
                continue
            F = other[M.indices[s:e]]
            A = F.T @ F + self.reg * (e - s) * eye
            out[r] = np.linalg.solve(A, F.T @ M.data[s:e])
        return out

    def fit(self, R):
        R = R.tocsr()
        self.set_bounds(R)
        coo = R.tocoo()
        n_users, n_items = R.shape
        self.mu = coo.data.mean()

        n_i = np.bincount(coo.col, minlength=n_items)
        self.item_bias = np.bincount(coo.col, weights=coo.data - self.mu, minlength=n_items) / (n_i + self.reg_item)
        n_u = np.bincount(coo.row, minlength=n_users)
        resid1 = coo.data - self.mu - self.item_bias[coo.col]
        self.user_bias = np.bincount(coo.row, weights=resid1, minlength=n_users) / (n_u + self.reg_user)
        resid = resid1 - self.user_bias[coo.row]

        E = sp.csr_matrix((resid, (coo.row, coo.col)), shape=R.shape, dtype=np.float64)
        Et = E.T.tocsr()
        rng = np.random.RandomState(SEED)
        self.P = rng.normal(0, 0.1, (n_users, self.factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.factors))
        for _ in range(self.iters):
            self.P = self._solve(E, self.Q)
            self.Q = self._solve(Et, self.P)
        return self

    def score(self, users):
        base = self.mu + self.user_bias[users][:, None] + self.item_bias[None, :]
        return self.clip(base + self.P[users] @ self.Q.T)


class ContentBased(Recommender):
    name = "Content-based"
    predicts_ratings = False

    def __init__(self, item_texts):
        self.item_texts = item_texts

    def fit(self, R):
        R = R.tocsr()
        counts = np.diff(R.indptr)
        sums = np.asarray(R.sum(1)).ravel()
        user_mean = np.where(counts > 0, sums / np.maximum(counts, 1), R.data.mean())
        Rc = R.copy().astype(np.float64)
        Rc.data = Rc.data - np.repeat(user_mean, counts)

        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.X = tfidf.fit_transform(self.item_texts)           # items x terms
        # a user's taste = what they liked minus what they disliked, in "term space"
        self.profiles = normalize(Rc @ self.X)
        return self

    def rank_score(self, users):
        return (self.profiles[users] @ self.X.T).toarray()


class Hybrid(Recommender):
    name = "Hybrid"

    def __init__(self, parts):
        self.parts = parts          # list of (model, weight)

    def fit(self, R):
        return self                 # members are already trained

    @staticmethod
    def zscore(x):
        return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)

    def rank_score(self, users):
        return sum(w * self.zscore(m.rank_score(users)) for m, w in self.parts)

    def score(self, users):
        rated = [(m, w) for m, w in self.parts if m.predicts_ratings]
        total = sum(w for _, w in rated)
        return sum(w * m.score(users) for m, w in rated) / total


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def rating_metrics(model, test_mat, batch=512):
    errs = []
    n_users = test_mat.shape[0]
    for s in range(0, n_users, batch):
        users = np.arange(s, min(s + batch, n_users))
        coo = test_mat[users].tocoo()
        if coo.nnz == 0:
            continue
        preds = model.score(users)[coo.row, coo.col]
        errs.append(preds - coo.data)
    err = np.concatenate(errs)
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err)))}


def ranking_metrics(model, train, rel, ks=(5, 10, 20), batch=512):
    """Top-K lists for every user (items already seen in training are excluded)."""
    max_k = max(ks)
    n_users, n_items = train.shape
    sums = {f"{m}@{k}": 0.0 for k in ks for m in ("Precision", "Recall", "NDCG", "HitRate")}
    n_eval = 0
    recommended = set()
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))

    for s in range(0, n_users, batch):
        users = np.arange(s, min(s + batch, n_users))
        rel_b = rel[users].toarray()
        n_rel = rel_b.sum(1)
        scores = model.rank_score(users).astype(float)
        seen_r, seen_c = train[users].nonzero()
        scores[seen_r, seen_c] = -np.inf

        top = np.argpartition(-scores, max_k - 1, axis=1)[:, :max_k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)
        hits = np.take_along_axis(rel_b, top, axis=1)

        has_rel = n_rel > 0
        recommended.update(top[has_rel, :ks[1] if len(ks) > 1 else ks[0]].ravel().tolist())
        for k in ks:
            h = hits[:, :k]
            prec = h.sum(1) / k
            rec = h.sum(1) / np.maximum(n_rel, 1)
            dcg = (h * discounts[:k]).sum(1)
            ideal_n = np.minimum(n_rel, k).astype(int)
            idcg = np.array([discounts[:n].sum() for n in ideal_n])
            ndcg = dcg / np.maximum(idcg, 1e-9)
            sums[f"Precision@{k}"] += prec[has_rel].sum()
            sums[f"Recall@{k}"] += rec[has_rel].sum()
            sums[f"NDCG@{k}"] += ndcg[has_rel].sum()
            sums[f"HitRate@{k}"] += (h.sum(1) > 0)[has_rel].sum()
        n_eval += has_rel.sum()

    out = {k: v / max(n_eval, 1) for k, v in sums.items()}
    out["Coverage"] = len(recommended) / n_items
    out["users_evaluated"] = int(n_eval)
    return out


def quick_tune(make_model, param_name, values, train_df, shape, seed=SEED):
    """Pick a hyper-parameter by RMSE on a validation slice cut out of the training data."""
    fit_df, val_df = split_per_user(train_df, 0.15, seed)
    R_fit, R_val = to_matrix(fit_df, shape), to_matrix(val_df, shape)
    best, best_rmse = None, np.inf
    for v in values:
        m = make_model(v).fit(R_fit)
        rmse = rating_metrics(m, R_val)["RMSE"]
        print(f"    {param_name}={v:<4} val RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best, best_rmse = v, rmse
    print(f"    -> using {param_name}={best}")
    return best


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_FOLDER, help="ratings csv/tsv, or a folder that contains it")
    ap.add_argument("--items", help="optional item metadata csv (titles, genres, descriptions...)")
    ap.add_argument("--user-col"); ap.add_argument("--item-col"); ap.add_argument("--rating-col")
    ap.add_argument("--demo", action="store_true", help="generate fake data and run on it")
    ap.add_argument("--min-user", type=int, default=5)
    ap.add_argument("--min-item", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--rel-threshold", type=float, default=8.0, help="rating that counts as 'liked' for ranking metrics")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.demo:
        args.data = make_demo_data("demo_data")
        print("Generated fake data in ./demo_data")
    if not os.path.exists(args.data):
        raise SystemExit(f"Folder not found: {args.data}\nFix DATA_FOLDER at the top of this file, or pass --data.")
    os.makedirs(args.out, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    df, meta, explicit = load_data(args)
    df = filter_sparse(df, args.min_user, args.min_item)
    df["u"], user_ids = pd.factorize(df["user"])
    df["i"], item_ids = pd.factorize(df["item"])
    n_users, n_items = len(user_ids), len(item_ids)
    density = len(df) / (n_users * n_items)
    print(f"After filtering: {n_users:,} users, {n_items:,} items, {len(df):,} ratings "
          f"(density {density:.2%})")
    print(f"Rating range {df['rating'].min()} - {df['rating'].max()}, mean {df['rating'].mean():.2f}")

    shape = (n_users, n_items)
    train_df, test_df = split_per_user(df, args.test_frac)
    # items that only show up in test can't be scored by CF models, so they're left out of the test set
    test_df = test_df[test_df["i"].isin(train_df["i"].unique())]
    R_train, R_test = to_matrix(train_df, shape), to_matrix(test_df, shape)
    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

    thr = args.rel_threshold
    if thr is None:
        top = df["rating"].max()
        thr = 1.0 if not explicit else (4.0 if top <= 5 else 7.0 if top <= 10 else df["rating"].quantile(0.75))
    print(f"A test item counts as relevant when rating >= {thr}")
    rel = to_matrix(test_df[test_df["rating"] >= thr].assign(rating=1.0), shape)

    # 2. tune a little -------------------------------------------------------
    print("\nTuning (validation split taken from the training data only)")
    if explicit:
        print("  Item-kNN neighbours:")
        best_k = quick_tune(lambda v: ItemKNN(k=v), "k", [20, 50, 100], train_df, shape)
        print("  ALS latent factors:")
        best_f = quick_tune(lambda v: ALSRecommender(factors=v), "factors", [10, 20, 40], train_df, shape)
    else:
        best_k, best_f = 50, 20

    # 3. train ---------------------------------------------------------------
    print("\nTraining final models on the full training set")
    pop = Popularity().fit(R_train)
    knn = ItemKNN(k=best_k).fit(R_train)
    mf = ALSRecommender(factors=best_f).fit(R_train)
    models = [pop, knn, mf]

    content = None
    text_cols = []
    if meta is not None:
        text_cols = [c for c in meta.columns if pd.api.types.is_string_dtype(meta[c]) and clean(c) not in
                     ("url", "imdburl", "link", "image", "imageurl") and "url" not in clean(c)]
    if text_cols:
        texts = (meta.reindex(item_ids)[text_cols].fillna("").astype(str).agg(" ".join, axis=1)).tolist()
        content = ContentBased(texts).fit(R_train)
        models.append(content)
        print(f"  content-based model uses these item columns: {text_cols}")
    else:
        print("  no item text found -> skipping the content-based model")

    parts = [(mf, 0.5), (knn, 0.3)] + ([(content, 0.2)] if content else [])
    models.append(Hybrid(parts))

    # 4. evaluate ------------------------------------------------------------
    print("\nEvaluating")
    rows = []
    for m in models:
        t0 = time.time()
        res = {"Model": m.name}
        if m.predicts_ratings and explicit:
            res.update(rating_metrics(m, R_test))
        res.update(ranking_metrics(m, R_train, rel))
        rows.append(res)
        print(f"  {m.name:<14} done in {time.time() - t0:.1f}s")

    results = pd.DataFrame(rows).set_index("Model")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== RESULTS ===")
    print(results.drop(columns=["users_evaluated"]).round(4).to_string())
    print(f"(ranking metrics averaged over {int(results['users_evaluated'].iloc[0])} users with at least one relevant test item)")
    results.round(4).to_csv(os.path.join(args.out, "metrics.csv"))

    # 5. plots ---------------------------------------------------------------
    rank_cols = ["Precision@10", "Recall@10", "NDCG@10", "HitRate@10"]
    fig, axes = plt.subplots(1, 2 if explicit else 1, figsize=(13 if explicit else 7, 4.5))
    axes = np.atleast_1d(axes)
    results[rank_cols].plot(kind="bar", ax=axes[0], rot=20)
    axes[0].set_title("Top-10 ranking quality (higher is better)")
    axes[0].set_ylabel("score")
    if explicit:
        results[["RMSE", "MAE"]].dropna().plot(kind="bar", ax=axes[1], rot=20, color=["#c0504d", "#9bbb59"])
        axes[1].set_title("Rating prediction error (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "model_comparison.png"), dpi=150)
    plt.close()

    # 6. a few example recommendations --------------------------------------
    best_name = results["NDCG@10"].idxmax()
    best_model = next(m for m in models if m.name == best_name)
    title_col = pick_column(meta, TITLE_NAMES) if meta is not None else None

    def label(i):
        raw = item_ids[i]
        if title_col is not None and raw in meta.index:
            return html.unescape(str(meta.loc[raw, title_col]))
        return str(raw)

    lines = [f"Example recommendations from the best model by NDCG@10: {best_name}\n"]
    rng = np.random.RandomState(SEED)
    for u in rng.choice(n_users, size=min(3, n_users), replace=False):
        liked = R_train[u].tocoo()
        fav = liked.col[np.argsort(-liked.data)[:5]]
        scores = best_model.rank_score(np.array([u])).astype(float)[0]
        scores[liked.col] = -np.inf
        recs = np.argsort(-scores)[:10]
        lines.append(f"user {user_ids[u]}")
        lines.append("  top rated in training : " + ", ".join(label(i) for i in fav))
        lines.append("  recommended           : " + ", ".join(label(i) for i in recs))
        lines.append("")
    with open(os.path.join(args.out, "sample_recommendations.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"Saved metrics.csv, model_comparison.png and sample_recommendations.txt in ./{args.out}")


if __name__ == "__main__":
    main()
