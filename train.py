import os
import io
import json
import zipfile
import itertools
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

K = 10            # we evaluate the top-10 recommendations
NEIGHBORS = 50    # neighbours kept per item
SHRINK = 10       # similarity shrinkage (penalises pairs with few common raters)

# ---------------------------------------------------------------- 1. Load data
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
if not os.path.exists("data/ml-latest-small/ratings.csv"):
    print("Downloading MovieLens (about 1 MB)...")
    with urllib.request.urlopen(URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall("data")

ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
movies = pd.read_csv("data/ml-latest-small/movies.csv")
print(f"Raw: {len(ratings)} ratings, {ratings.userId.nunique()} users, {len(movies)} movies")

# keep movies with >= 5 ratings and users with >= 10 ratings (too little data to learn from otherwise)
cnt = ratings.movieId.value_counts()
ratings = ratings[ratings.movieId.isin(cnt[cnt >= 5].index)]
ucnt = ratings.userId.value_counts()
ratings = ratings[ratings.userId.isin(ucnt[ucnt >= 10].index)].copy()
movies = (movies[movies.movieId.isin(ratings.movieId.unique())]
          .sort_values("movieId").reset_index(drop=True))

user_ids = np.sort(ratings.userId.unique())
ratings["u"] = ratings.userId.map({uid: k for k, uid in enumerate(user_ids)})
ratings["i"] = ratings.movieId.map({mid: k for k, mid in enumerate(movies.movieId)})
n_users, n_items = len(user_ids), len(movies)
print(f"After filtering: {len(ratings)} ratings, {n_users} users, {n_items} movies")
print(f"Matrix density: {len(ratings) / (n_users * n_items):.2%} (very sparse!)")

# long-tail plot
counts_sorted = np.sort(ratings.i.value_counts().values)[::-1]
plt.plot(counts_sorted)
plt.yscale("log")
plt.xlabel("Movies ranked by popularity")
plt.ylabel("Number of ratings (log scale)")
plt.title("Long-tail: few movies get most ratings")
plt.tight_layout()
plt.savefig("outputs/long_tail.png")
plt.close()

# ---------------------------------------------------- 2. Time-based split per user
# first 70% of each user's history = train, next 10% = validation, last 20% = test
ratings = ratings.sort_values(["u", "timestamp"])
ratings["pos"] = ratings.groupby("u").cumcount() / ratings.groupby("u")["u"].transform("size")
train = ratings[ratings.pos < 0.7]
val = ratings[(ratings.pos >= 0.7) & (ratings.pos < 0.8)]
test = ratings[ratings.pos >= 0.8]
print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")


def to_matrix(d):
    return sparse.csr_matrix(
        (d.rating.values.astype(np.float32), (d.u.values, d.i.values)),
        shape=(n_users, n_items),
    )


# -------------------------------------------------------------- 3. Helper functions
def center(R):
    """Subtract each user's mean rating from their ratings (fixes harsh vs generous raters)."""
    R = R.tocsr().astype(np.float32)
    counts = np.diff(R.indptr)
    sums = np.asarray(R.sum(axis=1)).ravel()
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    C = R.copy()
    C.data = C.data - np.repeat(means, counts)
    return C, means


def weighted_rating(R, m=20):
    """Bayesian average: movies with few ratings are pulled toward the global mean."""
    counts = np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float32)
    sums = np.asarray(R.sum(axis=0)).ravel()
    global_mean = sums.sum() / counts.sum()
    item_mean = sums / np.maximum(counts, 1)
    return counts / (counts + m) * item_mean + m / (counts + m) * global_mean


def item_similarity(R, C):
    """Item-item cosine similarity with shrinkage, keeping only the top neighbours per item."""
    B = (R > 0).astype(np.float32)
    co = (B.T @ B).toarray()                       # how many users rated both items
    S = cosine_similarity(C.T)
    S = S * (co / (co + SHRINK))                   # shrinkage
    np.fill_diagonal(S, 0)
    S = np.maximum(S, 0)
    drop = np.argpartition(-S, NEIGHBORS, axis=1)[:, NEIGHBORS:]
    np.put_along_axis(S, drop, 0, axis=1)
    return sparse.csr_matrix(S)


def svd_model(C, k):
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(C)
    return U @ svd.components_


tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
G = tfidf.fit_transform(movies.genres)             # items x genres


def content_model(C):
    profile = normalize(C @ G)                     # user taste profile over genres
    return (profile @ G.T).toarray()


def all_scores(R, svd_k):
    C, means = center(R)
    S = item_similarity(R, C)
    scores = {
        "Random": np.random.default_rng(42).random(R.shape),
        "Popularity": np.tile(weighted_rating(R), (R.shape[0], 1)),
        "Content-based": content_model(C),
        "Item-based CF": (C @ S).toarray(),
        "SVD": svd_model(C, svd_k),
    }
    return scores, means


def zscore(S):
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)


def relevant_sets(d, thr=4.0):
    d = d[d.rating >= thr]
    return d.groupby("u")["i"].apply(set).to_dict()


def evaluate(scores, seen_R, rel):
    """Ranking metrics. Items the user already rated are never recommended."""
    S = scores.astype(np.float32).copy()
    r, c = seen_R.nonzero()
    S[r, c] = -np.inf
    top = np.argpartition(-S, K, axis=1)[:, :K]
    rows = np.arange(S.shape[0])[:, None]
    top = top[rows, np.argsort(-S[rows, top], axis=1)]
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    P, Rc, N, H = [], [], [], []
    for u, items in rel.items():
        hits = np.array([it in items for it in top[u]], dtype=float)
        h = hits.sum()
        P.append(h / K)
        Rc.append(h / len(items))
        H.append(float(h > 0))
        N.append((hits * disc).sum() / disc[: min(K, len(items))].sum())
    return {
        f"precision@{K}": float(np.mean(P)),
        f"recall@{K}": float(np.mean(Rc)),
        f"ndcg@{K}": float(np.mean(N)),
        f"hit_rate@{K}": float(np.mean(H)),
        "coverage": float(len(np.unique(top)) / S.shape[1]),
    }


# --------------------------------------------- 4. Tune on VALIDATION (never on test)
R_train = to_matrix(train)
rel_val = relevant_sets(val)
C_tr, _ = center(R_train)

ks = [10, 30, 50, 100]
svd_val = []
for k in ks:
    m = evaluate(svd_model(C_tr, k), R_train, rel_val)
    svd_val.append(m[f"ndcg@{K}"])
    print(f"SVD k={k}: val NDCG@{K} = {m[f'ndcg@{K}']:.4f}")
best_k = ks[int(np.argmax(svd_val))]
print("Best SVD components:", best_k)

plt.plot(ks, svd_val, marker="o")
plt.xlabel("SVD components (k)")
plt.ylabel(f"Validation NDCG@{K}")
plt.title("Tuning SVD")
plt.tight_layout()
plt.savefig("outputs/svd_tuning.png")
plt.close()

COMP = ["SVD", "Item-based CF", "Content-based", "Popularity"]
val_scores, _ = all_scores(R_train, best_k)
Zv = {n: zscore(val_scores[n]) for n in COMP}
best_w, best_nd = None, -1.0
for w in itertools.product([0, 0.5, 1], repeat=4):
    if sum(w) == 0:
        continue
    s = sum(wi * Zv[n] for wi, n in zip(w, COMP))
    nd = evaluate(s, R_train, rel_val)[f"ndcg@{K}"]
    if nd > best_nd:
        best_w, best_nd = w, nd
print("Best hybrid weights", dict(zip(COMP, best_w)), f"(val NDCG@{K} = {best_nd:.4f})")

# --------------------------------- 5. Final test: refit on train+val, test on last 20%
R_trval = to_matrix(pd.concat([train, val]))
rel_test = relevant_sets(test)
test_scores, means_tv = all_scores(R_trval, best_k)
Zt = {n: zscore(test_scores[n]) for n in COMP}
test_scores["Hybrid"] = sum(wi * Zt[n] for wi, n in zip(best_w, COMP))

res = pd.DataFrame({n: evaluate(s, R_trval, rel_test) for n, s in test_scores.items()}).T
res = res.sort_values(f"ndcg@{K}", ascending=False)
print("\nTEST RESULTS (ranking):")
print(res.round(4).to_string())

# rating-prediction accuracy (RMSE) for SVD vs simple baselines
gm = R_trval.sum() / R_trval.nnz
u_idx, i_idx, y_true = test.u.values, test.i.values, test.rating.values
svd_pred = np.clip(means_tv[u_idx] + test_scores["SVD"][u_idx, i_idx], 0.5, 5)
rmse = {
    "Global mean": float(np.sqrt(np.mean((y_true - gm) ** 2))),
    "User mean": float(np.sqrt(np.mean((y_true - means_tv[u_idx]) ** 2))),
    "SVD": float(np.sqrt(np.mean((y_true - svd_pred) ** 2))),
}
print("\nTEST RMSE:", {k: round(v, 3) for k, v in rmse.items()})

res[f"ndcg@{K}"].sort_values().plot(kind="barh", title=f"Test NDCG@{K} by model")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# ------------------------------------------ 6. Save artifacts for the Streamlit app
R_full = to_matrix(ratings)
C_full, _ = center(R_full)
S_full = item_similarity(R_full, C_full)
sparse.save_npz("outputs/item_sim.npz", S_full)

movies["n_ratings"] = np.asarray((R_full > 0).sum(axis=0)).ravel()
movies["avg_rating"] = (np.asarray(R_full.sum(axis=0)).ravel() / movies["n_ratings"]).round(2)
movies[["title", "genres", "n_ratings", "avg_rating"]].to_csv("outputs/movies.csv", index=False)

with open("outputs/metrics.json", "w") as f:
    json.dump({"best_svd_k": best_k, "hybrid_weights": dict(zip(COMP, best_w)),
               "test_ranking": res.to_dict(), "test_rmse": rmse}, f, indent=2)

print("\nDone! Everything saved in outputs/")