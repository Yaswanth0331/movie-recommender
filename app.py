import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")


@st.cache_resource
def load():
    movies = pd.read_csv("outputs/movies.csv")
    S = sparse.load_npz("outputs/item_sim.npz").tocsr()
    return movies, S


movies, S = load()

st.title("🎬 Movie Recommender")
st.write("Pick movies you like and get recommendations from item-based collaborative filtering trained on MovieLens.")

picked = st.multiselect("Movies you like (choose 3 or more for better results)", movies["title"].tolist())
top_n = st.slider("Number of recommendations", 5, 20, 10)
min_ratings = st.slider("Minimum number of ratings a movie must have", 5, 100, 20)

if st.button("Recommend"):
    if not picked:
        st.warning("Please pick at least one movie.")
    else:
        idx = movies.index[movies["title"].isin(picked)].to_numpy()
        scores = np.asarray(S[idx].sum(axis=0)).ravel().astype(float)
        scores[idx] = -np.inf
        scores[(movies["n_ratings"] < min_ratings).to_numpy()] = -np.inf
        order = np.argsort(-scores)[:top_n]
        out = movies.iloc[order].copy()
        out["match score"] = scores[order].round(3)
        out = out[out["match score"] > 0]
        if out.empty:
            st.info("No strong matches found. Try different movies or lower the minimum ratings.")
        else:
            st.dataframe(out[["title", "genres", "avg_rating", "n_ratings", "match score"]],
                         hide_index=True, use_container_width=True)