import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def split_genres(text):
    return text.split('|')

def load_artifacts(base_dir=BASE_DIR):
    """Memuat seluruh berkas data, model, dan konfigurasi dari direktori proyek."""
    # 1. Muat Konfigurasi
    config_path = os.path.join(base_dir, "models", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # 2. Muat Mood Lexicon
    lexicon_path = os.path.join(base_dir, "models", "mood_lexicon.json")
    with open(lexicon_path, "r") as f:
        mood_lexicon = json.load(f)
        
    # 3. Muat Cleaned Data
    data_path = os.path.join(base_dir, "data", "music_clean.csv")
    df_clean = pd.read_csv(data_path)
    df_clean['genre_list'] = df_clean['genres'].apply(lambda x: x.split('|'))
    
    # 4. Muat TF-IDF Vectorizer
    vec_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    with open(vec_path, "rb") as f:
        tfidf = pickle.load(f)
        
    # 5. Muat TF-IDF Matrix
    matrix_path = os.path.join(base_dir, "models", "tfidf_matrix.npz")
    tfidf_matrix = load_npz(matrix_path)
    
    # 6. Muat Rules Apriori
    rules_path = os.path.join(base_dir, "models", "rules.csv")
    rules = pd.read_csv(rules_path)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: set(x.split('|')))
    rules['consequents'] = rules['consequents'].apply(lambda x: set(x.split('|')))
    
    return {
        'config': config,
        'mood_lexicon': mood_lexicon,
        'df_clean': df_clean,
        'tfidf': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'rules': rules
    }

def build_query_vector(genre_weights, tfidf):
    q_vec = np.zeros(len(tfidf.vocabulary_))
    for genre, weight in genre_weights.items():
        if genre in tfidf.vocabulary_:
            idx = tfidf.vocabulary_[genre]
            q_vec[idx] = weight
    if np.linalg.norm(q_vec) > 0:
        q_vec = normalize(q_vec.reshape(1, -1))[0]
    return q_vec

def expand_genres(genre_list, rules_df, alpha, tfidf):
    expanded_weights = {g: 1.0 for g in genre_list if g in tfidf.vocabulary_}
    if len(rules_df) == 0 or alpha == 0:
        return expanded_weights, []

    g_set = set(genre_list)
    triggered_rules = []

    for _, row in rules_df.iterrows():
        ants = row['antecedents']
        cons = row['consequents']
        if ants.issubset(g_set):
            conf = row['confidence']
            lift = row['lift']
            for c in cons:
                if c not in g_set and c in tfidf.vocabulary_:
                    added_w = alpha * conf
                    if c not in expanded_weights or added_w > expanded_weights[c]:
                        expanded_weights[c] = added_w
                    ants_str = ", ".join(list(ants))
                    triggered_rules.append(f"{ants_str} -> {c}, lift {round(lift, 2)}")

    return expanded_weights, list(set(triggered_rules))

def recommend(art, song_label=None, genres=None, mood=None, weights=None, top_n=10, use_expansion=True, alpha=None, year_range=None, seed=42):
    # Ambil variable dari artifacts
    df_clean = art['df_clean']
    tfidf = art['tfidf']
    tfidf_matrix = art['tfidf_matrix']
    rules = art['rules']
    mood_lexicon = art['mood_lexicon']
    config = art['config']
    
    if alpha is None:
        alpha = config['ALPHA']

    if song_label is None and genres is None and mood is None:
        raise ValueError("Mesin Rekomendasi Terpadu membutuhkan minimal satu kueri input!")

    w_s = weights.get('song', 1.0) if weights else 1.0
    w_g = weights.get('genre', 1.0) if weights else 1.0
    w_m = weights.get('mood', 1.0) if weights else 1.0

    v_s_vec = np.zeros(len(tfidf.vocabulary_))
    v_g_vec = np.zeros(len(tfidf.vocabulary_))
    v_m_vec = np.zeros(len(tfidf.vocabulary_))

    input_idx = None
    input_title = ""
    input_year = None
    input_suspect = False
    reasons_comp = []

    # 1. Komponen Lagu
    if song_label is not None:
        indices = df_clean.index[df_clean['song_label'] == song_label]
        if len(indices) > 0:
            input_idx = indices[0]
            v_s_vec = tfidf_matrix[input_idx].toarray()[0]
            v_s_vec = normalize(v_s_vec.reshape(1, -1))[0]
            input_title = df_clean.loc[input_idx, 'title']
            input_year = df_clean.loc[input_idx, 'year']
            input_suspect = df_clean.loc[input_idx, 'year_suspect']
            reasons_comp.append(f"Kemiripan lagu dengan {song_label}")

    # 2. Komponen Genre
    if genres is not None:
        g_list = [g.strip() for g in genres.split('|')]
        g_weights, _ = expand_genres(g_list, rules, alpha if use_expansion else 0.0, tfidf)
        v_g_vec = build_query_vector(g_weights, tfidf)
        reasons_comp.append(f"Kesesuaian genre: {genres}")

    # 3. Komponen Mood
    keywords_list = []
    if mood is not None and mood in mood_lexicon:
        lex = mood_lexicon[mood]
        m_weights = {g: 1.0 for g in lex['genres_primary']}
        for g in lex['genres_secondary']:
            m_weights[g] = 0.5
        if use_expansion:
            m_weights, _ = expand_genres(list(m_weights.keys()), rules, alpha, tfidf)
        v_m_vec = build_query_vector(m_weights, tfidf)
        keywords_list = lex['keywords']
        reasons_comp.append(f"Cocok untuk suasana: {mood}")

    # Gabungkan kueri
    combined_q = (w_s * v_s_vec) + (w_g * v_g_vec) + (w_m * v_m_vec)
    if np.linalg.norm(combined_q) > 0:
        combined_q = normalize(combined_q.reshape(1, -1))[0]

    cosine_sims = linear_kernel(combined_q.reshape(1, -1), tfidf_matrix).ravel()

    # Jarak Tahun
    year_sims = np.zeros(len(df_clean))
    use_year_sim = (input_idx is not None)
    max_y, min_y = df_clean['year'].max(), df_clean['year'].min()
    range_y = max_y - min_y if max_y != min_y else 1
    
    if use_year_sim:
        if input_suspect:
            year_sims = np.full(len(df_clean), 0.5)
        else: 
            year_diffs = np.abs(df_clean['year'].values - input_year)
            year_sims = 1.0 - (year_diffs / range_y)
            year_sims = np.where(df_clean['year_suspect'].values, 0.5, year_sims)

    bonus_arr = np.zeros(len(df_clean))
    if mood is not None and keywords_list:
        pattern = "|".join([rf"\b{w}\b" for w in keywords_list])
        bonus_arr = np.where(df_clean['title'].str.lower().str.contains(pattern, regex=True, na=False), 0.05, 0.0)

    if use_year_sim:
        scores = config['W_GENRE'] * (cosine_sims + bonus_arr) + config['W_YEAR'] * year_sims
    else:
        scores = cosine_sims + bonus_arr

    res_df = df_clean.copy()
    res_df['score'] = scores

    if year_range is not None:
        res_df = res_df[(res_df['year'] >= year_range[0]) & (res_df['year'] <= year_range[1])]

    if input_idx is not None:
        res_df = res_df[res_df.index != input_idx]

    np.random.seed(seed)
    noise = np.random.uniform(0, 1e-5, len(res_df))
    res_df['score'] += noise

    res_df = res_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    res_df['rank'] = res_df.index + 1

    reasons_str = " · ".join(reasons_comp)
    res_df['alasan'] = reasons_str

    cols = ['rank', 'song_label', 'genres', 'year', 'score', 'alasan']
    return res_df[cols].head(top_n)

def get_rules_for_genre(art, genre, side="antecedent", top_n=5):
    rules = art['rules']
    if len(rules) == 0:
        return pd.DataFrame()
    if side == "antecedent":
        mask = rules['antecedents'].apply(lambda x: genre in x)
    elif side == "consequent":
        mask = rules['consequents'].apply(lambda x: genre in x)
    else:
        return pd.DataFrame()
    filtered = rules[mask].copy()
    if len(filtered) == 0:
        return pd.DataFrame()
    filtered['antecedents'] = filtered['antecedents'].apply(lambda x: ", ".join(list(x)))
    filtered['consequents'] = filtered['consequents'].apply(lambda x: ", ".join(list(x)))
    return filtered[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(top_n)
