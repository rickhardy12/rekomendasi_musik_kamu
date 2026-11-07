import streamlit as st
import pandas as pd
import numpy as np
from recommender import load_artifacts, recommend, get_rules_for_genre

st.set_page_config(page_title="Rekomendasi Musik", page_icon="🎵", layout="centered")

# 1. Pemuatan Artefak Secara Ter-cache
@st.cache_resource
def get_cached_artifacts():
    return load_artifacts()

try:
    art = get_cached_artifacts()
except Exception as e:
    st.error(f"Gagal memuat data sistem: {e}")
    st.stop()

# Inisialisasi session state
if "halaman" not in st.session_state:
    st.session_state["halaman"] = "Beranda"
if "mode" not in st.session_state:
    st.session_state["mode"] = "Lagu"
if "hasil" not in st.session_state:
    st.session_state["hasil"] = None
if "seed_offset" not in st.session_state:
    st.session_state["seed_offset"] = 42

# Navigasi atas
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    if st.button("🏠 Beranda", use_container_width=True):
        st.session_state["halaman"] = "Beranda"
        st.session_state["hasil"] = None
        st.rerun()
with col_nav2:
    if st.button("🔍 Cari Lagu", use_container_width=True):
        st.session_state["halaman"] = "Cari Lagu"
        st.rerun()
with col_nav3:
    if st.button("ℹ️ Tentang", use_container_width=True):
        st.session_state["halaman"] = "Tentang"
        st.rerun()

st.markdown("--- ")

# ==========================================
# HALAMAN: BERANDA
# ==========================================
if st.session_state["halaman"] == "Beranda":
    st.title("🎵 Selamat Datang di Rekomendasi Musik Kamu")
    st.markdown("#### Temukan lagu yang paling cocok dengan seleramu.")
    
    st.write("Silakan pilih pintu masuk pencarian di bawah ini:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**💿 Dari Lagu Favorit**")
        if st.button("Cari Lewat Lagu", use_container_width=True):
            st.session_state["halaman"] = "Cari Lagu"
            st.session_state["mode"] = "Lagu"
            st.rerun()
            
        st.info("**🎸 Dari Genre**")
        if st.button("Cari Lewat Genre", use_container_width=True):
            st.session_state["halaman"] = "Cari Lagu"
            st.session_state["mode"] = "Genre"
            st.rerun()
            
    with col2:
        st.info("**💭 Dari Suasana Hati**")
        if st.button("Cari Lewat Mood", use_container_width=True):
            st.session_state["halaman"] = "Cari Lagu"
            st.session_state["mode"] = "Mood"
            st.rerun()
            
        st.info("**🧪 Racik Sendiri**")
        if st.button("Kombinasikan Semua", use_container_width=True):
            st.session_state["halaman"] = "Cari Lagu"
            st.session_state["mode"] = "Perpaduan"
            st.rerun()
            
    st.markdown("--- ")
    total_lagu = len(art['df_clean'])
    min_y = int(art['df_clean']['year'].min())
    max_y = int(art['df_clean']['year'].max())
    st.write(f"📊 **Katalog Saat Ini:** Menyediakan **{total_lagu:,}** pilihan lagu berkualitas dari tahun rilis era **{min_y}** hingga **{max_y}**.")

# ==========================================
# HALAMAN: CARI LAGU
# ==========================================
elif st.session_state["halaman"] == "Cari Lagu":
    st.title("🔍 Temukan Rekomendasi Lagumu")
    
    # Segmented control atau radio
    try:
        mode_pilihan = st.segmented_control(
            "Pilih Mode Pencarian",
            options=["Lagu", "Genre", "Mood", "Perpaduan"],
            default=st.session_state["mode"]
        )
        if mode_pilihan:
            st.session_state["mode"] = mode_pilihan
    except AttributeError:
        mode_pilihan = st.radio(
            "Pilih Mode Pencarian",
            options=["Lagu", "Genre", "Mood", "Perpaduan"],
            index=["Lagu", "Genre", "Mood", "Perpaduan"].index(st.session_state["mode"]),
            horizontal=True
        )
        st.session_state["mode"] = mode_pilihan

    # Setup Parameter dari Config
    config = art['config']
    low_threshold = config['LOW_MATCH_THRESHOLD']
    
    # Menyiapkan Opsi Tampilan Judul
    # "Title · Year · Genres" untuk membedakan judul kembar tanpa ID
    df_clean = art['df_clean']
    display_to_label = {}
    label_options = []
    for idx, row in df_clean.iterrows():
        disp_text = f"{row['title']} · {row['year'] if not row['year_suspect'] else 'tahun belum pasti'} · {row['genres'].replace('|', ', ')}"
        display_to_label[disp_text] = row['song_label']
        label_options.append(disp_text)
        
    # Input Skenario
    q_song = None
    q_genre = None
    q_mood = None
    weights = {'song': 1.0, 'genre': 1.0, 'mood': 1.0}
    
    st.markdown("### ")
    
    if st.session_state["mode"] == "Lagu":
        pilihan_disp = st.selectbox("Ketik judul lagu favoritmu", options=label_options)
        if pilihan_disp:
            q_song = display_to_label[pilihan_disp]
            
    elif st.session_state["mode"] == "Genre":
        flat_genres = [g for sublist in df_clean['genre_list'] for g in sublist]
        pop_genres = list(pd.Series(flat_genres).value_counts().index)
        
        st.session_state["g_select"] = st.multiselect("Pilih genre favoritmu (maksimal 3)", options=pop_genres, max_selections=3)
        
        if st.session_state["g_select"]:
            q_genre = "|".join(st.session_state["g_select"])
            
            # Chip rekomendasi padanan
            last_g = st.session_state["g_select"][-1]
            rules_g = get_rules_for_genre(art, last_g, side="antecedent", top_n=3)
            if len(rules_g) > 0:
                st.write("💡 *Sering dipadukan dengan:* ")
                for _, r in rules_g.iterrows():
                    conseqs = r['consequents'].split(', ')
                    for c in conseqs:
                        if c not in st.session_state["g_select"]:
                            if st.button(f"➕ {c}", key=f"chip_{c}"):
                                if len(st.session_state["g_select"]) < 3:
                                    st.session_state["g_select"].append(c)
                                    st.rerun()
                                    
    elif st.session_state["mode"] == "Mood":
        st.write("Pilih suasana hatimu saat ini:")
        moods = list(art['mood_lexicon'].keys())
        emoji_map = {
            'Romantis': '💖 Romantis', 'Ceria': '☀️ Ceria', 'Sedih': '😢 Sedih',
            'Energik': '⚡ Energik', 'Santai': '🍃 Santai', 'Nostalgia': '⏳ Nostalgia'
        }
        cols_mood = st.columns(3)
        for i, m in enumerate(moods):
            with cols_mood[i % 3]:
                if st.button(emoji_map.get(m, m), use_container_width=True):
                    st.session_state["selected_mood"] = m
                    
        selected_m = st.session_state.get("selected_mood", "Romantis")
        st.info(f"Mood aktif: **{selected_m}**")
        q_mood = selected_m
        
    elif st.session_state["mode"] == "Perpaduan":
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.write("**💿 Lagu Kueri**")
            f_song_on = st.checkbox("Aktifkan Lagu", value=True)
            f_song_disp = st.selectbox("Pilih Lagu kueri", options=label_options, label_visibility="collapsed")
            f_song_weight = st.select_slider("Bobot Lagu", options=["Rendah", "Sedang", "Tinggi"], value="Sedang", key="w_s")
            
        with col_f2:
            st.write("**🌈 Genre Kueri**")
            f_genre_on = st.checkbox("Aktifkan Genre", value=False)
            flat_genres = [g for sublist in df_clean['genre_list'] for g in sublist]
            pop_genres = list(pd.Series(flat_genres).value_counts().index)
            f_genre_sel = st.multiselect("Pilih Genre", options=pop_genres, max_selections=3, label_visibility="collapsed")
            f_genre_weight = st.select_slider("Bobot Genre", options=["Rendah", "Sedang", "Tinggi"], value="Sedang", key="w_g")
            
        with col_f3:
            st.write("**💭 Mood Kueri**")
            f_mood_on = st.checkbox("Aktifkan Mood", value=False)
            moods = list(art['mood_lexicon'].keys())
            f_mood_sel = st.selectbox("Pilih Mood", options=moods, label_visibility="collapsed")
            f_mood_weight = st.select_slider("Bobot Mood", options=["Rendah", "Sedang", "Tinggi"], value="Sedang", key="w_m")
            
        # Map bobot
        weight_map = {"Rendah": config['LOW_MATCH_THRESHOLD'], "Sedang": 1.0, "Tinggi": 2.0}
        
        if f_song_on:
            q_song = display_to_label[f_song_disp]
            weights['song'] = weight_map[f_song_weight]
        if f_genre_on and f_genre_sel:
            q_genre = "|".join(f_genre_sel)
            weights['genre'] = weight_map[f_genre_weight]
        if f_mood_on:
            q_mood = f_mood_sel
            weights['mood'] = weight_map[f_mood_weight]

    st.markdown("--- ")
    
    # Pilihan Umum
    st.write("**⚙️ Pengaturan Tambahan**")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        era_range = st.slider(
            "Batasi rentang tahun rilis",
            min_value=1926, max_value=2010, value=(1926, 2010)
        )
        n_limit = st.radio("Jumlah rekomendasi", options=[5, 10, 20], index=1, horizontal=True)
    with col_opt2:
        use_exp = st.checkbox("Sertakan genre ber-asosiasi kuat (Perluasan Genre)", value=True)
        include_suspect = st.checkbox("Sertakan lagu yang tahun rilisnya belum pasti", value=True)

    # Aksi Cari
    if st.button("🔥 Temukan Musik Relevan", use_container_width=True):
        if st.session_state["mode"] == "Perpaduan" and not (q_song or q_genre or q_mood):
            st.warning("Isi minimal satu: lagu, genre, atau mood.")
        else:
            # Hitung rekomendasi
            res_df = recommend(
                art,
                song_label=q_song,
                genres=q_genre,
                mood=q_mood,
                weights=weights if st.session_state["mode"] == "Perpaduan" else None,
                top_n=n_limit,
                use_expansion=use_exp,
                year_range=era_range,
                seed=st.session_state["seed_offset"]
            )
            
            if not include_suspect and len(res_df) > 0:
                # Filter out suspects jika checkbox tidak aktif
                # Ambil song_labels original tanpa suspect
                clean_songs = df_clean[~df_clean['year_suspect']]['song_label']
                res_df = res_df[res_df['song_label'].isin(clean_songs)].reset_index(drop=True)
                res_df['rank'] = res_df.index + 1
                
            st.session_state["hasil"] = res_df

    # Menampilkan Hasil Rekomendasi
    if st.session_state["hasil"] is not None:
        res = st.session_state["hasil"].head(n_limit)
        if len(res) == 0:
            st.info("Tidak ditemukan lagu yang cocok dengan kriteria filter tersebut.")
        else:
            max_score = res['score'].max()
            
            # Cek threshold kesesuaian rendah
            if max_score < low_threshold:
                st.warning("⚠️ Kombinasi ini jarang ada di katalog. Berikut lagu yang paling mendekati:")
            else:
                st.success("✅ Daftar lagu rekomendasi terbaik untukmu:")
                
            for _, r in res.iterrows():
                # Normalisasi kecocokan jika score > 1
                fit_pct = int(min(1.0, r['score'] / max_score if max_score > 1.0 else r['score']) * 100)
                
                # Tampilan label tahun
                df_row = df_clean[df_clean['song_label'] == r['song_label']].iloc[0]
                thn_disp = "tahun belum pasti" if df_row['year_suspect'] else str(r['year'])
                
                # Render kartu
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background-color:#F1F5F9; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                            <h4 style="margin:0; color:#0F172A;">🎵 {df_row['title']}</h4>
                            <p style="margin:5px 0; color:#475569; font-size:0.9em;">⏱️ Rilis: {thn_disp} · 🏷️ Genre: {r['genres'].replace('|', ', ')}</p>
                            <p style="margin:5px 0; color:#2563EB; font-weight:bold;">📊 Persentase Kecocokan: {fit_pct}%</p>
                            <p style="margin:5px 0 0 0; color:#0F172A; font-style:italic; font-size:0.95em;">💬 {r['alasan']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Tombol aksi hasil
            st.write(" ")
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                if st.button("🔄 Coba daftar lain", use_container_width=True):
                    st.session_state["seed_offset"] += 1
                    st.session_state["hasil"] = None
                    st.rerun()
            with col_act2:
                # Siapkan file CSV untuk diunduh
                csv_df = df_clean[df_clean['song_label'].isin(res['song_label'])][['title', 'year', 'genres']].copy()
                csv_data = csv_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh daftar (CSV)",
                    data=csv_data,
                    file_name="rekomendasi_musik.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
    # Catatan kaki untuk mood
    if st.session_state["mode"] in ["Mood", "Perpaduan"]:
        st.markdown("*Catatan kecil: Rekomendasi mood disusun dari genre lagu, bukan dari analisis audio.*")

# ==========================================
# HALAMAN: TENTANG
# ==========================================
elif st.session_state["halaman"] == "Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini memberikan rekomendasi musik pintar berbasis kemiripan karakteristik genre lagu serta pendekatan tahun rilis sebagai pemecah keselarasan.
    Melalui sistem matematika terpadu, sistem ini mampu memperluas pencarian genre secara cerdas memanfaatkan pola aturan asosiasi data historis.
    Seluruh proses kalkulasi penentuan kesesuaian berjalan secara efisien langsung di sisi server untuk memberikan respons jawaban seketika.
    """)
    
    st.write("🔗 **Tautan Repositori Kode:** [Link GitHub](https://github.com/rickhardy12/rekomendasi_musik_kamu)")
