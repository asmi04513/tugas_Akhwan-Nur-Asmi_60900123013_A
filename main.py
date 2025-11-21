import streamlit as st
import joblib
import re

# -----------------------------
# KONFIGURASI HALAMAN
# -----------------------------
st.set_page_config(
    page_title="Analisis Sentimen Film",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# LOAD MODEL & TOOLS
# -----------------------------
@st.cache_resource
def load_model_objects():
    try:
        model_bnb = joblib.load("model_bernoulli_nb.pkl")
        model_svm = joblib.load("model_linear_svm.pkl")
        model_ensemble = joblib.load("model_ensemble_voting.pkl")
        vectorizer = joblib.load("vectorizer_tfidf.pkl")
        tools = joblib.load("preprocessing_tools.pkl")
        return model_bnb, model_svm, model_ensemble, vectorizer, tools
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None, None, None, None

model_bnb, model_svm, model_ensemble, vectorizer, tools = load_model_objects()

# -----------------------------
# PREPROCESSING TEKS
# -----------------------------
def preprocess_text(text, stopword_remover, stemmer):
    text = re.sub(r'[^A-Za-z]+', ' ', text).lower().strip()
    text = re.sub(r'\s+', ' ', text)   # FIX WARNING
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

# -----------------------------
# CONFIDENCE BADGE
# -----------------------------
def get_confidence_badge(prob):
    if prob > 80:
        return "🟢 Tinggi", "success"
    elif prob > 60:
        return "🟡 Sedang", "warning"
    else:
        return "🔴 Rendah", "error"

# -----------------------------
# UI UTAMA
# -----------------------------
st.title("🎬 Analisis Sentimen Film")
st.markdown("### Model Ensemble (BernoulliNB + SVM)")

models_loaded = all([model_bnb, model_svm, model_ensemble, vectorizer, tools])

if not models_loaded:
    st.error("⚠ File model atau vectorizer tidak ditemukan.")
else:
    st.subheader("✍ Masukkan Ulasan Film")

    example_texts = [
        "Filmnya bagus banget, alurnya tidak ketebak!",
        "Film jelek, buang waktu saja",
        "Keren, aktingnya mantap sekali",
        "Goblok banget filmnya tidak bermutu",
        "Biasa aja sih, tidak terlalu bagus",
        "Luar biasa, sangat recommended!"
    ]

    selected_example = st.selectbox(
        "Pilih contoh ulasan:",
        ["-- Ketik manual --"] + example_texts
    )

    input_text = "" if selected_example == "-- Ketik manual --" else selected_example

    user_input = st.text_area("Masukkan ulasan film:", value=input_text, height=100)

    col1, col2, col3 = st.columns(3)
    with col1:
        predict_btn = st.button("🔍 Analisis", type="primary")
    with col2:
        show_comparison = st.checkbox("Bandingkan model", value=True)
    with col3:
        show_details = st.checkbox("Detail preprocessing", value=False)

    # -----------------------------
    # PREDIKSI
    # -----------------------------
    if predict_btn:
        if user_input.strip() == "":
            st.warning("⚠ Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menganalisis..."):
                try:
                    stopword_remover = tools['stopword']
                    stemmer = tools['stemmer']

                    processed = preprocess_text(user_input, stopword_remover, stemmer)
                    vec = vectorizer.transform([processed])

                    # Prediksi
                    pred_ensemble = model_ensemble.predict(vec)[0]
                    prob_ensemble = model_ensemble.predict_proba(vec)[0]

                    pred_bnb = model_bnb.predict(vec)[0]
                    pred_svm = model_svm.predict(vec)[0]

                    # Tampilan utama
                    st.subheader("🎯 Hasil Analisis (Ensemble)")

                    max_prob = max(prob_ensemble) * 100
                    conf_text, conf_type = get_confidence_badge(max_prob)

                    if pred_ensemble == "positive":
                        st.success("### ✅ Sentimen: POSITIF")
                    else:
                        st.error("### ❌ Sentimen: NEGATIF")

                    st.info(f"**Tingkat Keyakinan:** {conf_text} ({max_prob:.1f}%)")

                    # Probabilitas
                    st.write("📊 Probabilitas:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Negatif", f"
