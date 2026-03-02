import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# GITHUB REPO LINK

GITHUB_REPO = "https://github.com/DeepakPr2005/SHL-Assessment-Recommendation-Engine-using-product-catalogue"
# =====================================

# =====================================
# TOP GITHUB BUTTON
# =====================================
header_col1, header_col2 = st.columns([13, 5])

with header_col2:
    st.write("")      # vertical spacing
    st.write("")      # adjust alignment
    st.link_button("🐙 Project GitHub repo ", GITHUB_REPO)


# =====================================
# LOAD DATA + MODELS
# =====================================
df = pd.read_csv("model/shl_feature_engineered.csv")

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)



# =====================================
# SKILL NORMALIZATION
# =====================================
skill_map = {
    "machine learning": "ml",
    "deep learning": "deep",
    "natural language processing": "nlp",
    "artificial intelligence": "ai",
}


# =====================================
# RECOMMENDATION ENGINE
# =====================================
def recommend_assessments(user_input, top_n=5):

    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(
        user_vector, tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    results = df.iloc[top_indices].copy()

    # ---------------------------------
    # NORMALIZED MATCH SCORE
    # ---------------------------------
    raw_scores = similarity_scores[top_indices]

    normalized_scores = 70 + (
        (raw_scores - raw_scores.min())
        / (raw_scores.max() - raw_scores.min() + 1e-6)
    ) * 30

    results["Match_Score (%)"] = normalized_scores.round(2)

    # ---------------------------------
    # MATCHED SKILLS
    # ---------------------------------
    user_tokens = set(user_input.lower().split())

    def find_matches(skill_string):
        dataset_skills = set(skill_string.lower().split(","))
        matches = dataset_skills.intersection(user_tokens)

        if matches:
            return ", ".join(matches)
        else:
            return "No direct match"

    results["Matched_Skills"] = results["Skills"].apply(find_matches)

    results = results.sort_values(
        "Match_Score (%)", ascending=False
    )

    return results[
        [
            "Assessment",
            "Role",
            "Skills",
            "Matched_Skills",
            "Duration",
            "Difficulty",
            "Match_Score (%)",
        ]
    ]


# =====================================
# STREAMLIT UI
# =====================================
st.title("SHL Assessment Recommendation Engine")

st.write(
    "Enter candidate/job requirements to receive recommended SHL assessments."
)

role = st.text_input("Role")
skills = st.text_input("Skills (comma separated)")
level = st.selectbox("Level", ["Entry", "Mid", "Senior", "All"])
industry = st.text_input("Industry")

top_n = st.slider("Number of Recommendations", 1, 10, 5)


# =====================================
# BUTTON ACTION
# =====================================
if st.button("Recommend Assessments"):

    if role == "" or skills == "":
        st.warning("Please enter Role and Skills")

    else:
        # normalize user skills
        skills_clean = skills.lower()

        for k, v in skill_map.items():
            skills_clean = skills_clean.replace(k, v)

        skills_clean = skills_clean.replace(",", " ")

        user_profile = (
            f"{role.lower()} {skills_clean} {level.lower()} {industry.lower()}"
        )

        results = recommend_assessments(user_profile, top_n)

        st.success("Recommended Assessments")

        st.dataframe(results)

        st.info(
            "Recommendations generated using NLP TF-IDF vectorization and cosine similarity."
        )


        # py -m streamlit run app/app.py

# git init
# git init
# git commit -m "SHL Assessment Recommendation Engine"
# git branch -M main
# git remote add origin <github-repo>
# git push -u origin main