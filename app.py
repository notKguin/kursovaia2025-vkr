# app.py
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Прогноз отказов оборудования",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ======================
# LOADERS
# ======================
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("lr_model.pkl"),
        "Random Forest": joblib.load("rf_model.pkl"),
        "Gradient Boosting": joblib.load("gb_model.pkl"),
        "MLP (Neural Network)": joblib.load("mlp_model.pkl"),
    }


@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")


@st.cache_resource
def load_features():
    return joblib.load("top_features.pkl")


@st.cache_data
def load_data():
    return joblib.load("df_lagged_test.pkl")


models = load_models()
scaler = load_scaler()
features = load_features()
df = load_data()

# ======================
# DATA
# ======================
X = df[features]
y_true = df["label_future"].astype(int).values
X_scaled = scaler.transform(X)

# ======================
# UTILS
# ======================
def find_best_threshold_f1(y_true, y_prob):
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t, best_f1


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    roc_auc = roc_auc_score(y_true, y_prob)

    return precision, recall, f1, roc_auc, tp, fp, fn, tn


def plot_probabilities(y_prob, threshold):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.plot(y_prob, alpha=0.7)
    ax.axhline(threshold, linestyle="--")
    ax.set_xlabel("Временной шаг")
    ax.set_ylabel("P(failure)")
    st.pyplot(fig)


def get_feature_importance(name, model):
    """
    Возвращает DataFrame с важностями признаков
    """
    if name == "Logistic Regression":
        importance = np.abs(model.coef_[0])

    elif name in ["Random Forest", "Gradient Boosting"]:
        importance = model.feature_importances_

    elif name == "MLP (Neural Network)":
        perm = permutation_importance(
            model,
            X_scaled,
            y_true,
            n_repeats=5,
            random_state=42,
            scoring="f1",
        )
        importance = perm.importances_mean

    else:
        return None

    df_imp = pd.DataFrame({
        "Признак": features,
        "Важность": importance
    }).sort_values("Важность", ascending=False)

    return df_imp.head(5)


# ======================
# MODEL BLOCK
# ======================
def model_block(name, model):
    y_prob = model.predict_proba(X_scaled)[:, 1]

    best_threshold, best_f1 = find_best_threshold_f1(y_true, y_prob)

    threshold = st.slider(
        "Порог вероятности",
        min_value=0.0,
        max_value=1.0,
        value=best_threshold,
        step=0.01,
        key=f"slider_{name}",
    )

    precision, recall, f1, roc_auc, tp, fp, fn, tn = compute_metrics(
        y_true, y_prob, threshold
    )

    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{roc_auc:.3f}")
    c2.metric("Precision", f"{precision:.3f}")
    c3, c4 = st.columns(2)
    c3.metric("Recall", f"{recall:.3f}")
    c4.metric("F1-score", f"{f1:.3f}")

    st.markdown(
        f"""
        **🔎 Лучший порог по F1:** `{best_threshold:.2f}`  
        **🏆 Максимальное F1:** `{best_f1:.3f}`
        """
    )

    st.markdown(
        f"""
        ✔ **TP:** {tp}  ✔ **TN:** {tn}\n\n✖ **FP:** {fp}   ✖ **FN:** {fn} 
        \n\n**Всего наблюдений:** {tp + fp + fn + tn}
        """
    )

    plot_probabilities(y_prob, threshold)

    # ===== FEATURE IMPORTANCE =====
    st.markdown("**🔑 Топ-5 важных признаков:**")
    imp_df = get_feature_importance(name, model)
    st.dataframe(
        imp_df,
        width='stretch',
        hide_index=True,
    )


# ======================
# LAYOUT
# ======================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Logistic Regression")
    model_block("Logistic Regression", models["Logistic Regression"])

with col2:
    st.subheader("Random Forest")
    model_block("Random Forest", models["Random Forest"])

with col3:
    st.subheader("Gradient Boosting")
    model_block("Gradient Boosting", models["Gradient Boosting"])

with col4:
    st.subheader("MLP (Neural Network)")
    model_block("MLP (Neural Network)", models["MLP (Neural Network)"])
