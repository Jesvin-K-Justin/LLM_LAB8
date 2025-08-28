# evaluation.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from collections import Counter
import numpy as np
import pandas as pd

# ================================
# 🎯 Example Ground Truth & Model Responses (English, French, Spanish)
# ================================

examples = [
    {
        "query": "What is Artificial Intelligence?",
        "ground_truth": "Artificial Intelligence is the field of computer science that creates systems capable of performing tasks that require human-like intelligence, such as reasoning, learning, and problem-solving.",
        "model_response": "AI is a branch of computer science focused on making machines that can learn, reason, and act like humans."
    },
    {
        "query": "Traduisez 'Bonjour' en anglais.",
        "ground_truth": "Hello",
        "model_response": "Hello"
    },
    {
        "query": "¿Cuál es la capital de España?",
        "ground_truth": "La capital de España es Madrid.",
        "model_response": "Madrid es la capital de España."
    },
    {
        "query": "Résumez : L'internet connecte les ordinateurs à l'échelle mondiale, permettant la communication.",
        "ground_truth": "L'internet permet la communication mondiale entre ordinateurs.",
        "model_response": "Il connecte les ordinateurs du monde entier pour communiquer."
    },
    {
        "query": "Summarize: The internet connects computers globally allowing communication.",
        "ground_truth": "The internet enables worldwide computer communication.",
        "model_response": "It connects computers worldwide for communication."
    },
]

# ================================
# 📊 Evaluation Metrics
# ================================

def compute_f1(gt, pred):
    """Token-level F1 Score"""
    gt_tokens, pred_tokens = gt.split(), pred.split()
    common = Counter(gt_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(gt, pred):
    """Simplified BLEU (precision-based unigram)"""
    gt_tokens, pred_tokens = gt.split(), pred.split()
    overlap = len(set(gt_tokens) & set(pred_tokens))
    return overlap / max(len(pred_tokens), 1)


def compute_rouge_l(gt, pred):
    """Longest Common Subsequence (ROUGE-L)"""
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if gt[i] == pred[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[m][n]
    return lcs / max(len(gt), 1)


# ================================
# 📊 Streamlit Dashboard
# ================================
st.set_page_config(page_title="Evaluation Dashboard", layout="wide")

# Custom CSS
st.markdown("""
<style>
    body {
        background-color: #f4f6f9;
    }
    .stMetric {
        background: #ffffff;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .main-title {
        text-align: center;
        font-size: 35px;
        font-weight: 800;
        color: #2e86de;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 DeepSeek RAG Agent Evaluation Dashboard (EN/FR/ES)</p>', unsafe_allow_html=True)

# ================================
# 🔄 Run Evaluations
# ================================
results = []
for ex in examples:
    start = time.time()
    f1 = compute_f1(ex["ground_truth"], ex["model_response"])
    bleu = compute_bleu(ex["ground_truth"], ex["model_response"])
    rouge = compute_rouge_l(ex["ground_truth"], ex["model_response"])
    latency = round(random.uniform(0.8, 2.5), 2)  # Simulated latency
    end = time.time()

    results.append({
        "Query": ex["query"],
        "Ground Truth": ex["ground_truth"],
        "Model Response": ex["model_response"],
        "F1 Score": round(f1, 3),
        "BLEU": round(bleu, 3),
        "ROUGE-L": round(rouge, 3),
        "Response Time (s)": latency
    })

# ================================
# 📑 Show Results Table
# ================================
df = pd.DataFrame(results)
st.subheader("📑 Evaluation Results")
st.dataframe(df, use_container_width=True)

# ================================
# 📈 Visualizations
# ================================
st.subheader("📊 Metrics Visualization")

col1, col2 = st.columns(2)

with col1:
    st.write("### Average Scores")
    avg_f1 = np.mean(df["F1 Score"])
    avg_bleu = np.mean(df["BLEU"])
    avg_rouge = np.mean(df["ROUGE-L"])

    fig, ax = plt.subplots()
    sns.barplot(x=["F1", "BLEU", "ROUGE-L"], y=[avg_f1, avg_bleu, avg_rouge], ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Average Evaluation Metrics")
    st.pyplot(fig)

with col2:
    st.write("### Latency Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Response Time (s)"], bins=5, kde=True, ax=ax)
    ax.set_title("Response Time Distribution")
    st.pyplot(fig)

# ================================
# 🎯 Final Summary
# ================================
st.subheader("📌 Summary")
st.markdown(f"""
- ✅ **Average F1 Score:** {avg_f1:.3f}  
- ✅ **Average BLEU:** {avg_bleu:.3f}  
- ✅ **Average ROUGE-L:** {avg_rouge:.3f}  
- ⚡ **Average Response Time:** {df['Response Time (s)'].mean():.2f} sec  
""")
