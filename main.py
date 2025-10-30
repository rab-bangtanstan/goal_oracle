import streamlit as st
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

st.set_page_config(page_title="GoalOracle ⚽", layout="wide")

# --- Helper functions ------------------------------------------------------

def calculate_score_probabilities(lambda_a, lambda_b, max_goals=8):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = poisson.pmf(i, lambda_a) * poisson.pmf(j, lambda_b)
    return matrix

def calculate_outcome_probabilities(prob_matrix):
    win_a = np.tril(prob_matrix, -1).sum()
    draw = np.trace(prob_matrix)
    win_b = np.triu(prob_matrix, 1).sum()
    return win_a, draw, win_b

def most_probable_score(prob_matrix):
    idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    return idx, prob_matrix[idx]

# --- Custom CSS ------------------------------------------------------------

st.markdown("""
<style>
.centered-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.input-header {
    text-align: center;
    color: #003366; 
    margin-top: -10px;
    margin-bottom: -5px;
}
.center-buttons {
    display: flex;
    justify-content: center;
    gap: 100px;
    margin-top: 30px;
    margin-bottom: 20px;
}
button[kind="secondary"], button[kind="primary"] {
    width: 600px !important;
    height: 60px !important;
    font-size: 20px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
button:hover {
    box-shadow: 0px 0px 20px rgba(255, 253, 208, 0.7) !important;
    background-color: #f5f5dc !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header Section (Visible Logo + Inline Title) --------------------------

import base64
from io import BytesIO

st.markdown("""
<style>
.header-inline {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    gap: 20px;
    margin-top: -65px;
}
.header-inline img {
    border-radius: 50%;
    border: 4px solid rgb(255, 253, 208);
}
.header-inline h1 {
    font-size: 2.4em;
    margin: 0;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

try:
    logo = Image.open("Goal Oracle.png").convert("RGBA")
    size = (130, 130)
    logo = logo.resize(size)

    # circular mask
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)
    logo.putalpha(mask)

    # create final framed logo on transparent background
    frame_size = 6
    frame_color = (255, 253, 208, 255)
    framed_size = (size[0] + frame_size * 2, size[1] + frame_size * 2)
    framed = Image.new("RGBA", framed_size, (0, 0, 0, 0))
    draw_frame = ImageDraw.Draw(framed)
    draw_frame.ellipse((0, 0, framed_size[0]-1, framed_size[1]-1), fill=frame_color)
    framed.paste(logo, (frame_size, frame_size), logo)

    # convert logo to base64
    buffered = BytesIO()
    framed.save(buffered, format="PNG")
    encoded_logo = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div class='header-inline'>
            <img src='data:image/png;base64,{encoded_logo}' width='130'>
            <h1>GoalOracle ⚽ — Predict the Unpredictable</h1>
        </div>
        <hr>
        """,
        unsafe_allow_html=True
    )

except Exception as e:
    st.error(f"Logo could not be displayed: {e}")


# --- Layout Columns --------------------------------------------------------

col1, col3 = st.columns(2)

with col1:
    st.markdown("<h3 class='input-header'>Team A — Inputs</h3>", unsafe_allow_html=True)
    ta_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=1.2, step=0.1, format="%.2f", key="ta_goals")
    ta_conceded = st.number_input("Goals Conceded", min_value=0.0, value=1.0, step=0.1)
    ta_sot = st.number_input("Shots on Target", min_value=0.0, value=3.0, step=0.1)
    ta_chances = st.number_input("Chances Created", min_value=0.0, value=5.0, step=0.1)
    ta_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=52.0, step=0.1)
    ta_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)

with col3:
    st.markdown("<h3 class='input-header'>Team B — Inputs</h3>", unsafe_allow_html=True)
    tb_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=1.0, step=0.1, format="%.2f", key="tb_goals")
    tb_conceded = st.number_input("Goals Conceded", min_value=0.0, value=1.1, step=0.1)
    tb_sot = st.number_input("Shots on Target", min_value=0.0, value=2.7, step=0.1)
    tb_chances = st.number_input("Chances Created", min_value=0.0, value=4.0, step=0.1)
    tb_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=48.0, step=0.1)
    tb_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=79.0, step=0.1)

st.markdown("<div class='center-buttons'>", unsafe_allow_html=True)
predict = st.button("Predict")
reset = st.button("Reset")
st.markdown("</div>", unsafe_allow_html=True)

# --- Logic -----------------------------------------------------------------

if reset:
    for k in ["ta_goals", "tb_goals"]:
        if k in st.session_state:
            st.session_state[k] = 0.0
    st.experimental_rerun()

if predict:
    try:
        lambda_a = float(ta_goals)
        lambda_b = float(tb_goals)
        if lambda_a < 0 or lambda_b < 0:
            raise ValueError("Lambdas must be non-negative")

        prob_matrix = calculate_score_probabilities(lambda_a, lambda_b, max_goals=8)
        win_a, draw, win_b = calculate_outcome_probabilities(prob_matrix)
        (best_i, best_j), best_p = most_probable_score(prob_matrix)

        st.subheader("Prediction Results")
        st.write(f"Most Probable Score: {best_i} - {best_j} ({best_p:.2%})")
        st.write(f"Team A Win: {win_a:.2%}   |   Draw: {draw:.2%}   |   Team B Win: {win_b:.2%}")
        st.markdown("---")

        fig, ax = plt.subplots()
        im = ax.imshow(prob_matrix, origin='lower', aspect='auto')
        ax.set_xlabel('Team B Goals')
        ax.set_ylabel('Team A Goals')
        ax.set_title('Score Probability Matrix')
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                p = prob_matrix[i, j]
                if p > 0.001:
                    ax.text(j, i, f"{p:.1%}", ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Invalid input detected: {e}")

st.markdown("---")
st.caption("GoalOracle — Poisson-based score prediction using the 'Goals Scored' inputs as λ for each team.")
