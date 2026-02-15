import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Load Model
model = joblib.load("placement_model.pkl")

# Page Config
st.set_page_config(page_title="Smart Placement AI", layout="wide")

# Custom Header
st.markdown("""
    <h1 style='text-align:center; color:#1f4e79;'>
    🎓 AI Smart Campus Placement Intelligence Dashboard
    </h1>
    <hr>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🎓 Student Input Details")

with st.sidebar.form("student_form"):

    cgpa = st.number_input("CGPA", min_value=5.0, max_value=10.0, step=0.1)

    internships = st.number_input("Internships", min_value=0, max_value=5)

    projects = st.number_input("Projects", min_value=0, max_value=10)

    communication = st.number_input("Communication Skills (1-10)", 1, 10)

    resume_score = st.number_input("Resume Score (0-100)", 0, 100)

    submitted = st.form_submit_button("Predict Placement")

if submitted:
    input_data = [[cgpa, internships, projects, communication, resume_score]]
    prediction = model.predict_proba(input_data)[0][1] * 100

# Predict Button
if st.sidebar.button("🔮 Predict Placement"):

    input_data = pd.DataFrame({
        "CGPA": [cgpa],
        "Internships": [internships],
        "Projects": [projects],
        "Communication": [communication],
        "ResumeScore": [resume_score]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("## ✅ Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.success("🎉 Student is Likely to be Placed!")
        else:
            st.error("❌ Student Placement Chances Low")

    with col2:
       
    

     fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text': "Placement Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

   # Company Recommendation
if submitted:

    input_data = [[cgpa, internships, projects, communication, resume_score]]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    
    if probability*100 < 40:
        st.error("⚠ Improve CGPA above 7.5 and gain 2+ internships.")
    elif probability*100 < 70:
        st.warning("⚠ Moderate chances. Improve projects and resume score.")
    else:
        st.success("Excellent profile. High placement readiness!")



# Analytics Section
st.markdown("---")
st.markdown("## 📈 Placement Analytics Dashboard")

data = pd.read_csv("campus_placement.csv")

colA, colB = st.columns(2)

# Placement Distribution
with colA:
    st.subheader("✅ Placement Distribution")
    counts = data["Placed"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(counts, labels=["Placed", "Not Placed"], autopct="%1.1f%%")
    st.pyplot(fig)

# Resume Score Distribution
with colB:
    st.subheader("📄 Resume Score Analysis")
    fig, ax = plt.subplots()
    ax.hist(data["ResumeScore"], bins=5)
    st.pyplot(fig)
st.markdown("### 📊 Feature Importance Analysis")

import plotly.express as px

importances = model.feature_importances_

features = ["CGPA","Internships","Projects","Communication","ResumeScore"]

fig_importance = px.bar(
    x=features,
    y=importances,
    title="Feature Importance (Model Insight)",
    labels={'x': 'Features', 'y': 'Importance Score'}
)

# Professional upgrade
fig_importance.update_layout(
    xaxis_title="Features",
    yaxis_title="Importance Score",
    template="plotly_white",
    title_x=0.5
)

st.plotly_chart(fig_importance, use_container_width=True)
st.markdown("### 🔥 Correlation Heatmap")

corr = data.corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu",
    title="Feature Correlation Matrix"
)
fig_corr.update_layout(
    template="plotly_white",
    title_x=0.5
)


st.plotly_chart(fig_corr, use_container_width=True)


st.markdown("---")
st.markdown("✅ Final Year AI Dashboard Project Ready 🎓")


