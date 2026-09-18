
import streamlit as st
import pandas as pd
import numpy as np
import joblib



st.set_page_config(
    page_title="Lecture Engagement Pattern Analyzer",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_models():
    classification_model = joblib.load("classification_model.pkl")
    regression_model = joblib.load("regression_model.pkl" )
    return classification_model, regression_model

try: classification_model, regression_model = load_models()

except Exception as e:
    st.error(
        "Model files could not be loaded. "
        "Make sure classification_model.pkl and "
        "regression_model.pkl are in the same folder as app.py."
    )

    st.stop()


st.title("🎓 Lecture Engagement Pattern Analyzer")

st.write(
    "Enter the student's lecture-learning behavior to predict "
    "their performance category and expected next quiz score."
)

st.divider()



# INPUT SECTION

st.header("📊 Student Engagement Information")
col1, col2, col3 = st.columns(3)

with col1:

    video_completion_pct = st.number_input(
        "Video Completion (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    lecture_duration_min = st.number_input(
        "Lecture Duration (minutes)",
        min_value=1.0,
        max_value=300.0,
        value=45.0,
        step=1.0
    )

    pause_count = st.number_input(
        "Pause Count",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    rewatch_count = st.number_input(
        "Rewatch Count",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )



with col2:

    quiz_attempts = st.number_input(
        "Quiz Attempts",
        min_value=1,
        max_value=50,
        value=2,
        step=1
    )

    quiz_score_pct = st.number_input(
        "Current Quiz Score (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    notes_taken = st.selectbox(
        "Notes Taken",
        [ "No","Sometimes","Yes"]
    )

    questions_asked = st.number_input(
        "Questions Asked",
        min_value=0,
        max_value=50,
        value=2,
        step=1
    )



with col3:

    login_delay_min = st.number_input(
        "Login Delay (minutes)",
        min_value=0.0,
        max_value=180.0,
        value=5.0,
        step=1.0
    )

    sessions_per_week = st.number_input(
        "Study Sessions / Week",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    screen_time_hours = st.number_input(
        "Daily Screen Time (hours)",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    study_level = st.selectbox(
        "Study Level",
        [
            "Poor",
            "Low",
            "Medium",
            "Good",
            "Very Good"
        ]
    )

st.divider()


# FEATURE ENGINEERING


def create_features():

    completion_ratio = (
        video_completion_pct / 100
    )

    quiz_efficiency = (
        quiz_score_pct /
        max(quiz_attempts, 1)
    )

    engagement_score = (
        0.35 * video_completion_pct
        + 0.20 * quiz_score_pct
        + 8 * sessions_per_week
        + 5 * questions_asked
        + 4 * (notes_taken == "Yes")
        - 2 * login_delay_min
        - 1.5 * pause_count
    )

    engagement_score = np.clip(
        engagement_score,
        0,
        100
    )

    data = pd.DataFrame({
        "study_level": [study_level],
        "video_completion_pct": [video_completion_pct],
        "lecture_duration_min": [lecture_duration_min],
        "pause_count": [pause_count],
        "rewatch_count": [ rewatch_count],
        "quiz_attempts": [quiz_attempts],
        "quiz_score_pct": [quiz_score_pct],
        "notes_taken": [notes_taken],
        "questions_asked": [ questions_asked],
        "login_delay_min": [login_delay_min],
        "sessions_per_week": [ sessions_per_week ],
        "screen_time_hours": [ screen_time_hours],
        "completion_ratio": [ completion_ratio ],
        "quiz_efficiency": [ quiz_efficiency ],
        "engagement_score": [ engagement_score]
    })

    return data


if st.button(
    "🔮 Predict Student Performance",
    type="primary",
    use_container_width=True
):
    input_data = create_features()


 # Classification
    

    try:
        classification_prediction = (
            classification_model.predict(
                input_data
            )[0]
        )

    except Exception as e:
        st.error(
            f"Classification prediction failed: {e}"
        )

        st.stop()


 # REGRESSION
    
    try:

        regression_prediction = (
            regression_model.predict(
                input_data
            )[0]
        )

    except Exception as e:

        st.error(
            f"Regression prediction failed: {e}"
        )

        st.stop()


    # Keep score between 0 and 100
    regression_prediction = np.clip(
        regression_prediction,
        0,
        100
    )

    st.divider()

    st.header("📈 Prediction Results")


    result_col1, result_col2 = st.columns(2)


        # CLASSIFICATION RESULT
    
    with result_col1:

        st.subheader("🎯 Performance Classification")

        st.metric(
            "Predicted Performance",
            str(classification_prediction)
        )

        if classification_prediction == "Very Good":

            st.success(
                "The model predicts a Very Good performance level."
            )

        elif classification_prediction == "Good":

            st.success(
                "The model predicts a Good performance level."
            )

        elif classification_prediction == "Medium":

            st.info(
                "The model predicts a Medium performance level."
            )

        elif classification_prediction == "Low":

            st.warning(
                "The model predicts a Low performance level."
            )

        elif classification_prediction == "Poor":

            st.error(
                "The model predicts a Poor performance level."
            )


        # REGRESSION RESULT
    
    with result_col2:

        st.subheader("🧮 Next Quiz Prediction")

        st.metric(
            "Expected Next Quiz Score",
            f"{regression_prediction:.2f}%"
        )

        st.progress(
            int(regression_prediction)
        )

    
    # MODEL INFORMATION

    st.divider()

    st.caption(
        "Classification predicts the student's performance "
        "category. Regression predicts the expected next quiz "
        "score."
    )

