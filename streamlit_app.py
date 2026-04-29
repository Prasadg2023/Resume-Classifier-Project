#Streamlit app for interactive resume category classification.

#This module loads cleaned resume text, trains a text classification model,
#shows model evaluation metrics, and provides a live prediction interface.


import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(text: str) -> str:
   #Clean raw resume text for consistent training and prediction.
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)  # remove email addresses
    text = re.sub(r"http\S+|www\.[^\s]+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z0-9+.#\s]", " ", text)  # keep letters, digits and common tech symbols
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text


def load_data(csv_path: str) -> pd.DataFrame:
    #Load cleaned resume data from CSV and make sure it is ready for training.
    df = pd.read_csv(csv_path)
    if 'cleaned' not in df.columns:
        # If the cleaned column is missing, generate it from raw resume text
        df['cleaned'] = df['Resume_str'].astype(str).apply(clean_text)
    df = df.dropna(subset=['cleaned', 'Category'])
    df = df[df['cleaned'].str.strip() != '']
    return df.reset_index(drop=True)


def build_model() -> Pipeline:
    # Build a simple text classification pipeline with TF-IDF and LinearSVC.
    return Pipeline([
        (
            'tfidf',
            TfidfVectorizer(
                stop_words='english',
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
            ),
        ),
        (
            'clf',
            LinearSVC(C=1.5, class_weight='balanced', max_iter=10000),
        ),
    ])


def train_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Split data into train/test sets and fit the model on training data
    X = df['cleaned']
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    # Evaluate the model and return accuracy, report, and confusion matrix
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
    }


def render_classification_report(report: dict) -> pd.DataFrame:
    #Convert the sklearn classification report into a DataFrame for display.
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df_report = df_report.rename_axis('class').reset_index()
    return df_report


def render_confusion_matrix(cm: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(cm, index=labels, columns=labels)


@st.cache_data
def load_cached_data(csv_path: str) -> pd.DataFrame:
    return load_data(csv_path)


@st.cache_resource
def load_cached_model(csv_path: str) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = load_cached_data(csv_path)
    model, X_train, X_test, y_train, y_test = train_model(df)
    return model, X_train, X_test, y_train, y_test


def main() -> None:
    st.set_page_config(page_title='Resume Classifier', layout='wide')

    st.title('📄 Resume Category Classifier')
    st.write(
        'This interactive Streamlit app uses cleaned resume text to train a resume category model, evaluate performance, and classify custom resume snippets.'
    )

    st.sidebar.header('App controls')
    csv_path = st.sidebar.text_input('Cleaned data CSV path', 'cleaned_resume_data.csv')
    show_raw = st.sidebar.checkbox('Show raw data sample', value=False)
    show_eval = st.sidebar.checkbox('Show evaluation metrics', value=True)
    show_distribution = st.sidebar.checkbox('Show category distribution', value=True)

    # Load the cleaned dataset and keep the UI responsive with caching
    df = load_cached_data(csv_path)
    st.sidebar.markdown(f'**Loaded rows:** {len(df)}')
    st.sidebar.markdown(f'**Unique labels:** {df["Category"].nunique()}')

    if show_raw:
        st.subheader('Raw cleaned dataset sample')
        st.dataframe(df.head(10))

    if show_distribution:
        st.subheader('Category distribution')
        counts = df['Category'].value_counts().rename_axis('Category').reset_index(name='Count')
        st.bar_chart(counts.set_index('Category'))

    st.subheader('Train the model')
    with st.spinner('Training model...'):
        model, X_train, X_test, y_train, y_test = load_cached_model(csv_path)

    if show_eval:
        eval_results = evaluate_model(model, X_test, y_test)
        st.metric('Accuracy', f'{eval_results["accuracy"]:.4f}')

        st.markdown('### Classification report')
        report_df = render_classification_report(eval_results['report'])
        st.dataframe(report_df.style.format({
            'precision': '{:.2f}',
            'recall': '{:.2f}',
            'f1-score': '{:.2f}',
            'support': '{:.0f}',
        }))

        st.markdown('### Confusion matrix')
        cm_df = render_confusion_matrix(eval_results['confusion_matrix'], sorted(df['Category'].unique()))
        st.dataframe(cm_df)

    st.subheader('Predict a custom resume snippet')
    sample_text = st.text_area(
        'Paste a resume paragraph or job skills description here',
        value='python machine learning pandas data analysis'
    )

    if st.button('Predict category'):
        sample_text_clean = clean_text(sample_text)
        if not sample_text_clean:
            st.warning('Enter some text to classify.')
        else:
            prediction = model.predict([sample_text_clean])[0]
            st.success(f'Predicted category: **{prediction}**')
            st.markdown('#### Cleaned input text')
            st.write(sample_text_clean)

    st.markdown('---')
    st.markdown(
        'Built with Streamlit, scikit-learn, and pandas. The model is trained from the cleaned resume dataset, then evaluated on a holdout split for interactive feedback.'
    )


if __name__ == '__main__':
    main()
