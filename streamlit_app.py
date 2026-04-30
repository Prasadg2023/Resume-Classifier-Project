# Simple Streamlit UI for resume category classification.

# This app loads a cleaned resume dataset, trains a text classifier,
# shows basic evaluation, and lets users classify custom resume snippets.

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DEFAULT_CSV = Path('cleaned_resume_data.csv')


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\.[^\s]+", " ", text)
    text = re.sub(r"[^a-z0-9+.#\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(csv_source) -> pd.DataFrame:
    df = pd.read_csv(csv_source)
    if 'cleaned' not in df.columns:
        df['cleaned'] = df['Resume_str'].astype(str).apply(clean_text)
    df = df.dropna(subset=['cleaned', 'Category'])
    df = df[df['cleaned'].str.strip() != '']
    return df.reset_index(drop=True)


def build_model() -> Pipeline:
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
        ('clf', LinearSVC(C=1.5, class_weight='balanced', max_iter=10000)),
    ])


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df['cleaned']
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    model = build_model()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


def render_classification_report(report: dict) -> pd.DataFrame:
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    return df_report.rename_axis('class').reset_index()


def render_confusion_matrix(cm: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(cm, index=labels, columns=labels)


@st.cache_data
def load_cached_data(csv_path: str) -> pd.DataFrame:
    return load_data(csv_path)


def find_dataset() -> Path | None:
    return DEFAULT_CSV if DEFAULT_CSV.exists() else None


def get_example_snippets(df: pd.DataFrame) -> list[str]:
    examples = [
        'python machine learning pandas data analysis',
        'financial modeling excel budgeting forecasting',
        'project management agile scrum stakeholder communication',
    ]
    grouped = df.groupby('Category')['cleaned'].first().dropna()
    examples.extend(grouped.sample(min(len(grouped), 3), random_state=1).tolist())
    return examples


def main() -> None:
    st.set_page_config(page_title='Resume Classifier', layout='wide')

    st.title('📄 Resume Category Classifier')
    st.write(
        'A Streamlit interface for training a resume category model from cleaned resume text and classifying custom snippets.'
    )

    st.sidebar.header('Dataset & training settings')
    use_uploaded = st.sidebar.checkbox('Upload a dataset CSV', value=False)
    uploaded_file = None
    if use_uploaded:
        uploaded_file = st.sidebar.file_uploader('Choose a cleaned resume CSV', type='csv')

    test_size = st.sidebar.slider(
        'Test set size', min_value=0.1, max_value=0.4, value=0.2, step=0.05
    )
    random_state = st.sidebar.number_input('Random seed', min_value=0, max_value=9999, value=42)
    show_raw = st.sidebar.checkbox('Show raw data sample', value=False)
    show_eval = st.sidebar.checkbox('Show evaluation metrics', value=True)
    show_distribution = st.sidebar.checkbox('Show category distribution', value=True)
    show_preview = st.sidebar.checkbox('Show dataset preview', value=True)
    st.sidebar.caption('Upload or use cleaned_resume_data.csv from this folder.')

    if uploaded_file is not None:
        data_source = uploaded_file
        df = load_data(data_source)
    elif find_dataset() is not None:
        data_source = DEFAULT_CSV
        df = load_cached_data(str(data_source))
    else:
        st.error('No dataset found. Upload cleaned_resume_data.csv or place it in this folder.')
        return

    if df.empty:
        st.error('The dataset is empty after cleaning. Verify the file contains `Resume_str` and `Category` values.')
        return

    df['cleaned'] = df['cleaned'].astype(str)
    df['Category'] = df['Category'].astype(str)

    total_rows = len(df)
    unique_labels = df['Category'].nunique()
    category_counts = df['Category'].value_counts().rename_axis('Category').reset_index(name='Count')

    sidebar_stats, sidebar_stats2 = st.columns([1, 1])
    sidebar_stats.metric('Loaded rows', total_rows)
    sidebar_stats.metric('Unique categories', unique_labels)
    sidebar_stats2.metric('Train rows', int(total_rows * (1 - test_size)))
    sidebar_stats2.metric('Test rows', int(total_rows * test_size))

    if show_preview:
        with st.expander('Dataset preview and sample categories', expanded=True):
            st.write('First 10 rows from the cleaned dataset')
            st.dataframe(df.head(10))
            st.write('Top categories in the dataset')
            st.dataframe(category_counts)

    if show_distribution:
        st.subheader('Category distribution')
        st.bar_chart(category_counts.set_index('Category'))

    if show_raw:
        with st.expander('Raw resume text examples'):
            st.dataframe(df[['Resume_str', 'cleaned', 'Category']].head(10))

    st.subheader('Train the model')
    with st.spinner('Training model...'):
        model, X_train, X_test, y_train, y_test = train_model(df, test_size=test_size, random_state=random_state)

    if show_eval:
        eval_results = evaluate_model(model, X_test, y_test)
        col1, col2, col3 = st.columns(3)
        col1.metric('Accuracy', f"{eval_results['accuracy']:.4f}")
        col2.metric('Classes', unique_labels)
        col3.metric('Test sample size', len(X_test))

        with st.expander('Detailed evaluation metrics', expanded=True):
            st.markdown('#### Classification report')
            report_df = render_classification_report(eval_results['report'])
            st.dataframe(report_df)

            st.markdown('#### Confusion matrix')
            cm_df = render_confusion_matrix(eval_results['confusion_matrix'], sorted(df['Category'].unique()))
            st.dataframe(cm_df)

    st.markdown('---')
    st.subheader('Predict a custom resume snippet')

    default_examples = get_example_snippets(df)
    sample_options = ['Custom example'] + default_examples
    sample_choice = st.selectbox('Choose a sample snippet', sample_options)
    if sample_choice == 'Custom example':
        sample_text = st.text_area(
            'Paste a resume paragraph or job skills description here',
            value='python machine learning pandas data analysis',
            height=180,
        )
    else:
        sample_text = st.text_area(
            'Paste a resume paragraph or job skills description here',
            value=sample_choice,
            height=180,
        )

    if st.button('Predict category'):
        sample_text_clean = clean_text(sample_text)
        if not sample_text_clean:
            st.warning('Enter some text to classify.')
        else:
            prediction = model.predict([sample_text_clean])[0]
            st.success(f'Predicted category: **{prediction}**')
            with st.expander('Cleaned input text', expanded=True):
                st.write(sample_text_clean)

    st.markdown('---')
    st.markdown('Built with Streamlit, scikit-learn, and pandas. Run with `streamlit run streamlit_app.py`.')


if __name__ == '__main__':
    main()
