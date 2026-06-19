# Fake News Classifier

A machine learning web app that classifies news articles as **Real** or **Fake** using TF-IDF vectorization and Logistic Regression, deployed with Streamlit.


## Project Structure

```
fakenewsdetect/
├── assets/
│   └── demo.gif         # demo recording shown in README
├── app.ipynb           # Jupyter notebook: data loading, cleaning, training, evaluation
├── app.py              # Streamlit web app for inference
├── Fake.csv            # Fake news dataset
├── True.csv             # Real news dataset
├── lr_model.jb          # Trained Logistic Regression model (joblib)
├── vectorizer.jb         # Fitted TF-IDF vectorizer (joblib)
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```


## How It Works

1. **Data Loading** — `Fake.csv` and `True.csv` are loaded and labeled (`0` = fake, `1` = real).
2. **Preprocessing** — Title, subject, and date columns are dropped; article text is lowercased and cleaned (URLs, HTML tags, punctuation, digits, and extra whitespace removed) using regex.
3. **Feature Extraction** — Cleaned text is converted into numerical features using `TfidfVectorizer`.
4. **Model Training** — A `LogisticRegression` classifier is trained on a 75/25 train-test split.
5. **Evaluation** — Performance is measured using `classification_report` (precision, recall, F1-score, accuracy).
6. **Persistence** — The trained model and vectorizer are saved with `joblib` as `lr_model.jb` and `vectorizer.jb`.
7. **Inference App** — `app.py` loads the saved model/vectorizer and serves a Streamlit UI where users paste in article text and get an instant Real/Fake prediction.


## Tech Stack

- Python 3.10.0
- pandas
- scikit-learn (TfidfVectorizer, LogisticRegression)
- joblib
- Streamlit


## Dataset

This project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle, consisting of two CSV files: `Fake.csv` and `True.csv`.


## Installation & Local Setup

# Clone the repository
git clone https://github.com/rajalaxmibiswal/fakenewsdetect.git
cd fakenewsdetect

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


## Usage

### Retrain the model (optional)
Open and run `app.ipynb` from top to bottom. This regenerates `lr_model.jb` and `vectorizer.jb`.

### Run the web app
streamlit run app.py
Then open the local URL shown in your terminal (typically `http://localhost:8501`).


## Model Performance

| Metric | Score |
|---|---|
| Accuracy |99% (Fake), 98% (Real)|
| Precision | 99% |
| Recall |99% (Fake), 99% (Real)|
| F1-Score | 99% (Fake), 99% (Real) |

## Future Improvements

- Add cross-validation and hyperparameter tuning (`GridSearchCV`)
- Compare additional models (Naive Bayes, Passive Aggressive Classifier, Random Forest)
- Add confidence/probability score alongside the prediction
- Deploy publicly via Streamlit Community Cloud
- Add unit tests for the cleaning and prediction pipeline

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- Dataset by Clément Bisaillon on Kaggle
- Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/)
