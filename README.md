# Clean the Messy Text — NLP Basics

PSEB AI Bootcamp lab: understand, check, and implement the four steps of text cleaning
(normalize → tokenize → remove stopwords → stem/lemmatize).

## Files

| File | What it is |
|------|------------|
| `NLP_Clean_the_Messy_Text.ipynb` | The submitted Colab notebook, with all outputs saved |
| `app.py` | Streamlit web app that shows the same results interactively |
| `requirements.txt` | Python packages needed by the app |

## Run the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

> The first run downloads the small NLTK datasets (`punkt`, `punkt_tab`, `stopwords`,
> `wordnet`, and the POS taggers) automatically.

## Run the notebook

Upload `NLP_Clean_the_Messy_Text.ipynb` to [Google Colab](https://colab.research.google.com/)
and choose **Runtime → Run all**. Outputs are already embedded, so it can also be viewed
directly on GitHub without running anything.

## Notes on the implementation

- Negation words (`not`, `no`, `nor`, `never`, `none`) are protected during stopword removal,
  so "not good" does not become "good".
- Lemmatization uses POS tags (`pos_tag` → WordNet POS), which is what turns `went` into `go`.
- `us`, `u` and `i` are left as they are, because WordNet would otherwise turn `us` into `u`
  (and lowercasing already merges the country `US` with the pronoun `us`).
