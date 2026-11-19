import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Clean the Messy Text", page_icon="🧹", layout="wide")


# ---------------------------------------------------------------------------
# NLTK resources (downloaded once and cached by Streamlit)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    for package in (
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ):
        nltk.download(package, quiet=True)
    return {
        "stopwords": set(stopwords.words("english")),
        "stemmer": PorterStemmer(),
        "lemmatizer": WordNetLemmatizer(),
    }


RES = load_resources()
STOPWORDS = RES["stopwords"]
STEMMER = RES["stemmer"]
LEMMATIZER = RES["lemmatizer"]


# ---------------------------------------------------------------------------
# Part 1 — Normalize
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not",
    "i'm": "i am", "he's": "he is", "she's": "she is", "it's": "it is",
    "that's": "that is", "what's": "what is", "there's": "there is",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
    "mustn't": "must not", "let's": "let us", "you're": "you are",
    "we're": "we are", "they're": "they are",
}

EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF\U00002600-\U000026FF\uFE0F"
    "]+", flags=re.UNICODE
)
TEXT_SMILEY_RE = re.compile(r"(?::|;|=|8|x)(?:-|\^)?(?:\)|\(|d|D|p|P|/|\\|\||\])")


def normalize(text):
    text = text.lower().replace("\u2019", "'")
    for short, long in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(short) + r"\b", long, text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = EMOJI_RE.sub(" ", text)
    text = TEXT_SMILEY_RE.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Part 2 — Tokenize and remove stopwords
# ---------------------------------------------------------------------------
CUSTOM_STOPWORDS = {"rt", "omg"}
NEGATIONS = {"not", "no", "nor", "never", "none"}


def tokenize_and_filter(text, keep_negations=True):
    drop = set(STOPWORDS) | CUSTOM_STOPWORDS
    if keep_negations:
        drop = drop - NEGATIONS
    return [t for t in word_tokenize(text) if t not in drop and t.strip()]


# ---------------------------------------------------------------------------
# Part 3 — Lemmatization helpers
# ---------------------------------------------------------------------------
def wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


KEEP_AS_IS = {"us", "u", "i"}


def lemmatize_tokens(tokens):
    return [
        t if t in KEEP_AS_IS else LEMMATIZER.lemmatize(t, wordnet_pos(tag))
        for t, tag in pos_tag(tokens)
    ]


def full_pipeline(text, keep_negations=True):
    return lemmatize_tokens(tokenize_and_filter(normalize(text), keep_negations))


# ---------------------------------------------------------------------------
# Lab data (same six sentences as the notebook)
# ---------------------------------------------------------------------------
MESSY = [
    "OMG!! I can't believe this bootcamp is sooo good 😂😂 #AI",
    "Check out https://example.com for NLP notes!!! RT @friend",
    "We live in the US, and the Us team doesn't like us.",
    "He's going, she went, they goes... and I have gone 😅",
    "I did NOT like this movie. Not good at all!!",
    "Price was 9.99$ and rating 4.5/5 :) :p",
]

normalized = [normalize(s) for s in MESSY]
filtered = [tokenize_and_filter(s) for s in normalized]
clean_sentences = [full_pipeline(s) for s in MESSY]
all_tokens = [t for tokens in clean_sentences for t in tokens]

counts = pd.DataFrame({
    "sentence": [f"S{i + 1}" for i in range(len(MESSY))],
    "text": MESSY,
    "words before": [len(s.split()) for s in MESSY],
    "words after": [len(t) for t in clean_sentences],
})
counts["removed"] = counts["words before"] - counts["words after"]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🧹 Clean the Messy Text — NLP Basics")
st.caption("PSEB AI Bootcamp · interactive version of the lab notebook")

tabs = st.tabs([
    "Understand",
    "1. Normalize",
    "2. Stopwords",
    "3. Stem vs lemma",
    "4. Pipeline + chart",
    "Opinion",
])

with tabs[0]:
    st.header("The idea")
    st.write("Computers cannot work with messy text, so we clean it first — the way you wash vegetables before cooking.")
    st.dataframe(pd.DataFrame({
        "Step": ["Normalize", "Tokenize", "Remove stopwords", "Stem / lemmatize"],
        "Simple words": ["Make the text neat and in one style", "Cut the text into pieces",
                         "Drop common filler words", "Bring word forms back to one base form"],
        "Before → After": ["Hello, World!! → hello world", "hello world → ['hello', 'world']",
                           "['this','is','a','good','movie'] → ['good','movie']",
                           "studies → studi (stem) / study (lemma)"],
    }), hide_index=True)

    st.subheader("Check your understanding — answers")
    st.dataframe(pd.DataFrame({
        "#": [1, 2, 3, 4, 5, 6],
        "Question": [
            "Why clean text before a computer sees it?",
            "Normalize 'I LOVE this Movie!!! :)'",
            "Tokens in 'good morning class'",
            "Stopwords in 'this is a very good lab'",
            "Which turns 'went' into 'go'?",
            "Risky for 'I did not like it'?",
        ],
        "Answer": [
            "So the model sees one neat form, not noise like capitals, punctuation and links.",
            "i love this movie",
            "3 — good, morning, class",
            "this, is, a, very",
            "Lemmatization with pos='v': it uses a dictionary that knows went is the past of go.",
            "Removing 'not' turns it into 'I like it' and flips the meaning.",
        ],
    }), hide_index=True)

    st.info("Two traps: lowercasing merges 'US' (country) with 'us' (people), and stopword removal can delete 'not'.")

with tabs[1]:
    st.header("Part 1 — Normalize")
    st.write("Lowercase, expand contractions, then remove URLs, @mentions, hashtags, emoji, text smileys and punctuation; finally collapse extra spaces.")
    st.dataframe(pd.DataFrame({"before": MESSY, "after": normalized}), hide_index=True)
    with st.expander("Show the normalize() code"):
        st.code(
            "def normalize(text):\n"
            "    text = text.lower().replace('\\u2019', \"'\")\n"
            "    for short, long in CONTRACTIONS.items():\n"
            "        text = re.sub(r'\\b' + re.escape(short) + r'\\b', long, text)\n"
            "    text = re.sub(r'https?://\\S+|www\\.\\S+', ' ', text)\n"
            "    text = re.sub(r'@\\w+', ' ', text)\n"
            "    text = re.sub(r'#\\w+', ' ', text)\n"
            "    text = EMOJI_RE.sub(' ', text)\n"
            "    text = TEXT_SMILEY_RE.sub(' ', text)\n"
            "    text = re.sub(r'[^\\w\\s]', ' ', text)\n"
            "    return re.sub(r'\\s+', ' ', text).strip()",
            language="python",
        )

with tabs[2]:
    st.header("Part 2 — Tokenize and remove stopwords")
    st.write("word_tokenize + NLTK English stopwords, plus two custom stopwords: **rt** (retweet marker) and **omg** (exclamation with no topic meaning).")
    st.dataframe(pd.DataFrame({"normalized": normalized, "tokens after stopwords": filtered}),
                 hide_index=True)
    st.subheader("The negation trap")
    st.write("Sentence 5 with `keep_negations=False`:")
    st.code(str(tokenize_and_filter(normalized[4], keep_negations=False)))
    st.write("Sentence 5 with `keep_negations=True` (what the final pipeline uses):")
    st.code(str(tokenize_and_filter(normalized[4], keep_negations=True)))
    st.warning("Without the guard, 'not good' becomes 'good' — a negative review turns positive.")

with tabs[3]:
    st.header("Part 3 — Stemming vs lemmatization")
    words = ["go", "goes", "going", "gone", "went", "studies", "better", "running"]
    stem_table = pd.DataFrame({
        "word": words,
        "stem": [STEMMER.stem(w) for w in words],
        "lemma": [LEMMATIZER.lemmatize(w) for w in words],
    })
    st.dataframe(stem_table, hide_index=True)
    pos_table = pd.DataFrame({
        "word": words,
        "lemma (no pos)": [LEMMATIZER.lemmatize(w) for w in words],
        "lemma (pos='v')": [LEMMATIZER.lemmatize(w, pos="v") for w in words],
    })
    st.dataframe(pos_table, hide_index=True)
    st.write("**Lemmatization looked more correct:** `studies → study` is a real word, while the stemmer gives `studi`. With `pos='v'` it also maps `goes / going / gone / went → go`. Without POS information it leaves `went` unchanged, which is why the pipeline uses POS tags.")

with tabs[4]:
    st.header("Part 4 — Full pipeline")
    st.write("normalize → tokenize → remove stopwords → POS-aware lemmatize. `us`, `u` and `i` are left alone because WordNet would change `us` into `u`.")
    for original, clean in zip(MESSY, clean_sentences):
        st.markdown(f"- **{original}**  \n  → `{clean}`")

    st.subheader("Word counts before and after cleaning")
    st.dataframe(counts, hide_index=True)

    chart_counts = counts.set_index("sentence")[["words before", "words after"]]
    st.bar_chart(chart_counts)

    st.subheader("10 most common words after cleaning")
    top10 = pd.Series(all_tokens).value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(top10.index, top10.values, color="#4C72B0")
    ax.set_xlabel("Word")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 10 words after cleaning")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Try it yourself")
    user_text = st.text_area("Type a messy sentence:", "I did NOT love this movie!!! 😡 #review")
    if user_text.strip():
        st.code(str(full_pipeline(user_text)))

    st.download_button(
        "Download cleaned results (CSV)",
        data=pd.DataFrame({"sentence": MESSY, "clean_tokens": [" ".join(t) for t in clean_sentences]}).to_csv(index=False),
        file_name="cleaned_results.csv",
        mime="text/csv",
    )

with tabs[5]:
    st.header("Opinion — Topic 1: Do stopwords always help?")
    st.write(
        "Removing stopwords does **not** always help, because a word can be common and still carry the whole meaning "
        "of a sentence. The clearest example is in this lab: sentence 5 is \"I did NOT like this movie. Not good at all!!\". "
        "When I remove NLTK stopwords the usual way, `not` disappears and the result is `like movie good` — a negative "
        "review has quietly become a positive one. That is why `tokenize_and_filter` keeps negation words: the cleaned "
        "version `not like movie not good` still says what the writer meant. In real life the same thing happens in a "
        "school feedback form: \"not good\" and \"good\" must never collapse into the same text. A sensible rule is to "
        "remove stopwords for tasks like topic modelling or counting words, but to keep words such as `not`, `no` and "
        "`never` for sentiment analysis, and to check a few examples by eye before trusting the output. Cleaning is a "
        "set of choices, not a recipe you can follow without thinking."
    )

st.sidebar.success("NLTK data loaded and cached ✅")
st.sidebar.markdown("Files: `NLP_Clean_the_Messy_Text.ipynb`, `app.py`, `requirements.txt`")
