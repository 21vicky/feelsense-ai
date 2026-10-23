# FeelSense AI 🤖

An AI-powered web application that analyzes the sentiment and emotions expressed in English text.

## Features

* Positive and negative sentiment classification
* Emotion detection using a pretrained NLP model
* Interactive emotion distribution chart
* Emotion score breakdown
* Recent analysis history
* Dark-themed Gradio interface

## Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Gradio
* Plotly
* Pandas

## Models

**Sentiment Analysis:** DistilBERT fine-tuned on SST-2.

**Emotion Classification:** DistilRoBERTa fine-tuned for English emotion classification.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python app.py
```

## Limitations

The predictions depend on the pretrained models and may be inaccurate for sarcasm, ambiguous wording, or context-dependent expressions. Emotion scores should not be interpreted as psychological assessments.

Analysis history is stored in application memory and is not permanently saved.

## Future Improvements

* Persistent analysis history
* Neutral sentiment classification
* User accounts and saved analyses
* Additional languages
* More advanced analytics
