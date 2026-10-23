
import gradio as gr
import plotly.express as px
import pandas as pd

from transformers import pipeline

print("Loading FeelSense AI models...")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

history = []


def history_dataframe():
    columns = ["Text", "Sentiment", "Emotion", "Confidence"]

    if not history:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(history[::-1], columns=columns)


def analyze_text(text):
    if not text or not text.strip():
        return (
            "Please enter some text.",
            "—",
            "—",
            None,
            pd.DataFrame(columns=["Emotion", "Score"]),
            history_dataframe()
        )

    text = text.strip()

    sentiment = sentiment_model(text)[0]
    emotions = emotion_model(text)[0]

    emotions.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    dominant = emotions[0]

    history.append({
        "Text": text,
        "Sentiment": sentiment["label"],
        "Emotion": dominant["label"].title(),
        "Confidence": round(sentiment["score"] * 100, 2)
    })

    emotion_df = pd.DataFrame([
        {
            "Emotion": item["label"].title(),
            "Score": round(item["score"], 4)
        }
        for item in emotions
    ])

    fig = px.bar(
        emotion_df,
        x="Emotion",
        y="Score",
        title="Emotion Distribution",
        template="plotly_dark",
        color="Score",
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font_color="#F9FAFB",
        yaxis_range=[0, 1],
        showlegend=False,
        coloraxis_showscale=False
    )

    summary = f"""
## 🧠 Analysis Results

**Your text:** {text}

**Predicted sentiment:** {sentiment['label']}

**Dominant emotion:** {dominant['label'].title()}

*Predictions are model estimates and may be incorrect.*
"""

    return (
        summary,
        f"{sentiment['label']} · {sentiment['score']:.2%}",
        f"{dominant['label'].title()} · {dominant['score']:.2%}",
        fig,
        emotion_df,
        history_dataframe()
    )


def clear_history():
    history.clear()

    return history_dataframe(), "Analysis history cleared."


css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
    background: linear-gradient(135deg, #080B18, #15112A);
}
#hero {
    text-align: center;
    padding: 30px 20px;
    margin-bottom: 20px;
    border-radius: 22px;
    background: rgba(124, 58, 237, 0.18);
    border: 1px solid rgba(255,255,255,0.12);
}
#hero h1 {
    font-size: 42px;
    color: #C4B5FD;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css,
    title="FeelSense AI"
) as demo:

    gr.HTML("""
    <div id="hero">
        <h1>✦ FeelSense AI</h1>
        <p>AI-powered sentiment and emotion intelligence</p>
    </div>
    """)

    with gr.Tab("✨ Analyze"):

        text_input = gr.Textbox(
            label="Enter Your Text",
            placeholder="Example: I finally got my dream job!",
            lines=5
        )

        with gr.Row():
            analyze_button = gr.Button(
                "✨ Analyze Text",
                variant="primary"
            )

            clear_input_button = gr.ClearButton(
                components=[text_input],
                value="🧹 Clear Text"
            )

        summary_output = gr.Markdown()

        with gr.Row():
            sentiment_output = gr.Textbox(
                label="📊 Sentiment Prediction",
                interactive=False
            )

            emotion_output = gr.Textbox(
                label="🎭 Dominant Emotion",
                interactive=False
            )

        chart_output = gr.Plot(label="Emotion Distribution")

        emotion_table = gr.Dataframe(
            label="Emotion Score Breakdown",
            interactive=False
        )

        gr.Examples(
            examples=[
                ["I am so happy that I passed my exam!"],
                ["I feel sad because I miss my friends."],
                ["I am angry about this unfair decision."],
                ["I am surprised by the wonderful news."],
                ["I love spending time with my family."],
                ["I am afraid of failing my interview."]
            ],
            inputs=text_input
        )

    with gr.Tab("📜 Analysis History"):

        history_table = gr.Dataframe(
            value=history_dataframe(),
            label="Recent Analyses",
            interactive=False
        )

        clear_history_button = gr.Button("🗑️ Clear History")
        history_message = gr.Markdown()

    analyze_button.click(
        fn=analyze_text,
        inputs=[text_input],
        outputs=[
            summary_output,
            sentiment_output,
            emotion_output,
            chart_output,
            emotion_table,
            history_table
        ]
    )

    clear_history_button.click(
        fn=clear_history,
        inputs=[],
        outputs=[history_table, history_message]
    )

demo.launch()
