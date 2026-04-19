import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

# ---------------------------
# 1. Load model
# ---------------------------

MODEL_NAME = "distilbert-base-uncased"   # change if needed
MODEL_PATH = Path(__file__).resolve().parent / "sentiment_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found at: {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---------------------------
# 2. Load tokenizer
# ---------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------------
# 3. Prediction function
# ---------------------------

def predict(text_list):
    inputs = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    labels_map = {0: "Negative", 1: "Positive"}

    results = []
    for text, pred, prob in zip(text_list, preds, probs):
        results.append({
            "text": text,
            "prediction": labels_map[pred.item()],
            "confidence": float(prob[pred.item()])
        })

    return results


def print_sample_predictions(results, title="===== SAMPLE PREDICTIONS =====", sep_width=70):
    """Print predictions in a readable block (Text / Prediction / Confidence)."""
    print(title)
    print()
    sep = "-" * sep_width
    for item in results:
        print(f"Text       : {item['text']}")
        print(f"Prediction : {item['prediction']}")
        print(f"Confidence : {item['confidence']:.4f}")
        print(sep)


# ---------------------------
# 4. Test
# ---------------------------

if __name__ == "__main__":
    samples = [
        "I love this product! It works perfectly.",
        "This is the worst experience I have ever had.",
        "Absolutely fantastic service, highly recommend!",
        "I hate this so much, total waste of money.",
        "Best purchase ever, I'm so happy!",
        "Not bad, could be better.",
        "The movie was okay, not great but not terrible.",
        "I expected more, but it's fine.",
        "Yeah right, this product is just amazing... (not!)",
        "Oh great, another delay. Fantastic service!",
        "Loved the design, but performance is horrible.",
        "Hate the interface but love the features.",
        "I wouldn't say it's great, but it's not terrible either.",
    ]

    outputs = predict(samples)
    print_sample_predictions(outputs)