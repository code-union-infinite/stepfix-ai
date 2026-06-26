from django.shortcuts import render
from django.conf import settings
from huggingface_hub import InferenceClient

MODEL_NAME = "google/flan-t5-large"

# create client safely
client = InferenceClient(token=settings.HF_TOKEN)
def analyze_with_ai(question, answer):
    import requests

    try:
        r = requests.get("https://huggingface.co", timeout=10)

        return f"""
[MISTAKE]
HF Website Status: {r.status_code}

[WHY]
Test

[CONCEPT]
Test

[CORRECT]
Test

[PRACTICE]
Test
"""

    except Exception as e:
        return f"""
[MISTAKE]
{repr(e)}

[WHY]

[CONCEPT]

[CORRECT]

[PRACTICE]
"""

def index(request):
    result = None

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        answer = request.POST.get("answer", "").strip()

        if question and answer:
            ai_response = analyze_with_ai(question, answer)
            result = result = {
    "mistake": ai_response,
    "why": "",
    "concept": "",
    "correct": "",
    "practice": ""
}

    return render(request, "home/index.html", {"result": result})