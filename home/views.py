from django.shortcuts import render
from django.conf import settings
from huggingface_hub import InferenceClient

MODEL_NAME = "google/flan-t5-large"

# create client safely
client = InferenceClient(token=settings.HF_TOKEN)

def analyze_with_ai(question, answer):
    prompt = f"""
You are StepFix.

Analyze the student's work.

Return ONLY in this format:

[MISTAKE]
...

[WHY]
...

[CONCEPT]
...

[CORRECT]
...

[PRACTICE]
...

Question:
{question}

Student Solution:
{answer}
"""

    try:
        # ✅ CORRECT HF METHOD (NOT chat.completions)
        response = client.text_generation(
            model=MODEL_NAME,
            prompt=prompt,
            max_new_tokens=1000,
        )

        return response.strip()

    except Exception as e:
        return f"[MISTAKE]\nAI Error: {str(e)}"


def parse_response(text):
    result = {
        "mistake": "",
        "why": "",
        "concept": "",
        "correct": "",
        "practice": ""
    }

    current = None

    for line in text.splitlines():
        line = line.strip()

        if line == "[MISTAKE]":
            current = "mistake"
            continue
        elif line == "[WHY]":
            current = "why"
            continue
        elif line == "[CONCEPT]":
            current = "concept"
            continue
        elif line == "[CORRECT]":
            current = "correct"
            continue
        elif line == "[PRACTICE]":
            current = "practice"
            continue

        if current:
            result[current] += line + "\n"

    return result


def index(request):
    result = None

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        answer = request.POST.get("answer", "").strip()

        if question and answer:
            ai_response = analyze_with_ai(question, answer)
            result = parse_response(ai_response)

    return render(request, "home/index.html", {"result": result})