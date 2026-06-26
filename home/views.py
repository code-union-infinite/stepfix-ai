from django.shortcuts import render
from django.conf import settings
from huggingface_hub import InferenceClient

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(
    api_key=settings.HF_TOKEN
)


def analyze_with_ai(question, answer):
    prompt = f"""
You are StepFix.

Analyze the student's work.

Return ONLY in this format.

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
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        return response.choices[0].message.content.strip()

    except TimeoutError:
        return """
[MISTAKE]
The AI took too long to respond.

[WHY]
Our servers may be busy.

[CONCEPT]
Please try again in a few seconds.

[CORRECT]

[PRACTICE]
"""

    except Exception:
        return """
[MISTAKE]
We couldn't reach the AI.

[WHY]
The AI service is temporarily unavailable.

[CONCEPT]
Please refresh the page and try again.

[CORRECT]

[PRACTICE]
"""

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

        if line == "[WHY]":
            current = "why"
            continue

        if line == "[CONCEPT]":
            current = "concept"
            continue

        if line == "[CORRECT]":
            current = "correct"
            continue

        if line == "[PRACTICE]":
            current = "practice"
            continue

        if current:
            result[current] += line + "\n"

    for key in result:
        result[key] = result[key].strip()

    return result


def index(request):
    result = None

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        answer = request.POST.get("answer", "").strip()

        if question and answer:
            ai_response = analyze_with_ai(question, answer)
            result = parse_response(ai_response)
        else:
            result = {
                "mistake": "Please enter both the question and your solution.",
                "why": "",
                "concept": "",
                "correct": "",
                "practice": ""
            }

    return render(
        request,
        "home/index.html",
        {
            "result": result
        }
    )