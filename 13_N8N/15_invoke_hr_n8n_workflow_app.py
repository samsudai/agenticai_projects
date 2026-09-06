# pip install flask requests

from flask import Flask, request, render_template_string
import requests

app = Flask(__name__)

# Replace with your actual n8n webhook URL
N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/hr-policy-search"

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>HR Policy Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 40px auto;
            padding: 20px;
        }
        textarea, input[type="text"] {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            margin-top: 10px;
        }
        button {
            margin-top: 12px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .box {
            margin-top: 25px;
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background: #f8f8f8;
        }
        .label {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h2>HR Policy Chatbot</h2>
    <form method="POST">
        <label for="question"><b>Ask a question:</b></label>
        <input type="text" id="question" name="question" placeholder="Enter your question..." required value="{{ question or '' }}">
        <button type="submit">Ask</button>
    </form>

    {% if answer %}
    <div class="box">
        <div><span class="label">Your Question:</span> {{ question }}</div>
        <br>
        <div><span class="label">Answer:</span> {{ answer }}</div>

        {% if matched_question %}
        <br>
        <div><span class="label">Matched Question:</span> {{ matched_question }}</div>
        {% endif %}

        {% if category %}
        <div><span class="label">Category:</span> {{ category }}</div>
        {% endif %}

        {% if score is not none %}
        <div><span class="label">Score:</span> {{ score }}</div>
        {% endif %}
    </div>
    {% endif %}

    {% if error %}
    <div class="box" style="background:#ffecec; border-color:#ffb3b3;">
        <div><span class="label">Error:</span> {{ error }}</div>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    matched_question = None
    score = None
    category = None
    error = None
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if question:
            try:
                response = requests.post(
                    N8N_WEBHOOK_URL,
                    json={"question": question},
                    timeout=30
                )
                response.raise_for_status()

                data = response.json()

                answer = data.get("answer", "No answer returned.")
                matched_question = data.get("matched_question")
                score = data.get("score")
                category = data.get("category")

            except requests.exceptions.RequestException as e:
                error = f"Failed to connect to n8n: {str(e)}"
            except ValueError:
                error = "n8n did not return valid JSON."

    return render_template_string(
        HTML_PAGE,
        question=question,
        answer=answer,
        matched_question=matched_question,
        score=score,
        category=category,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)