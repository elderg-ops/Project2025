from flask import Flask, render_template, request
import google.generativeai as genai
import os
import re

# 🔐 Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyD2rwr19WRjGwngGOdfk2EWfhwBaGhc84U"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

# 🚀 Flask App Setup
app = Flask(__name__)

# 🔍 Regex filter for Jinja2
@app.template_filter('regex_search')
def regex_search(s, pattern, group=1):
    match = re.search(pattern, s, re.DOTALL)
    return match.group(group).strip() if match else 'No data found.'

# 🏠 Home Route
@app.route('/')
def index():
    return render_template("index.html")

# 🧾 Signup Page
@app.route('/signup')
def signup():
    return render_template("signup.html")

# 🔑 Login Page
@app.route('/login')
def login():
    return render_template("login.html")

# 🥗 Meal Plan Route
@app.route("/mealplan", methods=["GET", "POST"])
def mealplan():
    diet_plan = None

    if request.method == "POST":
        age = request.form.get("age")
        gender = request.form.get("gender")
        weight = request.form.get("weight")
        height = request.form.get("height")
        activity_level = request.form.get("activity_level")
        goal = request.form.get("goal")
        diet_type = request.form.get("diet_type")

        prompt = f"""
        Create a personalized diet plan with only 3 sections: Breakfast, Lunch, and Dinner.

        🔹 Format it exactly like this (strictly):
        Breakfast:
        - Item 1 - X kcal
        - Item 2 - Y kcal

        Lunch:
        - Item 1 - X kcal
        - Item 2 - Y kcal

        Dinner:
        - Item 1 - X kcal
        - Item 2 - Y kcal

        🔸 User Info:
        - Age: {age}
        - Gender: {gender}
        - Weight: {weight} kg
        - Height: {height} cm
        - Activity Level: {activity_level}
        - Goal: {goal}
        - Diet Type: {diet_type}
        """

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Convert response to HTML table
            sections = re.split(r'\n(?=[A-Z][a-z]+:)', text)
            table_html = '<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">'
            table_html += '<tr><th>Meal</th><th>Food Item</th><th>Calories</th></tr>'

            for section in sections:
                lines = section.strip().split('\n')
                if len(lines) == 0:
                    continue
                meal = lines[0].replace(':', '')
                for line in lines[1:]:
                    match = re.match(r'- (.+?) - (\d+)\s*kcal', line)
                    if match:
                        item = match.group(1).strip()
                        kcal = match.group(2).strip()
                        table_html += f'<tr><td>{meal}</td><td>{item}</td><td>{kcal}</td></tr>'

            table_html += '</table>'
            diet_plan = table_html

        except Exception as e:
            diet_plan = f"Error generating diet plan: {str(e)}"

    return render_template("mealplan.html", diet_plan=diet_plan)

# 🏁 Start app
if __name__ == "__main__":
    app.run(debug=True)
