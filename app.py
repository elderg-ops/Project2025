from flask import Flask, render_template, request
import google.generativeai as genai
import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# 🔐 Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyD2rwr19WRjGwngGOdfk2EWfhwBaGhc84U"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ Rename Gemini model to avoid clash with ML model
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# 🚀 Flask App Setup
app = Flask(__name__)
CORS(app)

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

# 🔧 Equipment Page
@app.route('/equipment')
def equipment():
    return render_template("equipment.html")

# ✅ NEW: Generate Meal Plan via Gemini AI
@app.route('/generateMealPlan', methods=['POST'])
def generateMealPlan():
    try:
        data = request.json
        calories = data['calories']

        prompt = f"Create a meal plan under {calories} calories including breakfast, lunch, and dinner. " \
                 f"Include Indian foods. Present it in a table with columns: Meal, Dish, Calories."

        response = gemini_model.generate_content(prompt)
        return jsonify({'meal_plan': response.text})
    except Exception as e:
        return jsonify({'error': f"Error generating diet plan: {str(e)}"}), 500

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
            response = gemini_model.generate_content(prompt)
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

# 🏋️ Workout Plan Route
@app.route('/workout')
def workout():
    return render_template("workout.html")

# 💪 Workout Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        age = int(data['age'])
        height = float(data['height'])
        weight = float(data['weight'])
        gender = encoders['gender'].transform([data['gender']])[0]
        goal = encoders['goal'].transform([data['goal']])[0]
        level = encoders['level'].transform([data['level']])[0]
        preferences = encoders['preferences'].transform([data['preferences']])[0]
        days = int(data.get('days', 7))

        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)

        # Predict body type
        features = np.array([[age, height, weight, gender, goal, level, preferences]])
        prediction = model.predict(features)[0]
        body_type = encoders['target'].inverse_transform([prediction])[0]

        # Define possible options
        duration_options = {
            "beginner": [20, 25],
            "intermediate": [30, 35],
            "advanced": [40, 45]
        }
        intensity_levels = {
            "beginner": ["Low", "Moderate"],
            "intermediate": ["Moderate"],
            "advanced": ["Moderate", "High"]
        }
        sets_options = {
            "beginner": [2, 3],
            "intermediate": [3, 4],
            "advanced": [4, 5]
        }
        reps_options = {
            "beginner": [10, 12],
            "intermediate": [12, 15],
            "advanced": [15, 20]
        }

        level_text = data['level'].lower()

        # Define workout pool
        exercise_pool = {
            "Overweight": ["Jumping Jacks", "Push-ups", "Jogging", "Lunges", "Burpees", "Step-ups", "Walking"],
            "Obese": ["Walking", "Cycling", "Chair Squats", "Arm Circles", "Resistance Band", "Swimming", "Stretching"],
            "Lean": ["Deadlifts", "Squats", "Push Press", "Pull-ups", "Lunges", "HIIT", "Plank"],
            "Normal": ["Yoga", "Mountain Climbers", "Skipping", "Pilates", "Plank", "Bodyweight Squats", "Stretching"],
            "Custom": ["Push-ups", "Crunches", "Lunges", "Burpees", "Plank", "Jump Rope", "Yoga"]
        }

        pool = exercise_pool.get(body_type, exercise_pool["Custom"])

        # Generate personalized plan
        workout_plan = []
        for i in range(days):
            np.random.shuffle(pool)
            selected_exercises = pool[:3]
            exercises = []
            for ex in selected_exercises:
                exercises.append({
                    "name": ex,
                    "sets": int(np.random.choice(sets_options[level_text])),
                    "reps": int(np.random.choice(reps_options[level_text]))
                })

            workout_plan.append({
                "day": i + 1,
                "exercises": exercises,
                "duration": f"{int(np.random.choice(duration_options[level_text]))} minutes",
                "intensity": str(np.random.choice(intensity_levels[level_text]))
            })

        return jsonify({
            "body_type": body_type,
            "bmi": round(bmi, 2),
            "plan": workout_plan
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🏁 Start app
if __name__ == "__main__":
    app.run(debug=True)
