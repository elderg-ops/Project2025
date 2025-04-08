from flask import Flask, request, jsonify, render_template, send_file, make_response, Response, redirect, url_for
import google.generativeai as genai
import os
import re
import pickle
import pdfkit
import io
import json
import numpy as np
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading

# 🔐 Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyD4h0g4LQ3B9Ljgf5N2e9QgiugOHirlmWA"
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
camera = cv2.VideoCapture(0)

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

@app.route('/tracker')
def tracker():
    return render_template("tracker.html")

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

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    raw_html = request.form['diet_html']  # ✅ Fixed here

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 10px;
                text-align: center;
            }}
            th {{
                background-color: #007bff;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h2>Personalized Diet Plan</h2>
        {raw_html}
    </body>
    </html>
    """

    # 👇 Path to wkhtmltopdf (update this as per your system)
    config = pdfkit.configuration(wkhtmltopdf='C:\\Users\\Admin\\Desktop\\Project2025\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')

    # Generate PDF from HTML string
    pdf = pdfkit.from_string(full_html, False, configuration=config)

    # Return PDF as a downloadable response
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=diet_plan.pdf'
    return response

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

# TTS Setup
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

# Global variables
counter = 0
stage = None
last_warning = ""
last_speak_time = 0

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def speak_warning(warning):
    global last_warning, last_speak_time
    current_time = time.time()
    if warning != last_warning or (current_time - last_speak_time > 10):
        threading.Thread(target=speak_text, args=(warning,)).start()
        last_warning = warning
        last_speak_time = current_time


pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def detect_pushups(frame):
        global counter, stage
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            back_angle = calculate_angle(shoulder, hip, ankle)

            if elbow_angle > 160:
                stage = "up"
            if elbow_angle < 90 and stage == "up":
                stage = "down"
                counter += 1

            bad_form_warning = ""
            if elbow_angle > 90 and stage == "down":
                bad_form_warning = "Go lower!"
            elif back_angle < 150:
                bad_form_warning = "Keep your back straight!"
            elif hip[1] > shoulder[1] + 0.05:
                bad_form_warning = "Don't drop your hips!"

            if bad_form_warning:
                speak_warning(bad_form_warning)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(image, f'Elbow: {int(elbow_angle)}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f'Back: {int(back_angle)}', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f'Push-ups: {counter}', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if bad_form_warning:
                cv2.putText(image, bad_form_warning, (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return image

def generate_frames():
    while True:
        success, frame = camera.read()
        frame = cv2.resize(frame, (640, 480))
        if not success:
            break
        else:
            processed_frame = detect_pushups(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    global streaming
    streaming = False
    camera.release()
    return redirect(url_for('index'))

# 🏁 Start app
if __name__ == "__main__":
    app.run(debug=True)