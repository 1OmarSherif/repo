from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import google.generativeai as genai

# Configure the generative AI model
genai.configure(api_key="AIzaSyCOkaGL-rl8_zXAFTIY6O9zyJjdtZp_wUE")
model = genai.GenerativeModel("gemini-1.5-flash")

# Flask app initialization
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and preprocess data
try:
    file_path = './final.xlsx'
    df = pd.read_excel(file_path)  # Ensure the Excel file has all required columns
    df.fillna('', inplace=True)  # Replace NaN values with empty strings
    X = df[['skills_required', 'difficulty_level', 'category', 'keywords']].astype(str).agg(' '.join, axis=1)
    y = df['title']

    # Oversampling to handle imbalance
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X.to_frame(name='combined_features'), y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled['combined_features'], y_resampled, test_size=0.5, random_state=42, stratify=y_resampled
    )

    # Train pipeline
    pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
    pipeline.fit(X_train, y_train)
except Exception as e:
    print(f"Error initializing data or pipeline: {e}")
    pipeline = None

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if pipeline is None:
            return jsonify({"error": "Pipeline not initialized"}), 500

        # Extract data from request
        data = request.json
        skills_required = data.get('skills_required', '')
        difficulty_level = data.get('difficulty_level', '')
        category = data.get('category', '')
        keywords = data.get('keywords', '')

        # Validate inputs
        if not all([skills_required, difficulty_level, category, keywords]):
            return jsonify({"error": "Missing required fields"}), 400

        # Combine features
        combined_features = f"{skills_required} {difficulty_level} {category} {keywords}"

        # Predict top 5 projects
        probabilities = pipeline.predict_proba([combined_features])
        top_indices = probabilities[0].argsort()[-5:][::-1]

        top_predictions = []
        for i in top_indices:
            title = pipeline.classes_[i]
            project = df[df['title'] == title].iloc[0]

            # Include additional fields
            top_predictions.append({
                'id': project.get('id', ''),
                'title': title,
                'description': project.get('description', ''),
                'image': project.get('image', ''),
                'link': project.get('link', ''),
                'probability': probabilities[0][i]
            })

        return jsonify({'top_predictions': top_predictions})

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': "Internal server error"}), 500

# Chatbot endpoint
@app.route("/chat", methods=["POST"])
def chatbot():
    try:
        # Get user input from request
        user_input = request.json.get("user_input", "")
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400

        # Generate AI response
        response = model.generate_content(user_input)
        return jsonify({"AI_response": response.text})

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
