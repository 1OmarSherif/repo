from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
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
app = Flask(__name__)  # Use __name__ instead of _name_
CORS(app)  # Enable CORS for the entire app

# Example data and pipeline
file_path = './final.xlsx'
df = pd.read_excel(file_path)  # Use pd.read_excel directly to read the Excel file into a DataFrame
X = df[['skills_required', 'difficulty_level', 'category', 'keywords']].astype(str).agg(' '.join, axis=1)
y = df['title']

# Oversampling to handle imbalance
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X.to_frame(name='combined_features'), y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled['combined_features'], y_resampled, test_size=0.5, random_state=42, stratify=y_resampled
)

# Pipeline
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
pipeline.fit(X_train, y_train)

# Project prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract input data
        skills_required = data.get('skills_required', '')
        difficulty_level = data.get('difficulty_level', '')
        category = data.get('category', '')
        keywords = data.get('keywords', '')

        # Combine fields into a single feature
        combined_features = f"{skills_required} {difficulty_level} {category} {keywords}"

        # Get probabilities for each class
        probabilities = pipeline.predict_proba([combined_features])

        # Get the top 5 predictions
        top_indices = probabilities[0].argsort()[-5:][::-1]
        top_predictions = []

        for i in top_indices:
            title = pipeline.classes_[i]
            project = df[df['title'] == title].iloc[0]

            # Include additional fields
            top_predictions.append({
                'id': project['id'],  # Ensure the ID column exists in your data
                'title': title,
                'description': project['description'],
                'image': project['image'],  # Link or path to the image
                'link': project['link'],    # Link to the project
                'probability': probabilities[0][i]
            })

        return jsonify({'top_predictions': top_predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define the chatbot API endpoint
@app.route("/chat", methods=["POST"])
def chatbot():
    try:
        # Get user input from the request JSON body
        user_input = request.json.get("user_input", "")
        
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400
        
        # Generate response using the generative AI model
        response = model.generate_content(user_input)
        
        # Return the AI response as JSON
        return jsonify({"AI_response": response.text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
