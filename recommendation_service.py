import os
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Flask application
app = Flask(__name__)

# --- Mock Data and ML Model Setup ---

# This is our catalog of courses. In a real-world scenario, this would come from a database.
COURSES_DATA = {
    'course_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'title': [
        'Introduction to Python Programming',
        'Advanced Python for Data Science',
        'Web Development with Flask',
        'Machine Learning Fundamentals',
        'Deep Learning with TensorFlow',
        'Data Visualization with Matplotlib',
        'Introduction to SQL Databases',
        'Cloud Computing on Azure',
        'Building APIs with Python',
        'Cybersecurity Essentials'
    ],
    'tags': [
        'python, programming, beginner, development',
        'python, data science, machine learning, advanced',
        'python, web development, flask, apis',
        'machine learning, data science, python, theory',
        'machine learning, deep learning, tensorflow, python',
        'data science, python, visualization, matplotlib',
        'databases, sql, data management, beginner',
        'cloud, azure, infrastructure, devops',
        'python, apis, web development, backend',
        'security, cybersecurity, networking, beginner'
    ]
}

courses_df = pd.DataFrame(COURSES_DATA)

# --- Machine Learning Recommendation Logic (Content-Based Filtering) ---
# This part of the code sets up the recommendation engine.
# It converts the text 'tags' for each course into a numerical format
# so we can calculate similarity between them.

# 1. Create a TF-IDF Vectorizer
# This tool helps measure how important a word is to a document in a collection.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# 2. Create the TF-IDF matrix by fitting and transforming the data
# This creates a matrix where rows are courses and columns are unique words in the tags.
tfidf_matrix = tfidf_vectorizer.fit_transform(courses_df['tags'])

# 3. Calculate the cosine similarity matrix
# This matrix shows how similar each course is to every other course based on their tags.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- API Endpoints ---

@app.route('/')
def home():
    """A simple home page to confirm the service is running."""
    return "<h1>Course Recommendation and Analytics Service</h1><p>Use the /analytics or /recommend endpoints.</p>"

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """
    This endpoint provides mock analytics data.
    In a real application, this would be generated from user activity logs.
    """
    analytics_data = {
        'total_courses_offered': len(courses_df),
        'students_enrolled': 1578,
        'courses_completed': 890,
        'most_popular_course': 'Introduction to Python Programming'
    }
    return jsonify(analytics_data)

@app.route('/recommend', methods=['POST'])
def recommend_courses():
    """
    This endpoint generates course recommendations based on a user's interests.
    It expects a JSON payload with a 'title' of a course the user liked.
    """
    request_data = request.get_json()
    if not request_data or 'title' not in request_data:
        return jsonify({'error': 'Please provide a course title you are interested in.'}), 400

    title = request_data['title']

    # Find the index of the course that matches the title
    if title not in courses_df['title'].values:
        return jsonify({'error': f"Course with title '{title}' not found."}), 404

    idx = courses_df[courses_df['title'] == title].index[0]

    # Get the pairwise similarity scores of all courses with that course
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the courses based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar courses (excluding the course itself)
    sim_scores = sim_scores[1:6]

    # Get the course indices
    course_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 5 most similar courses
    recommendations = courses_df['title'].iloc[course_indices].tolist()

    return jsonify({'recommendations': recommendations})

# --- Main execution ---
if __name__ == '__main__':
    # Use the PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
