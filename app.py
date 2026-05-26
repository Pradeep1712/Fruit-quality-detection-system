from flask import Flask, request, jsonify, render_template
import os
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI Client 
# (Ensure your OPENAI_API_KEY environment variable is set in your terminal)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route('/')
def index():
    # Renders the HTML frontend from the templates/ folder
    return render_template('index.html')

@app.route('/upload_blob', methods=['POST'])
def process_frames():
    """
    MOCK BACKEND: Bypasses TensorFlow since local .h5 models are absent.
    Returns simulated probabilities to allow UI and API testing.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Fake probabilities to simulate a working model 
    # (e.g., 95% confident it's an Apple, 90% confident it's Fresh)
    mock_type_probs = [0.95, 0.03, 0.02]  # Corresponds to ['Apple', 'Banana', 'Orange']
    mock_fresh_probs = [0.90, 0.10]       # Corresponds to ['Fresh', 'Rotten']
    
    return jsonify({
        'type_probs': mock_type_probs,
        'fresh_probs': mock_fresh_probs
    })

@app.route('/get_nutrition', methods=['POST'])
def get_nutrition():
    data = request.json
    fruit_name = data.get('fruit_name', 'Unknown Fruit')
    
    try:
        # Request a concise summary from GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a concise nutritionist. Provide a brief 2-sentence nutritional summary."},
                {"role": "user", "content": f"Provide nutritional info for a {fruit_name}."}
            ],
            max_tokens=100
        )
        nutrition_info = response.choices[0].message.content.strip()
        return jsonify({'nutrition': nutrition_info})
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return jsonify({'error': 'Failed to fetch nutrition data. Check API key.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)