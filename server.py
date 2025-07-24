from flask import Flask, request, jsonify
from flask_cors import CORS
import worker  # your existing worker.py with init_llm, process_document, process_prompt

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = f"./uploads/{file.filename}"
    file.save(file_path)
    worker.process_document(file_path)
    return jsonify({'message': 'Document processed successfully'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    answer = worker.process_prompt(question)
    return jsonify({'answer': answer})

@app.route('/test-cors', methods=['GET'])
def test_cors():
    return jsonify({'message': 'CORS is working!'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

