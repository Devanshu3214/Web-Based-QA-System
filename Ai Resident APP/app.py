from flask import Flask, render_template, request, jsonify
import warnings

# Import the necessary functions from llm_processing
from new import load_and_process_documents, split_text, create_vector_index, create_qa_chain, get_answer

warnings.filterwarnings("ignore")
app = Flask(__name__)

# Set up the knowledge base using URLs
urls = [
    'https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file#milestone-papers',
    'https://stanford-cs324.github.io/winter2022/lectures/',
    'https://stanford-cs324.github.io/winter2022/lectures/introduction/',
    'https://stanford-cs324.github.io/winter2022/lectures/capabilities/',
    'https://stanford-cs324.github.io/winter2022/lectures/data/',
    'https://stanford-cs324.github.io/winter2022/lectures/training/',
    'https://stanford-cs324.github.io/winter2022/lectures/environment/',
]

# Initialize the LLM processing components once to create the knowledge base

context = load_and_process_documents(urls)
texts = split_text(context)
vector_index = create_vector_index(texts)
qa_chain = create_qa_chain()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/create_knowledgebase", methods=["POST"])
def create_knowledgebase():
    # Assuming this is a placeholder for any future reinitialization of the knowledge base
    return jsonify({"status": "Knowledge base created"}), 200

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    return get_chat_response(msg)

def get_chat_response(text):
        response = get_answer(vector_index, qa_chain, text)
        return response['output_text']

if __name__ == '__main__':
    app.run(debug=True)
