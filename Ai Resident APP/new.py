import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

urls = [
    'https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file#milestone-papers',
    'https://stanford-cs324.github.io/winter2022/lectures/',
    'https://stanford-cs324.github.io/winter2022/lectures/introduction/',
    'https://stanford-cs324.github.io/winter2022/lectures/capabilities/',
    'https://stanford-cs324.github.io/winter2022/lectures/data/',
    'https://stanford-cs324.github.io/winter2022/lectures/training/',
    'https://stanford-cs324.github.io/winter2022/lectures/environment/',
]

# Set the Google API key

# Load and process documents from URLs
def load_and_process_documents(urls):
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    context = "\n\n".join(str(p.page_content) for p in data)
    return context

# Split text into chunks
def split_text(context, chunk_size=7000, chunk_overlap=1500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print("The total number of words in the context:", len(context))
    texts = text_splitter.split_text(context)
    print(len(texts))
    return texts

# Create embeddings and vector index
def create_vector_index(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
    return vector_index

# Create the QA chain
def create_qa_chain():

    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    # provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
    # """

    prompt_template= """You are the help assistant. Answer based on the context provided.  context: {context} input: {question} answer: """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.environ["GOOGLE_API_KEY"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    print('Done')
    return chain

# Process the question and get the answer
def get_answer(vector_index, qa_chain, question):
    docs = vector_index.get_relevant_documents(question)
    print(docs)
    response = qa_chain({"input_documents":docs, "question": question}, return_only_outputs=True)
    return response

# URLs to load and process


if __name__ == "__main__":
    context = load_and_process_documents(urls)
    texts = split_text(context)
    vector_index = create_vector_index(texts)
    qa_chain = create_qa_chain()

    # Example question
    # question = "What are some milestone model architectures and papers in the last few years"
    question = "What are the layers in a transformer block"
    # question = "What are trending llms"
    # question= "Tell me about datasets used to train LLMs and how they’re cleaned"
    answer = get_answer(vector_index, qa_chain, question)
    print(answer)
