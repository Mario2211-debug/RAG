import os
import faiss
import pickle
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("Qwen/Qwen3-0.6B", device="cpu")



def save_vector_db(index, chunks):
    faiss.write_index(index, "index.faiss")
    
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)



def load_vector_db():
    index = faiss.read_index("index.faiss")
    
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def load_docs(folder="docs"):
    documents = []

    for filename in os.listdir(folder):
        reader = PdfReader(f"docs/{filename}")
        for page in reader.pages:
            documents.append(page.extract_text())

        """
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                documents.append(file.read())
        """
    return documents


def create_chunks(document, chunk_size=1000):
    # print(documents)
    chunks = []

    for lines in document:
        for i in range(0, len(document), chunk_size):
            chunk = lines[i:i + chunk_size]
            chunks.append(chunk)
    return chunks


def create_embeddings(chunks):
    return model.encode(
        chunks,
        convert_to_numpy=True,
        batch_size=1,
        show_progress_bar=True
        )


def create_vector_db(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)
    return index


def search(question, index, chunks, k=3):
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True)

    distances, indexes = index.search(question_embedding, k)

    results = []

    for i in indexes[0]:
        results.append(chunks[i])

    return results


def ask_llm(question, context):
    prompt = f""" Responde à pergunta utilizando apenas
    o contexto fornecido.Se a resposta não existir no contexto,
    diz: "Não encontrei essa informação."
    CONTEXTO:
        {context}

    PERGUNTA:
        {question}
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2",
              "prompt": prompt,
              "stream": False}
                             )
    return response.json()["response"]


if __name__ == "__main__":
    try:
        if os.path.exists("index.faiss") and os.path.exists("chunks.pkl"):
            print("A carregar...")
            index, chunks = load_vector_db()
        else:
            print("A criar vector DB...")
            documents = load_docs()
            chunks = create_chunks(documents)
            embeddings = create_embeddings(chunks)
            index = create_vector_db(embeddings)
        while True:
            question = input("\nPergunta: ")

            if question.lower() == "exit":
                break

            context_chunks = search(
                question, index, chunks)

            context = "\n\n".join(context_chunks)
            answer = ask_llm(question, context)

            print("\nResposta: ")
            print(answer)
    except Exception as e:
        print(f"Error: {e}")
