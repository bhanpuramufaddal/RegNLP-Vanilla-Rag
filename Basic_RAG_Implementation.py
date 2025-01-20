import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# Step 1: Load the documents
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                documents.extend(file.readlines())
    return documents

# Step 2: Generate Embeddings for Documents
def embed_documents(documents, embedding_model):
    embeddings = embedding_model.encode(documents, convert_to_tensor=True)
    return embeddings.cpu().numpy()  # FAISS requires numpy arrays

# Step 3: Build FAISS Vector Store
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(embeddings)
    return index

# Step 4: Query the Vector Store
def retrieve_documents(query, embedding_model, index, documents, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# # Step 5: Generate Answer Using a Language Model
def generate_answer(question, context, model_name):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create the text-generation pipeline
    nlp_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # Prepare input text
    input_text = f"Answer this question based on the context:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    
    # Generate response
    response = nlp_pipeline(
        input_text,
        max_new_tokens=150,  
        truncation=True,     
        pad_token_id=tokenizer.eos_token_id  
    )

    return response[0]["generated_text"]

# Main Function
def main():
    # Load and prepare documents
    file_path = "/content/gdrive/My Drive/RegNLPDocuments"
    documents = load_documents_from_folder(file_path)

    print('Load documents step complete')

    # Use Sentence Transformers for embeddings
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    print("Model loaded successfully.")
    
    document_embeddings = embed_documents(documents, embedding_model)
    print('Document embeddings created')

    # Build FAISS index
    faiss_index = build_faiss_index(document_embeddings)
    print('Faiss Index - ', faiss_index)

    query = "Question: What is the legal basis for ADGM’s legal system?"

    # Retrieve relevant documents
    top_documents = retrieve_documents(query, embedding_model, faiss_index, documents)
    context = " ".join(top_documents)

    # Generate an answer
    answer = generate_answer(query, context, "gpt2")
    print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()


# Example Output by Model - 

# Question: What is the legal basis for ADGM’s legal system?

# Answer: The law must be interpreted, understood and adopted and relevant. ADGM requires a'system to be followed' which 
# follows an established and consistent set of common law statutory rules that is based on the rules of common legal practice, 
# as implemented by the applicant and adopted by the Government of the Commonwealth or other responsible statutory authorities 
# (i.e. the Council of the United Kingdom). The use of common law statutory code legislation is the best way for ADRM to identify, 
# identify and implement common law rule issues and to identify any potential unintended consequences. These rules generally have no major 
# conflict of interest and thus should be disregarded. In addition, the use of common law statutory code legislation should generally be 
# adhered to for the sole purpose of preventing unlawful and unreasonable

