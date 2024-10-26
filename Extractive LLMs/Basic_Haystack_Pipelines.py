#############################################################################################
# Extractive QA using Haystack pipeline and In-Memory Document Store
#############################################################################################

# Installations:
# pip install haystack-ai
# pip install sentence-transformers>=3.0.0

# Sample Data:
sample_texts = {
    "text1": "In 2023, the GDP of Country A grew by 3.5%, reaching a total of $1.5 trillion. This growth is attributed to increased consumer spending and a booming technology sector.",
    "text2": "As of 2024, the population of City B is approximately 2.3 million, reflecting an increase of 4% from the previous year. The city has implemented new housing policies to accommodate the growing population.",
    "text3": "In the 2022 season, Team C won 28 out of 50 games, achieving a win rate of 56%. They finished first in their division and qualified for the playoffs.",
    "text4": "According to the latest environmental report, Company D reduced its carbon emissions by 20% over the past five years, bringing the total emissions down to 80,000 tons per year.",
    "text5": "In 2023, School E had a graduation rate of 92%, with 300 out of 325 students successfully completing their programs. The school also reported an increase in student enrollment by 15%."
}

# Step 1: Prepare data in Document format for Document Store
from haystack import Document

documents = [Document(content=val) for val in sample_texts.values()]

# Step 2: Save Documents in DocumentStore using Pipeline (InMemoryDocumentStore)
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

# Define the model for embeddings
model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Initialize the Document Store
document_store = InMemoryDocumentStore()

# Create an indexing pipeline
indexing_pipeline = Pipeline()

# Add components to the indexing pipeline
indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model), name="embedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
indexing_pipeline.connect("embedder.documents", "writer.documents")

# Run the indexing pipeline to save documents
indexing_pipeline.run({"documents": documents})

# Step 3: Create a pipeline for Question Answering
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersTextEmbedder

# Initialize retriever and reader
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
reader = ExtractiveReader()
reader.warm_up()  # Warm up the reader to improve performance

# Create an extractive QA pipeline
extractive_qa_pipeline = Pipeline()
extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
extractive_qa_pipeline.add_component(instance=reader, name="reader")

# Connect components in the extractive QA pipeline
extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

# Step 4: Execute the pipeline with a query
query = "What was the graduation rate for School E in 2023?"
result = extractive_qa_pipeline.run(
    data={"embedder": {"text": query}, "retriever": {"top_k": 3}, "reader": {"query": query, "top_k": 1}}
)

# Step 5: Extract and print the response
response = result["reader"]["answers"][0].data
print(f"Response: {response}")  # Expected response: 92%
