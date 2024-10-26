#############################################################################################
# Extractive QA using Haystack Pipeline with In-Memory Document Store and Chunking
#############################################################################################
# This script demonstrates how to create an Extractive Question Answering (QA) system
# using Haystack's pipeline. It utilizes a specific extractive model from Hugging Face 
# ("deepset/bert-base-uncased-squad2" with a max of 512 tokens) instead of the default RoBERTa model. 
# Additionally, it implements chunking to ensure that the input texts stay within the token limits of the model.

# Installations:
# pip install haystack-ai
# pip install sentence-transformers>=3.0.0
# pip install langchain-text-splitters tiktoken

# Sample text for testing extraction
text = """
In 2023, the global technology industry saw a remarkable transformation driven by advancements in artificial intelligence, 
machine learning, and the Internet of Things (IoT). According to the International Data Corporation (IDC), global spending 
on AI systems is projected to reach $500 billion, demonstrating a compound annual growth rate (CAGR) of 20% over the next 
five years. Companies across various sectors have begun to integrate AI into their operations, leading to increased efficiency 
and reduced costs.

In the automotive industry, electric vehicles (EVs) gained significant traction, with sales increasing by 35% compared to 
2022. Major automakers such as Tesla, Ford, and General Motors announced plans to expand their EV offerings. In addition, 
governments worldwide have implemented incentives to promote the adoption of electric vehicles, including tax credits and 
subsidies. The push for sustainable transportation solutions is also supported by advancements in battery technology, 
which have improved the range and affordability of EVs.

The healthcare sector has also experienced a significant transformation due to technology. Telehealth services surged in 
popularity, with a report from McKinsey & Company indicating that 40% of patients utilized telehealth options in 2023. 
This shift has made healthcare more accessible, particularly for individuals in rural areas. Furthermore, artificial 
intelligence has been instrumental in improving diagnostic accuracy, with AI algorithms capable of analyzing medical images 
and predicting patient outcomes with high precision.

In the finance sector, blockchain technology continued to gain momentum. The total market capitalization of cryptocurrencies 
surpassed $2 trillion, with Bitcoin and Ethereum remaining the most dominant currencies. Traditional financial institutions 
have begun to explore blockchain solutions for various applications, including cross-border payments, smart contracts, and 
asset tokenization. Regulatory bodies are also working to establish frameworks to ensure the security and legitimacy of 
cryptocurrency transactions.

Moreover, the retail industry underwent significant changes due to the increasing reliance on e-commerce. Online sales 
accounted for over 25% of total retail sales in 2023, according to the U.S. Department of Commerce. Retailers such as 
Amazon and Walmart expanded their online marketplaces, while smaller businesses also embraced e-commerce to reach a broader 
audience. Personalized shopping experiences powered by AI and machine learning algorithms have enhanced customer satisfaction 
and loyalty.

Sustainability emerged as a key focus for businesses across all sectors. Companies are increasingly adopting sustainable 
practices to reduce their carbon footprints and improve their environmental impact. According to a survey by Deloitte, 75% 
of consumers are willing to pay more for products from companies that demonstrate a commitment to sustainability. As a 
result, many organizations have set ambitious sustainability goals, including carbon neutrality by 2030 and zero waste to landfill.

In education, technology played a pivotal role in enhancing the learning experience. Online learning platforms gained popularity, 
providing access to quality education for students worldwide. The COVID-19 pandemic accelerated the adoption of remote learning, 
leading to innovations in digital classrooms and interactive learning materials. Educators have also begun to integrate gamification 
and adaptive learning technologies to engage students and personalize their educational journeys.

The entertainment industry has also seen a shift due to the rise of streaming services. Platforms like Netflix, Hulu, and Disney+ 
have gained millions of subscribers, leading to a decline in traditional cable television viewership. Original content production 
has skyrocketed, with streaming services investing heavily in creating exclusive shows and movies to attract and retain subscribers. 
This shift has also changed how audiences consume media, with binge-watching becoming a common behavior.

As we move forward into 2024, the convergence of these technological advancements will continue to shape various industries. 
Businesses that embrace innovation and prioritize sustainability will likely thrive in this rapidly changing landscape. The 
integration of technology in everyday life is expected to deepen, influencing how we work, learn, and interact with one another. 
Stakeholders across sectors must adapt to these changes and recognize the opportunities presented by emerging technologies.
"""

# Step 0: Split text using the max token limit of the model 
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=512
)
texts = text_splitter.split_text(text)

# Step 1: Prepare data in Document format for Document Store
from haystack import Document

documents = [Document(content=val) for val in texts]

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
reader = ExtractiveReader(model="deepset/bert-base-uncased-squad2")
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
query = "What is the projected global spending on AI systems in 2023?"
result = extractive_qa_pipeline.run(
    data={"embedder": {"text": query}, "retriever": {"top_k": 3}, "reader": {"query": query, "top_k": 1}}
)

# Step 5: Extract and print the response
response = result["reader"]["answers"][0].data
print(f"Response: {response}")  # Expected response: $500 billion