import os
from pathlib import Path

from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.components.converters import PDFToTextConverter, TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

api_key = "api"  # Replace with your actual OpenAI API key

# Set up the path to the data directory
HERE = Path(__file__).resolve().parent
data_dir = HERE / "data"

# Collect all text and PDF files in the data directory
file_paths = [data_dir / name for name in os.listdir(data_dir) if name.endswith(('.txt', '.pdf'))]

# Initialize the document store
document_store = ChromaDocumentStore()

# Create the indexing pipeline
indexing = Pipeline()
indexing.add_component("text_converter", TextFileToDocument())
indexing.add_component("pdf_converter", PDFToTextConverter())
indexing.add_component("writer", DocumentWriter(document_store))

# Connect components in the pipeline
indexing.connect("text_converter", "writer")
indexing.connect("pdf_converter", "writer")

# Run the indexing pipeline for text files
text_files = [file for file in file_paths if file.suffix == '.txt']
if text_files:
    indexing.run({"text_converter": {"sources": text_files}})

# Run the indexing pipeline for PDF files
pdf_files = [file for file in file_paths if file.suffix == '.pdf']
if pdf_files:
    indexing.run({"pdf_converter": {"sources": pdf_files}})
# Create some initial documents for the RAG pipeline with first aid information
initial_documents = [
    Document(content="First aid is the immediate care given to a person who has been injured or suddenly taken ill. It includes self-help and care delivered by other people."),
    Document(content="The ABC of first aid stands for Airway, Breathing, and Circulation. These are the primary assessments in any emergency situation."),
    Document(content="For minor burns, run cool water over the affected area for at least 10 minutes. Do not apply ice directly to the burn."),
    Document(content="If someone is choking, perform the Heimlich maneuver by standing behind them and placing a fist between their navel and rib cage, then pull sharply inwards and upwards."),
    Document(content="In case of a suspected heart attack, have the person sit down, rest, and try to keep calm. Call emergency services immediately.")
]
document_store.write_documents(initial_documents)
# Build a RAG pipeline
prompt_template = """
You are an AI assistant specializing in first aid. Use the following documents to answer the question. If the information isn't in the documents, say you don't have enough information to answer safely.

Documents:
{% for doc in documents %}
Document {{ loop.index }} (Relevance: {{ doc.score }}):
{{ doc.content }}
Source: {{ doc.meta.source if doc.meta.source else "Unknown" }}

{% endfor %}
Question: {{question}}

Answer:
"""

# Initialize the retriever, prompt builder, and the OpenAI generator
retriever = ChromaQueryTextRetriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(api_key=Secret.from_token(api_key))

# Create and configure the RAG pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Ask a question using the RAG pipeline
question = "Who is first aid?"
results = rag_pipeline.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    }
)

print(results["llm"]["replies"])

# Create the querying pipeline for document retrieval
querying = Pipeline()
querying.add_component("retriever", retriever)

# Run the querying pipeline
query_results = querying.run({"retriever": {"query": "Variable declarations", "top_k": 3}})

# Print the retrieved documents
for d in query_results["retriever"]["documents"]:
    print(d.meta, d.score)

