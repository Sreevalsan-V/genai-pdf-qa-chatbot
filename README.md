## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
The goal is to create a system that can read and extract text from a PDF, use LangChain to process this text, and then generate relevant answers based on user queries. This will automate the extraction of specific information from documents and make it easily accessible.

### DESIGN STEPS:

#### STEP 1:
Set up the OpenAI API key to access the language model.

#### STEP 2:
Create a function to extract text from a provided PDF file.

#### STEP 3:
Define the LangChain prompt template to guide the model in understanding the document's content.

#### STEP 4:
Implement the model using OpenAI and integrate it with LangChain.

#### STEP 5:
Build a function to query the PDF content using the extracted text and provide answers to user questions.

#### STEP 6:
Test the system with a sample PDF and a user query.

### PROGRAM:
```py
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

prompt_template = """
You are a knowledgeable assistant that can answer questions based on the content of a document. 
The document content is as follows:

{document_content}

Please answer the following question:
{question}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["document_content", "question"])

def query_pdf_content(pdf_path: str, user_query: str) -> str:
    document_content = extract_text_from_pdf(pdf_path)
    model = OpenAI(temperature=0, openai_api_key="your-api-key")
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run({"document_content": document_content, "question": user_query})
    return response

pdf_path = "/content/pokemon_index.pdf"
user_query = "what are all the pokemon with flying and fire type?"
response = query_pdf_content(pdf_path, user_query)

print(f"Response: {response}")

```

### OUTPUT:
![image](https://github.com/user-attachments/assets/4e3cbb32-d816-4519-a1b2-d10e50c6b664)


### RESULT:
The question-answering chatbot successfully extracts information from the provided PDF document and provides relevant answers based on the content. This demonstrates the chatbot's capability to understand and respond to queries related to specific documents, making it a useful tool for document processing and information retrieval.
