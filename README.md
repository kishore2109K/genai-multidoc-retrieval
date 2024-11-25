## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
The growing volume of research articles across various domains makes it increasingly difficult for users to retrieve and synthesize information efficiently. Traditional search systems often retrieve results that lack relevance or context. A multidocument retrieval agent powered by LlamaIndex addresses this problem by enabling a modular, efficient, and context-aware query-response system, providing users with accurate and concise information from multiple sources.
### DESIGN STEPS:

#### STEP 1:
Collect research articles (PDFs or text files) and preprocess them to extract clean text. Split the text into manageable chunks for efficient indexing.
#### STEP 2:
Create indexes for the document chunks using LlamaIndex and vector embeddings. Set up tools for semantic search (vector tool) and summarization (summary tool).
#### STEP 3:
Combine the tools into a retrieval agent using LlamaIndex's AgentRunner. Accept user queries, retrieve relevant chunks, and provide synthesized responses.
### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

# List of papers to load
papers = [
    "knowledge_card.pdf",
    "swebench.pdf",
    "longlora.pdf"
]

# Loading vector tools and summary tools for each paper
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print("Loading,wait")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]


from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=False
)
agent = AgentRunner(agent_worker)


print("Available documents are:")
for p in papers:
    print(p)


question = input("Ask your query related to these documents: \n")

response = agent.query(question) 

print("Response: \n")
print(response)
```

### OUTPUT:
![exp 4o](https://github.com/user-attachments/assets/c65438a4-ecb9-489a-b7a6-75820bbc4ed4)


### RESULT:
Thus, the multidocument retrieval agent using LlamaIndex was successfully implemented, efficiently synthesizing accurate responses from multiple research articles and adaptable to various domains.
