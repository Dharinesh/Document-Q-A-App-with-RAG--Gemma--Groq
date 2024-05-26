# Document Question Answering App with RAGs, Gemma, and GroqAPI

This project is a Streamlit application that implements a document question-answering system using the Retrieval-Augmented Generation (RAG) approach. 
It combines vector similarity search for retrieving relevant document chunks with a language model for generating answers based on the retrieved context.

 # What is RAG?

Retrieval augmented generation, or RAG, is an architectural approach that can improve the efficiency of large language model (LLM) applications by leveraging custom data. This is done by retrieving data/documents relevant to a question or task and providing them as context for the LLM. RAG has shown success in support chatbots and Q&A systems that need to maintain up-to-date information or access domain-specific knowledge.

The main Drawbacks of LLM without RAG is that the models tend to hallucinate and provide its own responses without any context and as a result, the response might not be accurate
