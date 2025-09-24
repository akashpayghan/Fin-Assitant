from openai import OpenAI
import chromadb
import os
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv
import uuid

load_dotenv()

class RAGEngine:
    
    def __init__(self):
        # Initialize OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            raise ValueError("Please set your OPENAI_API_KEY in the .env file")
            
        self.client = OpenAI(api_key=api_key)
        
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        self.chat_model = os.getenv('CHAT_MODEL', 'gpt-4o-mini')
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "financial_documents"
        
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Personal finance documents and data"}
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def add_documents(self, document_chunks: List[Dict[str, Any]]) -> bool:
        try:
            if not document_chunks:
                return False
            
            texts = [chunk['text'] for chunk in document_chunks]
            metadatas = [chunk['metadata'] for chunk in document_chunks]
            ids = [str(uuid.uuid4()) for _ in document_chunks]
            
            embeddings = self.generate_embeddings(texts)
            
            if not embeddings:
                return False
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            st.success(f"Successfully added {len(document_chunks)} document chunks to knowledge base!")
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to database: {str(e)}")
            return False
    
    def search_similar_documents(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i] 
                })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        try:
            # Create context from retrieved documents
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('filename', 'Unknown')}\n{doc['text']}"
                for doc in context_documents
            ])
            
            # Create prompt with financial expertise
            system_prompt = """You are a knowledgeable personal finance assistant. 
            Use the provided context from the user's financial documents to answer their questions.
            
            Guidelines:
            - Provide specific, actionable financial advice
            - Reference the user's actual financial data when available
            - Explain financial concepts clearly
            - Suggest practical next steps
            - If you need more information, ask specific questions
            - Always prioritize the user's financial wellbeing and security
            
            Context Information:
            {context}
            """
            
            messages = [
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": query}
            ]
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.3, 
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request. Please try again."
    
    def get_database_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.collection.get()
            
            stats = {
                'total_documents': len(collection_info['documents']),
                'unique_files': len(set([
                    metadata.get('filename', 'Unknown') 
                    for metadata in collection_info['metadatas']
                ])),
                'file_types': list(set([
                    metadata.get('file_type', 'Unknown') 
                    for metadata in collection_info['metadatas']
                ]))
            }
            
            return stats
            
        except Exception as e:
            st.error(f"Error getting database stats: {str(e)}")
            return {'total_documents': 0, 'unique_files': 0, 'file_types': []}
    
    def clear_database(self) -> bool:
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Personal finance documents and data"}
            )
            return True
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")
            return False
