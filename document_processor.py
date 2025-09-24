import PyPDF2
import pandas as pd
import re
from typing import List, Dict
import streamlit as st
from io import StringIO

class DocumentProcessor:
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.csv']
    
    def process_document(self, uploaded_file) -> List[Dict[str, str]]:
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return self._process_pdf(uploaded_file)
        elif file_extension == 'txt':
            return self._process_text(uploaded_file)
        elif file_extension == 'csv':
            return self._process_csv(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return []
    
    def _process_pdf(self, uploaded_file) -> List[Dict[str, str]]:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            return self._create_chunks(text, uploaded_file.name, 'pdf')
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []
    
    def _process_text(self, uploaded_file) -> List[Dict[str, str]]:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
            return self._create_chunks(text, uploaded_file.name, 'text')
            
        except Exception as e:
            st.error(f"Error processing text file: {str(e)}")
            return []
    
    def _process_csv(self, uploaded_file) -> List[Dict[str, str]]:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Convert CSV to readable text format
            text = f"Financial Data from {uploaded_file.name}:\n\n"
            
            # Add summary information
            text += f"Total rows: {len(df)}\n"
            text += f"Columns: {', '.join(df.columns)}\n\n"
            
            # Convert each row to readable format
            for index, row in df.iterrows():
                text += f"Record {index + 1}:\n"
                for col in df.columns:
                    text += f"  {col}: {row[col]}\n"
                text += "\n"
            
            return self._create_chunks(text, uploaded_file.name, 'csv')
            
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            return []
    
    def _create_chunks(self, text: str, filename: str, file_type: str, 
                      chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, str]]:
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'filename': filename,
                        'file_type': file_type,
                        'chunk_id': chunk_id,
                        'start_char': start,
                        'end_char': end
                    }
                })
                chunk_id += 1
            
            start = end - overlap
        
        return chunks
