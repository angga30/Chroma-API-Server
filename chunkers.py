from abc import ABC, abstractmethod
import re
import json
from bs4 import BeautifulSoup
import uuid
from typing import List, Dict, Any, Tuple, Optional

class BaseChunker(ABC):
    """Base abstract class for all chunkers"""
    
    @abstractmethod
    def chunk(self, content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Chunk the content into smaller pieces
        
        Args:
            content: The content to chunk
            chunk_size: The target size of each chunk
            chunk_overlap: The overlap between chunks
            
        Returns:
            List of dictionaries with 'content' and 'metadata' keys
        """
        pass

class TextChunker(BaseChunker):
    """Chunker for plain text content"""
    
    def _validate_metadata_value(self, value):
        """Ensure metadata value is a valid type (str, int, float, bool, None)"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Convert lists to string
        elif isinstance(value, list):
            return ' '.join(str(item) for item in value)
        # Convert other types to string
        else:
            return str(value)
    
    def chunk(self, content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Chunk text content by paragraphs, then sentences, then characters"""
        # First try to split by paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If paragraph fits in current chunk, add it
            if current_size + len(para) <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
                current_size += len(para) + 2  # +2 for the newlines
            else:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append({
                        "content": current_chunk,
                        "metadata": {
                            "chunk_type": self._validate_metadata_value("text"),
                            "chunk_id": str(uuid.uuid4()),
                            "chunk_index": len(chunks)
                        }
                    })
                
                # Start a new chunk with this paragraph
                # If paragraph is larger than chunk_size, we'll need to split it further
                if len(para) > chunk_size:
                    # Split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    
                    current_chunk = ""
                    current_size = 0
                    
                    for sentence in sentences:
                        if current_size + len(sentence) <= chunk_size:
                            if current_chunk:
                                current_chunk += " "
                            current_chunk += sentence
                            current_size += len(sentence) + 1  # +1 for the space
                        else:
                            # If sentence doesn't fit and we have content, add current chunk
                            if current_chunk:
                                chunks.append({
                                    "content": current_chunk,
                                    "metadata": {
                                        "chunk_type": self._validate_metadata_value("text"),
                                        "chunk_id": str(uuid.uuid4()),
                                        "chunk_index": len(chunks)
                                    }
                                })
                            
                            # If sentence is still too long, split by characters
                            if len(sentence) > chunk_size:
                                for i in range(0, len(sentence), chunk_size - chunk_overlap):
                                    chunk_text = sentence[i:i + chunk_size]
                                    chunks.append({
                                        "content": chunk_text,
                                        "metadata": {
                                            "chunk_type": self._validate_metadata_value("text"),
                                            "chunk_id": str(uuid.uuid4()),
                                            "chunk_index": len(chunks)
                                        }
                                    })
                            else:
                                # Start new chunk with this sentence
                                current_chunk = sentence
                                current_size = len(sentence)
                else:
                    current_chunk = para
                    current_size = len(para)
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append({
                "content": current_chunk,
                "metadata": {
                    "chunk_type": self._validate_metadata_value("text"),
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": len(chunks)
                }
            })
        
        return chunks

class HTMLChunker(BaseChunker):
    """Chunker for HTML content"""
    
    def _validate_metadata_value(self, value):
        """Ensure metadata value is a valid type (str, int, float, bool, None)"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Convert lists to string
        elif isinstance(value, list):
            return ' '.join(str(item) for item in value)
        # Convert other types to string
        else:
            return str(value)
    
    def chunk(self, content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Chunk HTML content by semantic sections while preserving structure"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract title and metadata
        title = soup.title.string if soup.title else ""
        
        # Extract metadata from meta tags
        metadata = {
            "chunk_type": self._validate_metadata_value("html"),
            "title": self._validate_metadata_value(title),
        }
        
        for meta in soup.find_all('meta'):
            if meta.get('name') and meta.get('content'):
                # Ensure metadata values are of valid types (str, int, float, bool, None)
                content = meta['content']
                # Try to convert to appropriate type if possible
                if content.lower() == 'true':
                    metadata[meta['name']] = True
                elif content.lower() == 'false':
                    metadata[meta['name']] = False
                elif content.isdigit():
                    metadata[meta['name']] = int(content)
                elif content.replace('.', '', 1).isdigit() and content.count('.') == 1:
                    metadata[meta['name']] = float(content)
                else:
                    metadata[meta['name']] = self._validate_metadata_value(content)
        
        chunks = []
        
        # Define semantic section tags
        section_tags = ['div', 'section', 'article', 'main', 'header', 'footer', 'nav', 'aside']
        
        # First try to chunk by semantic sections
        sections = []
        for tag in section_tags:
            sections.extend(soup.find_all(tag))
        
        if sections:
            for section in sections:
                section_text = section.get_text(separator=' ', strip=True)
                if not section_text:
                    continue
                    
                # If section is small enough, add it as a chunk
                if len(section_text) <= chunk_size:
                    section_metadata = metadata.copy()
                    section_metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": len(chunks),
                        "section_id": self._validate_metadata_value(section.get('id', '')),
                        "section_class": self._validate_metadata_value(section.get('class', '')),
                    })
                    
                    chunks.append({
                        "content": section_text,
                        "metadata": section_metadata
                    })
                else:
                    # If section is too large, use TextChunker to split it further
                    text_chunker = TextChunker()
                    sub_chunks = text_chunker.chunk(section_text, chunk_size, chunk_overlap)
                    
                    # Add HTML metadata to each sub-chunk
                    for i, sub_chunk in enumerate(sub_chunks):
                        sub_chunk_metadata = metadata.copy()
                        sub_chunk_metadata.update({
                            "chunk_id": str(uuid.uuid4()),
                            "chunk_index": len(chunks) + i,
                            "section_id": self._validate_metadata_value(section.get('id', '')),
                            "section_class": self._validate_metadata_value(section.get('class', '')),
                            "sub_chunk": True,
                            "original_chunk_type": self._validate_metadata_value(sub_chunk["metadata"]["chunk_type"])
                        })
                        
                        chunks.append({
                            "content": sub_chunk["content"],
                            "metadata": sub_chunk_metadata
                        })
        else:
            # If no semantic sections found, fall back to TextChunker
            text_content = soup.get_text(separator='\n', strip=True)
            text_chunker = TextChunker()
            text_chunks = text_chunker.chunk(text_content, chunk_size, chunk_overlap)
            
            # Add HTML metadata to each chunk
            for i, text_chunk in enumerate(text_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "original_chunk_type": self._validate_metadata_value(text_chunk["metadata"]["chunk_type"])
                })
                
                chunks.append({
                    "content": text_chunk["content"],
                    "metadata": chunk_metadata
                })
        
        return chunks

class CodeChunker(BaseChunker):
    """Chunker for source code content"""
    
    def _validate_metadata_value(self, value):
        """Ensure metadata value is a valid type (str, int, float, bool, None)"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Convert lists to string
        elif isinstance(value, list):
            return ' '.join(str(item) for item in value)
        # Convert other types to string
        else:
            return str(value)
    
    def _detect_language(self, content: str) -> str:
        """Detect programming language from content"""
        # Simple language detection based on file patterns
        patterns = {
            r'import\s+[\w\.]+|from\s+[\w\.]+\s+import': 'python',
            r'function\s+[\w]+\s*\(|const\s+[\w]+\s*=|let\s+[\w]+\s*=|var\s+[\w]+\s*=': 'javascript',
            r'public\s+class|private\s+class|protected\s+class': 'java',
            r'#include\s*<|#include\s*"': 'c/c++',
            r'package\s+[\w\.]+;': 'java',
            r'using\s+[\w\.]+;': 'c#',
            r'<!DOCTYPE\s+html|<html': 'html',
            r'<\?php': 'php',
        }
        
        for pattern, language in patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                return language
        
        return 'unknown'
    
    def chunk(self, content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Chunk code by logical sections (functions, classes, etc.)"""
        language = self._detect_language(content)
        
        # Split content into lines
        lines = content.split('\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        in_function = False
        in_class = False
        function_depth = 0
        
        # Track imports/includes separately to include in each chunk
        imports = []
        
        for i, line in enumerate(lines):
            # Detect imports/includes
            if re.match(r'\s*(import|from|#include|using)\s+', line):
                imports.append(line)
                continue
            
            # Check if line starts a function or class
            starts_function = re.match(r'\s*(def|function|public|private|protected)\s+[\w]+\s*\(', line)
            starts_class = re.match(r'\s*(class)\s+[\w]+', line)
            
            # Check for opening/closing braces to track nesting
            opens = line.count('{') - line.count('}')
            
            # Track function/class depth
            if starts_function:
                in_function = True
                function_depth += 1
            elif starts_class:
                in_class = True
                function_depth += 1
            
            function_depth += opens
            
            # If we're at the end of a function/class
            if in_function or in_class:
                if function_depth <= 0 or i == len(lines) - 1:
                    in_function = False
                    in_class = False
                    function_depth = 0
                    
                    # Add the current chunk if not empty
                    if current_chunk:
                        chunk_text = '\n'.join(imports + current_chunk)
                        chunks.append({
                            "content": chunk_text,
                            "metadata": {
                    "chunk_type": self._validate_metadata_value("code"),
                    "language": self._validate_metadata_value(language),
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": len(chunks)
                }
                        })
                        current_chunk = []
                        current_size = 0
            
            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline
            
            # If chunk is too large and we're not in a function/class, split it
            if current_size > chunk_size and not (in_function or in_class):
                chunk_text = '\n'.join(imports + current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "chunk_type": self._validate_metadata_value("code"),
                        "language": self._validate_metadata_value(language),
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": len(chunks)
                    }
                })
                current_chunk = []
                current_size = 0
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = '\n'.join(imports + current_chunk)
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "chunk_type": self._validate_metadata_value("code"),
                    "language": self._validate_metadata_value(language),
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": len(chunks)
                }
            })
        
        return chunks

class JsonChunker(BaseChunker):
    """Chunker for JSON content"""
    
    def _validate_metadata_value(self, value):
        """Ensure metadata value is a valid type (str, int, float, bool, None)"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Convert lists to string
        elif isinstance(value, list):
            return ' '.join(str(item) for item in value)
        # Convert other types to string
        else:
            return str(value)
    
    def chunk(self, content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Split JSON content into chunks based on structure and size"""
        chunks = []
        
        try:
            # Parse JSON content
            json_data = json.loads(content)
            
            # Extract metadata if available
            metadata = {
                "chunk_type": self._validate_metadata_value("json")
            }
            
            # Handle different JSON structures
            if isinstance(json_data, dict):
                # Process dictionary
                self._process_dict(json_data, chunks, metadata, chunk_size)
            elif isinstance(json_data, list):
                # Process list
                self._process_list(json_data, chunks, metadata, chunk_size)
            else:
                # Simple value, just add as a single chunk
                chunks.append({
                    "content": content,
                    "metadata": {
                        "chunk_type": self._validate_metadata_value("json"),
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": 0
                    }
                })
        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to TextChunker
            text_chunker = TextChunker()
            return text_chunker.chunk(content, chunk_size, chunk_overlap)
            
        return chunks
    
    def _process_dict(self, json_dict: Dict, chunks: List[Dict[str, Any]], base_metadata: Dict[str, Any], chunk_size: int):
        """Process a dictionary and create chunks"""
        current_chunk = {}
        current_size = 0
        
        for key, value in json_dict.items():
            # Convert the key-value pair to a string representation
            item_str = json.dumps({key: value})
            item_size = len(item_str)
            
            if current_size + item_size <= chunk_size:
                # Add to current chunk
                current_chunk[key] = value
                current_size += item_size
            else:
                # Current chunk is full, add it to chunks
                if current_chunk:
                    metadata = base_metadata.copy()
                    metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": len(chunks)
                    })
                    
                    chunks.append({
                        "content": json.dumps(current_chunk),
                        "metadata": metadata
                    })
                    
                    # Start a new chunk
                    current_chunk = {key: value}
                    current_size = item_size
                else:
                    # Item is larger than chunk_size, add it as a separate chunk
                    metadata = base_metadata.copy()
                    metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": len(chunks)
                    })
                    
                    chunks.append({
                        "content": item_str,
                        "metadata": metadata
                    })
        
        # Add the last chunk if not empty
        if current_chunk:
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": len(chunks)
            })
            
            chunks.append({
                "content": json.dumps(current_chunk),
                "metadata": metadata
            })
    
    def _process_list(self, json_list: List, chunks: List[Dict[str, Any]], base_metadata: Dict[str, Any], chunk_size: int):
        """Process a list and create chunks"""
        current_chunk = []
        current_size = 0
        
        for item in json_list:
            # Convert the item to a string representation
            item_str = json.dumps(item)
            item_size = len(item_str)
            
            if current_size + item_size <= chunk_size:
                # Add to current chunk
                current_chunk.append(item)
                current_size += item_size
            else:
                # Current chunk is full, add it to chunks
                if current_chunk:
                    metadata = base_metadata.copy()
                    metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": len(chunks)
                    })
                    
                    chunks.append({
                        "content": json.dumps(current_chunk),
                        "metadata": metadata
                    })
                    
                    # Start a new chunk
                    current_chunk = [item]
                    current_size = item_size
                else:
                    # Item is larger than chunk_size, add it as a separate chunk
                    metadata = base_metadata.copy()
                    metadata.update({
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": len(chunks)
                    })
                    
                    chunks.append({
                        "content": item_str,
                        "metadata": metadata
                    })
        
        # Add the last chunk if not empty
        if current_chunk:
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": len(chunks)
            })
            
            chunks.append({
                "content": json.dumps(current_chunk),
                "metadata": metadata
            })

class SmartChunker:
    """Smart chunker that detects content type and uses appropriate chunker"""
    
    def _validate_metadata_value(self, value):
        """Ensure metadata value is a valid type (str, int, float, bool, None)"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Convert lists to string
        elif isinstance(value, list):
            return ' '.join(str(item) for item in value)
        # Convert other types to string
        else:
            return str(value)
    
    def detect_content_type(self, content: str) -> str:
        """Detect content type from the content"""
        # Check if content is HTML
        if re.search(r'<!DOCTYPE\s+html|<html|<body|<div|<p>|<head>', content, re.IGNORECASE):
            return "html"
        
        # Check if content is JSON
        # Trim whitespace at the beginning and end
        trimmed_content = content.strip()
        # Check if content starts with { or [ and ends with } or ]
        if (trimmed_content.startswith('{') and trimmed_content.endswith('}')) or \
           (trimmed_content.startswith('[') and trimmed_content.endswith(']')):
            try:
                # Try to parse as JSON
                json.loads(trimmed_content)
                return "json"
            except json.JSONDecodeError:
                # Not valid JSON
                pass
        
        # Check if content is code
        code_patterns = [
            r'import\s+[\w\.]+|from\s+[\w\.]+\s+import',  # Python
            r'function\s+[\w]+\s*\(|const\s+[\w]+\s*=|let\s+[\w]+\s*=',  # JavaScript
            r'public\s+class|private\s+class|protected\s+class',  # Java
            r'#include\s*<|#include\s*"',  # C/C++
            r'package\s+[\w\.]+;',  # Java
            r'using\s+[\w\.]+;',  # C#
            r'<\?php',  # PHP
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return "code"
        
        # Default to text
        return "text"
    
    def chunk(self, content: str, content_type: Optional[str] = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Chunk content based on detected or specified type"""
        # Detect content type if not specified
        if not content_type:
            content_type = self.detect_content_type(content)
        
        # Use appropriate chunker
        if content_type == "html":
            chunker = HTMLChunker()
        elif content_type == "code":
            chunker = CodeChunker()
        elif content_type == "json":
            chunker = JsonChunker()
        else:  # Default to text
            chunker = TextChunker()
        
        return chunker.chunk(content, chunk_size, chunk_overlap)