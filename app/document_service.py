"""
Document processing service for medical transcripts/reports.
Handles PDF and image uploads, extracts text, finds medical entities.
"""

import logging
from typing import Dict, List, Optional
import PyPDF2
from PIL import Image
import pytesseract
import io
import os

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """Service for processing medical documents."""
    
    def __init__(self):
        """Initialize document processor."""
        # Set Tesseract path for Windows (if needed)
        if os.name == 'nt':  # Windows
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Tesseract path set: {tesseract_path}")
            else:
                logger.warning("Tesseract not found at default Windows location. OCR may not work.")
        
        logger.info("Document processing service initialized")
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_bytes: PDF file as bytes
            
        Returns:
            Extracted text
        """
        try:
            pdf_file = io.BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(page_text)
                        logger.info(f"Extracted text from page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from {len(pdf_reader.pages)} pages")
            
            if not full_text.strip():
                raise ValueError("No text could be extracted from PDF. It may be a scanned image.")
            
            return full_text
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_text_from_image(self, file_bytes: bytes) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            file_bytes: Image file as bytes
            
        Returns:
            Extracted text
        """
        try:
            image = Image.open(io.BytesIO(file_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use pytesseract for OCR
            logger.info(f"Running OCR on image size: {image.size}")
            text = pytesseract.image_to_string(image, lang='eng')
            
            logger.info(f"Extracted {len(text)} characters from image")
            
            if not text.strip():
                raise ValueError("No text could be extracted from image. Image may be blank or text may be unclear.")
            
            return text
            
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract OCR not found. Please install Tesseract.")
            raise ValueError(
                "OCR engine (Tesseract) not installed. "
                "Please install from: https://github.com/UB-Mannheim/tesseract/wiki"
            )
        except Exception as e:
            logger.error(f"Image OCR error: {e}")
            raise ValueError(f"Failed to extract text from image: {str(e)}")
    
    def process_medical_document(
        self,
        file_bytes: bytes,
        file_type: str,
        filename: str
    ) -> Dict[str, any]:
        """
        Process medical document and extract structured information.
        
        Args:
            file_bytes: File content as bytes
            file_type: Type of file (pdf, jpg, png)
            filename: Original filename
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logger.info(f"Processing document: {filename} (type: {file_type})")
        
        # Extract text based on file type
        if file_type == "pdf":
            text = self.extract_text_from_pdf(file_bytes)
        elif file_type in ["jpg", "jpeg", "png"]:
            text = self.extract_text_from_image(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Clean extracted text
        text = self._clean_text(text)
        
        return {
            "extracted_text": text,
            "text_length": len(text),
            "file_type": file_type,
            "filename": filename
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters that might cause issues
        # But keep medical symbols and basic punctuation
        
        return text.strip()


# Singleton instance
_document_service = None

def get_document_service() -> DocumentProcessingService:
    """Get or create document service singleton."""
    global _document_service
    if _document_service is None:
        _document_service = DocumentProcessingService()
    return _document_service