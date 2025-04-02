# Text Summarization API on Google Colab - PyTorch Version
# Import required libraries
import os
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
from typing import List, Optional
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pydantic import BaseModel, HttpUrl, Field
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from heapq import nlargest
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import io

# Download necessary NLTK data
try:
    nltk.download('stopwords')
except:
    print("Error downloading stopwords, but continuing...")

# Define the Pydantic models
class StyleInfo(BaseModel):
    """
    Information about a summarization style
    """
    name: str
    description: str

class TextInput(BaseModel):
    """
    Model for text input requests
    """
    text: str = Field(..., description="Text content to summarize")
    max_length: int = Field(150, description="Maximum length of the generated summary")
    min_length: int = Field(30, description="Minimum length of the generated summary")
    style: str = Field("default", description="Summarization style to use")

class UrlInput(BaseModel):
    """
    Model for URL input requests
    """
    url: HttpUrl = Field(..., description="Web URL to fetch and summarize")
    max_length: int = Field(150, description="Maximum length of the generated summary")
    min_length: int = Field(30, description="Minimum length of the generated summary")
    style: str = Field("default", description="Summarization style to use")

class SummaryResponse(BaseModel):
    """
    Model for summarization response
    """
    summary: str
    original_length: int
    summary_length: int
    style: str
    style_description: str

class StylesResponse(BaseModel):
    """
    Model for available styles response
    """
    styles: List[StyleInfo]

class TranslationInput(BaseModel):
    """
    Model for translation requests
    """
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code")

# Utility functions
def extract_text_from_url(url):
    """
    Extract text content from a webpage

    Args:
        url (str): The URL to fetch and extract text from

    Returns:
        str: Extracted text content
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text from paragraphs, headings, and other relevant tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'article'])

        # Extract text and join with spaces
        text = ' '.join([elem.get_text(strip=True) for elem in text_elements])

        # Clean up whitespace
        text = ' '.join(text.split())

        return text
    except Exception as e:
        raise Exception(f"Failed to extract text from URL: {str(e)}")

def extract_text_from_pdf(file_content):
    """
    Extract text content from a PDF file

    Args:
        file_content (bytes): The PDF file content

    Returns:
        str: Extracted text content
    """
    try:
        pdf_file = fitz.open(stream=io.BytesIO(file_content), filetype="pdf")
        text = ""

        for page_num in range(len(pdf_file)):
            page = pdf_file[page_num]
            text += page.get_text()

        # Clean up whitespace
        text = ' '.join(text.split())

        return text
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

# Enhanced PyTorch Summarizer for Abstractive Summarization
class EnhancedPTSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the summarization model with PyTorch backend

        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.model_name = model_name
        print(f"Loading model: {model_name}")

        # Set device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and tokenizer with PyTorch backend
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        # For faster inference set eval mode
        self.model.eval()

        # Define summarization styles
        self.styles = {
            "default": {
                "description": "Balanced summary with key information",
                "params": {
                    "num_beams": 4,
                    "no_repeat_ngram_size": 3,
                    "length_penalty": 1.0,
                    "early_stopping": True
                }
            },
            "concise": {
                "description": "Very brief summary focusing only on the most critical points",
                "params": {
                    "num_beams": 5,
                    "no_repeat_ngram_size": 3,
                    "length_penalty": 0.6,  # Prefer shorter outputs
                    "early_stopping": True
                }
            },
            "detailed": {
                "description": "Comprehensive summary covering more information",
                "params": {
                    "num_beams": 5,
                    "no_repeat_ngram_size": 2,
                    "length_penalty": 2.0,  # Increased to favor longer outputs
                    "early_stopping": False,
                    "min_length_factor": 0.2,  # 20% of original text length
                    "max_length_factor": 0.4   # 40% of original text length
                }
            },
            "very_detailed": {
                "description": "Highly comprehensive summary with extensive details",
                "params": {
                    "num_beams": 6,
                    "no_repeat_ngram_size": 2,
                    "length_penalty": 3.0,     # Strong preference for longer outputs
                    "temperature": 0.7,        # Slightly more creative generation
                    "early_stopping": False,
                    "min_length_factor": 0.3,  # 30% of original text length
                    "max_length_factor": 0.5   # 50% of original text length
                }
            },
            "aggressive": {
                "description": "Highly abstractive summary that condenses information significantly",
                "params": {
                    "num_beams": 6,
                    "no_repeat_ngram_size": 4,
                    "length_penalty": 0.4,  # Strongly prefer shorter outputs
                    "early_stopping": True
                }
            },
            "creative": {
                "description": "More paraphrased and creatively reworded summary",
                "params": {
                    "num_beams": 5,
                    "temperature": 1.2,  # More diversity in generation
                    "top_k": 50,
                    "top_p": 0.9,
                    "no_repeat_ngram_size": 2,
                    "length_penalty": 1.0,
                    "early_stopping": True
                }
            },
            "bullets": {
                "description": "Summary formatted as bullet points",
                "params": {
                    "num_beams": 4,
                    "no_repeat_ngram_size": 3,
                    "length_penalty": 1.0,
                    "early_stopping": True,
                    "prefix": "Key points:\n• ",
                    "format_bullets": True
                }
            },
            "eli5": {
                "description": "Explain Like I'm 5 - Summary in simple language",
                "params": {
                    "num_beams": 4,
                    "no_repeat_ngram_size": 2,
                    "length_penalty": 1.0,
                    "prefix": "In simple terms: ",
                    "early_stopping": True
                }
            },
            "academic": {
                "description": "Formal academic style summary",
                "params": {
                    "num_beams": 5,
                    "no_repeat_ngram_size": 2,
                    "length_penalty": 1.2,
                    "early_stopping": True
                }
            }
        }

        # Warm up model with a sample if on GPU
        if torch.cuda.is_available():
            print("Warming up model on GPU...")
            warm_up_text = "This is a warm-up text to initialize the model on GPU."
            self.summarize(warm_up_text, max_length=50, min_length=10)
            print("Model warm-up complete")

    def get_available_styles(self):
        """
        Get all available summarization styles

        Returns:
            dict: Dictionary of style names and descriptions
        """
        return {name: style["description"] for name, style in self.styles.items()}

    def summarize(self, text, max_length=150, min_length=30, style="default"):
        """
        Summarize the provided text using the specified style

        Args:
            text (str): The text to summarize
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            style (str): Summarization style to use

        Returns:
            dict: Summary information
        """
        start_time = time.time()

        # Get style configuration
        if style not in self.styles:
            print(f"Style '{style}' not found, using default style")
            style = "default"

        style_config = self.styles[style]
        style_params = style_config["params"].copy()

        # Calculate dynamic lengths based on input size if factors are provided
        text_length = len(text.split())
        if "min_length_factor" in style_params:
            min_length_factor = style_params.pop("min_length_factor")
            min_length = max(min_length, int(text_length * min_length_factor))

        if "max_length_factor" in style_params:
            max_length_factor = style_params.pop("max_length_factor")
            max_length = max(max_length, int(text_length * max_length_factor))

        # Ensure max_length is at least min_length
        max_length = max(max_length, min_length)

        # Extract special parameters
        prefix = style_params.pop("prefix", "")
        format_bullets = style_params.pop("format_bullets", False)

        # For long texts, we need to chunk them
        if len(text.split()) > 1024:  # Most models have a limit of ~1024 tokens
            text = self._truncate_text(text, 1024)

        # Tokenize the input
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate summary with style-specific parameters
        with torch.no_grad():  # Disable gradient calculation for inference
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                **style_params
            )

        # Decode the generated tokens
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Apply post-processing based on style
        if prefix and not summary.startswith(prefix):
            summary = prefix + summary

        if format_bullets:
            summary = self._format_as_bullets(summary)

        if style == "academic" and not any(word in summary.lower() for word in ["research", "study", "analysis", "therefore", "consequently"]):
            # Add some academic flair if it's not already present
            if "." in summary:
                parts = summary.split(".")
                parts[-2] = parts[-2] + ", therefore"
                summary = ".".join(parts)

        # Print performance statistics
        end_time = time.time()
        print(f"Summarization completed in {end_time - start_time:.2f} seconds")

        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "style": style,
            "style_description": style_config["description"]
        }

    def summarize_long_document(self, text, max_length=300, min_length=100, style="detailed"):
        """
        Summarize a long document by breaking it into segments, summarizing each,
        and then combining and summarizing the results.

        Args:
            text (str): The text to summarize
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            style (str): Summarization style to use

        Returns:
            dict: Summary information
        """
        # Split document into segments (e.g., paragraphs or sections)
        segments = self._split_into_segments(text)

        # Summarize each segment
        segment_summaries = []
        for segment in segments:
            if len(segment.split()) > 50:  # Only summarize substantial segments
                summary = self.summarize(segment, max_length=150, min_length=30, style=style)
                segment_summaries.append(summary["summary"])

        # Combine segment summaries
        combined_summary = " ".join(segment_summaries)

        # Create a meta-summary of the combined summaries
        if len(combined_summary.split()) > max_length:
            final_summary = self.summarize(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                style="default"  # Use default style for final summary
            )
            return final_summary
        else:
            return {
                "summary": combined_summary,
                "original_length": len(text),
                "summary_length": len(combined_summary),
                "style": style,
                "style_description": self.styles[style]["description"]
            }

    def _truncate_text(self, text, max_tokens):
        """
        Truncate text to max_tokens (approximate implementation)
        """
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])

    def _split_into_segments(self, text, max_segment_tokens=800):
        """
        Split text into meaningful segments (paragraphs or sections)
        """
        # Simple paragraph-based splitting
        paragraphs = text.split('\n\n')

        segments = []
        current_segment = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para.split())

            if current_length + para_length <= max_segment_tokens:
                current_segment.append(para)
                current_length += para_length
            else:
                if current_segment:
                    segments.append(' '.join(current_segment))
                current_segment = [para]
                current_length = para_length

        # Add the last segment
        if current_segment:
            segments.append(' '.join(current_segment))

        return segments

    def _format_as_bullets(self, text):
        """
        Format text as bullet points
        """
        if "• " not in text:
            # If the model didn't generate bullet points, create them
            sentences = text.split('. ')
            if len(sentences) <= 1:
                return text

            # Remove any existing prefix
            if sentences[0].startswith("Key points:"):
                sentences.pop(0)

            # Format as bullet points
            bullet_text = "Key points:\n"
            for sentence in sentences:
                if sentence and not sentence.isspace():
                    # Clean up the sentence
                    sentence = sentence.strip()
                    if not sentence.endswith('.'):
                        sentence += '.'
                    bullet_text += f"• {sentence}\n"

            return bullet_text.strip()
        else:
            # The model already generated bullet points
            return text

# Extractive Summarizer with improved sentence tokenization
class ExtractiveTextSummarizer:
    def __init__(self):
        """
        Initialize the extractive summarization class
        """
        # Try to get stopwords from NLTK if available
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback common English stopwords
            self.stop_words = set([
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
            ])

        self.styles = {
            "tfidf_basic": {
                "description": "Basic TF-IDF extractive summary highlighting key sentences",
                "params": {
                    "ratio": 0.3  # Extract 30% of sentences by default
                }
            },
            "tfidf_short": {
                "description": "Very concise TF-IDF summary with only the most critical sentences",
                "params": {
                    "ratio": 0.15  # Extract 15% of sentences
                }
            },
            "tfidf_detailed": {
                "description": "Comprehensive TF-IDF extractive summary with more context",
                "params": {
                    "ratio": 0.4  # Extract 40% of sentences
                }
            },
            "textrank": {
                "description": "Graph-based extractive summary using TextRank algorithm",
                "params": {
                    "ratio": 0.3,
                    "algorithm": "textrank"
                }
            },
            "centroid": {
                "description": "Centroid-based extractive summary focusing on central concepts",
                "params": {
                    "ratio": 0.3,
                    "algorithm": "centroid"
                }
            }
        }

    def get_available_styles(self):
        """
        Get all available summarization styles

        Returns:
            dict: Dictionary of style names and descriptions
        """
        return {name: style["description"] for name, style in self.styles.items()}

    def _clean_text(self, text):
        """
        Clean the text by removing special characters, numbers, etc.

        Args:
            text (str): The text to clean

        Returns:
            str: Cleaned text
        """
        # Only remove special brackets but keep periods and other punctuation
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        # Remove duplicate spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def summarize(self, text, max_length=150, min_length=30, style="tfidf_basic"):
        """
        Summarize the provided text using the specified extractive style

        Args:
            text (str): The text to summarize
            max_length (int): Maximum summary length (used as guidance)
            min_length (int): Minimum summary length (used as guidance)
            style (str): Summarization style to use

        Returns:
            dict: Summary information
        """
        import time
        start_time = time.time()

        # Get style configuration
        if style not in self.styles:
            print(f"Style '{style}' not found, using tfidf_basic style")
            style = "tfidf_basic"

        style_config = self.styles[style]
        style_params = style_config["params"].copy()

        # Clean the text
        cleaned_text = self._clean_text(text)

        # Custom sentence tokenizer
        import re
        # Split on periods, exclamation points, or question marks followed by a space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]  # Only keep non-empty sentences

        print(f"Number of sentences: {len(sentences)}")

        # Check if we have enough sentences
        if len(sentences) <= 3:
            return {
                "summary": text,
                "original_length": len(text),
                "summary_length": len(text),
                "style": style,
                "style_description": style_config["description"]
            }

        # Choose the algorithm based on style
        algorithm = style_params.get("algorithm", "tfidf")
        ratio = style_params.get("ratio", 0.3)

        # Calculate the number of sentences to include
        num_sentences = max(3, min(int(len(sentences) * ratio), 10))
        print(f"Selecting top {num_sentences} sentences")

        try:
            # Choose algorithm based on style
            if algorithm == "textrank":
                summary_sentences = self._textrank_summarize(sentences, num_sentences)
            elif algorithm == "centroid":
                summary_sentences = self._centroid_summarize(sentences, num_sentences)
            else:  # Default to TF-IDF
                summary_sentences = self._tfidf_summarize(sentences, num_sentences)

            # Join sentences to create the final summary
            summary = " ".join(summary_sentences)

        except Exception as e:
            print(f"Error in summarization algorithm: {e}")
            # Fallback to a simple position-based summary
            summary_sentences = []

            # Take first sentence
            if len(sentences) > 0:
                summary_sentences.append(sentences[0])

            # Take some sentences from the middle
            if len(sentences) > 3:
                mid_point = len(sentences) // 2
                summary_sentences.append(sentences[mid_point])

            # Take the last sentence if available
            if len(sentences) >= 3:
                summary_sentences.append(sentences[-1])

            summary = " ".join(summary_sentences)

        # Print performance statistics
        end_time = time.time()
        print(f"Extractive summarization completed in {end_time - start_time:.2f} seconds")

        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "style": style,
            "style_description": style_config["description"]
        }

    def _tfidf_summarize(self, sentences, num_sentences):
        """
        Summarize using TF-IDF approach

        Args:
            sentences (list): List of sentences
            num_sentences (int): Number of sentences to include in summary

        Returns:
            list: List of summary sentences
        """
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        # Check if we have enough sentences
        if len(sentences) < 2:
            return sentences

        # Calculate TF-IDF matrix
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calculate sentence scores based on the sum of TF-IDF values
            sentence_scores = []
            for i in range(len(sentences)):
                score = sum(tfidf_matrix[i].toarray()[0])
                sentence_scores.append((i, score))

            # Get top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
            # Sort by original position
            top_sentences = sorted(top_sentences, key=lambda x: x[0])

            # Extract selected sentences in order
            summary_sentences = [sentences[i] for i, _ in top_sentences]

            return summary_sentences

        except Exception as e:
            print(f"Error in TF-IDF: {e}")
            # Return a subset of sentences if TF-IDF fails
            return sentences[:min(num_sentences, len(sentences))]

    def _textrank_summarize(self, sentences, num_sentences):
        """
        Summarize using a simplified TextRank algorithm

        Args:
            sentences (list): List of sentences
            num_sentences (int): Number of sentences to include in summary

        Returns:
            list: List of summary sentences
        """
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        # Check if we have enough sentences
        if len(sentences) < 2:
            return sentences

        try:
            # Calculate TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calculate sentence similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Initialize scores
            scores = np.ones(len(sentences))

            # Apply TextRank algorithm (simplified)
            damping = 0.85
            iterations = 10

            for _ in range(iterations):
                new_scores = np.ones(len(sentences)) * (1 - damping)
                for i in range(len(sentences)):
                    for j in range(len(sentences)):
                        if i != j and similarity_matrix[j, i] > 0:
                            # Avoid division by zero
                            total_similarity = max(np.sum(similarity_matrix[j]), 0.0001)
                            new_scores[i] += damping * scores[j] * similarity_matrix[j, i] / total_similarity
                scores = new_scores

            # Get sentence indices with top scores
            top_indices = np.argsort(scores)[::-1][:num_sentences]
            top_indices = sorted(top_indices)  # Sort to maintain original order

            # Get top sentences
            summary_sentences = [sentences[i] for i in top_indices]

            return summary_sentences

        except Exception as e:
            print(f"Error in TextRank: {e}")
            # Fallback to TF-IDF if TextRank fails
            return self._tfidf_summarize(sentences, num_sentences)

    def _centroid_summarize(self, sentences, num_sentences):
        """
        Summarize using centroid-based approach

        Args:
            sentences (list): List of sentences
            num_sentences (int): Number of sentences to include in summary

        Returns:
            list: List of summary sentences
        """
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        # Check if we have enough sentences
        if len(sentences) < 2:
            return sentences

        try:
            # Calculate TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calculate centroid vector (average of all sentence vectors)
            centroid = np.mean(tfidf_matrix.toarray(), axis=0)

            # Calculate similarity to centroid
            sentence_scores = []
            for i in range(len(sentences)):
                # Calculate cosine similarity between sentence and centroid
                similarity = cosine_similarity(
                    tfidf_matrix[i],
                    centroid.reshape(1, -1)
                )[0][0]
                sentence_scores.append((i, similarity))

            # Get top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
            # Sort by original position
            top_sentences = sorted(top_sentences, key=lambda x: x[0])

            # Extract selected sentences in order
            summary_sentences = [sentences[i] for i, _ in top_sentences]

            return summary_sentences

        except Exception as e:
            print(f"Error in Centroid: {e}")
            # Fallback to TF-IDF if Centroid fails
            return self._tfidf_summarize(sentences, num_sentences)

# Set your Ngrok auth token here
NGROK_AUTH_TOKEN = "2uy7TCidg0hdlvTTM9pNwpI1B1t_6ka8hQJoimon27752ZfEF"  # Replace with your actual token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Apply nest_asyncio for running FastAPI in Colab
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Text Summarization API",
    description="API for summarizing text from various sources using multiple styles",
    version="1.0.0"
)

# Initialize both summarizers
model_name = os.environ.get("MODEL_NAME", "facebook/bart-large-cnn")
abstractive_summarizer = EnhancedPTSummarizer(model_name=model_name)
extractive_summarizer = ExtractiveTextSummarizer()

# Simple Translator class as replacement for googletrans
# Simple Translator class as replacement for googletrans
class SimpleTranslator:
    def translate(self, text, dest):
        # This is a dummy implementation since we're focusing on the summarization
        # For a real implementation, you might want to use a PyTorch translation model
        return type('obj', (object,), {
            'text': f"[Translated to {dest}]: {text}",
            'src': "en"
        })

translator = SimpleTranslator()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    """
    Root endpoint - returns basic API information
    """
    return {
        "message": "Welcome to the Enhanced Text Summarization API (PyTorch Version)",
        "model": abstractive_summarizer.model_name,
        "endpoints": {
            "GET /styles/{summarization_type}": "Get available summarization styles for a type",
            "GET /summarization_types": "Get available summarization types",
            "POST /summarize/text": "Summarize plain text",
            "POST /summarize/url": "Summarize content from URL",
            "POST /summarize/pdf": "Summarize content from PDF",
            "POST /translate": "Translate text to another language"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "model": abstractive_summarizer.model_name, "device": str(abstractive_summarizer.device)}

@app.get("/summarization_types")
async def get_summarization_types():
    """
    Get available summarization types (abstractive and extractive)
    """
    return {
        "types": [
            {
                "name": "abstractive",
                "description": "AI-generated summary that paraphrases the content"
            },
            {
                "name": "extractive",
                "description": "Selects and combines the most important sentences from the original text"
            }
        ]
    }

@app.get("/styles/{summarization_type}", response_model=StylesResponse)
async def get_styles_for_type(summarization_type: str):
    """
    Get available summarization styles for the specified type
    """
    if summarization_type == "abstractive":
        styles_dict = abstractive_summarizer.get_available_styles()
    elif summarization_type == "extractive":
        styles_dict = extractive_summarizer.get_available_styles()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown summarization type: {summarization_type}")

    styles_list = [StyleInfo(name=name, description=desc) for name, desc in styles_dict.items()]
    return {"styles": styles_list}

@app.post("/summarize/text", response_model=SummaryResponse)
async def summarize_text(input_data: TextInput):
    """
    Summarize plain text input with specified style
    """
    try:
        # Extract summarization type from style (default to abstractive)
        summarization_type = "abstractive"
        style = input_data.style

        # Check if the style specifies a summarization type
        if ":" in input_data.style:
            parts = input_data.style.split(":", 1)
            summarization_type = parts[0]
            style = parts[1]

        # Choose the appropriate summarizer
        if summarization_type == "extractive":
            # Use extractive summarizer
            result = extractive_summarizer.summarize(
                input_data.text,
                max_length=input_data.max_length,
                min_length=input_data.min_length,
                style=style
            )
        else:
            # Use abstractive summarizer (default)
            # Check if this is a long document that needs hierarchical summarization
            if len(input_data.text.split()) > 1000 and style in ["detailed", "very_detailed"]:
                result = abstractive_summarizer.summarize_long_document(
                    input_data.text,
                    max_length=input_data.max_length,
                    min_length=input_data.min_length,
                    style=style
                )
            else:
                result = abstractive_summarizer.summarize(
                    input_data.text,
                    max_length=input_data.max_length,
                    min_length=input_data.min_length,
                    style=style
                )

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/url", response_model=SummaryResponse)
async def summarize_webpage(input_data: UrlInput):
    """
    Fetch a webpage and summarize its content with specified style
    """
    try:
        # Extract text from webpage
        text = extract_text_from_url(str(input_data.url))

        if not text:
            raise HTTPException(status_code=422, detail="Could not extract text from the URL")

        # Extract summarization type from style (default to abstractive)
        summarization_type = "abstractive"
        style = input_data.style

        # Check if the style specifies a summarization type
        if ":" in input_data.style:
            parts = input_data.style.split(":", 1)
            summarization_type = parts[0]
            style = parts[1]

        # Choose the appropriate summarizer
        if summarization_type == "extractive":
            # Use extractive summarizer
            result = extractive_summarizer.summarize(
                text,
                max_length=input_data.max_length,
                min_length=input_data.min_length,
                style=style
            )
        else:
            # Use abstractive summarizer (default)
            # Check if this is a long document that needs hierarchical summarization
            if len(text.split()) > 1000 and style in ["detailed", "very_detailed"]:
                result = abstractive_summarizer.summarize_long_document(
                    text,
                    max_length=input_data.max_length,
                    min_length=input_data.min_length,
                    style=style
                )
            else:
                result = abstractive_summarizer.summarize(
                    text,
                    max_length=input_data.max_length,
                    min_length=input_data.min_length,
                    style=style
                )

        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/pdf", response_model=SummaryResponse)
async def summarize_pdf(
    file: UploadFile = File(...),
    max_length: int = Form(150),
    min_length: int = Form(30),
    style: str = Form("default")
):
    """
    Summarize content from a PDF file with specified style
    """
    try:
        # Validate file mimetype
        if not file.content_type or "pdf" not in file.content_type.lower():
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")

        # Read file content
        file_content = await file.read()

        # Extract text from PDF
        text = extract_text_from_pdf(file_content)

        if not text:
            raise HTTPException(status_code=422, detail="Could not extract text from the PDF")

        # Extract summarization type from style (default to abstractive)
        summarization_type = "abstractive"
        style_name = style

        # Check if the style specifies a summarization type
        if ":" in style:
            parts = style.split(":", 1)
            summarization_type = parts[0]
            style_name = parts[1]

        # Choose the appropriate summarizer
        if summarization_type == "extractive":
            # Use extractive summarizer
            result = extractive_summarizer.summarize(
                text,
                max_length=max_length,
                min_length=min_length,
                style=style_name
            )
        else:
            # Use abstractive summarizer (default)
            # Check if this is a long document that needs hierarchical summarization
            if len(text.split()) > 1000 and style_name in ["detailed", "very_detailed"]:
                result = abstractive_summarizer.summarize_long_document(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    style=style_name
                )
            else:
                result = abstractive_summarizer.summarize(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    style=style_name
                )

        return result
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_text(data: dict):
    """
    Translate text to the target language
    """
    try:
        text = data.get("text", "")
        target_language = data.get("target_language", "en")

        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Don't translate if already in target language
        if target_language == "en":
            return {
                "translated_text": text,
                "source_language": "en",
                "target_language": target_language
            }

        # Translate the text
        translated = translator.translate(text, dest=target_language)

        return {
            "translated_text": translated.text,
            "source_language": translated.src,
            "target_language": target_language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler
    """
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"},
    )

# Function to check GPU memory usage
def check_gpu_memory():
    if torch.cuda.is_available():
        try:
            print("\nGPU Memory usage:")
            !nvidia-smi
        except:
            print("Could not run nvidia-smi")
    else:
        print("\nNo GPU available")

# Start the application
def main():
    # Check GPU memory before starting
    check_gpu_memory()

    # Set up the Ngrok connection
    public_url = ngrok.connect(8001)
    print(f"\n * Running on {public_url}")
    print(" * The API is now publicly accessible")

    # Display usage instructions
    print("\nAPI Endpoints:")
    print("- GET /styles/{summarization_type} - List styles for a specific summarization type")
    print("- GET /summarization_types - List available summarization types")
    print("- POST /summarize/text - Summarize plain text")
    print("- POST /summarize/url - Summarize content from URL")
    print("- POST /summarize/pdf - Summarize content from PDF")
    print("- POST /translate - Translate text")
    print("\nExample API usage with curl:")
    print(f'curl -X POST "{public_url}/summarize/text" -H "Content-Type: application/json" -d \'{{"text": "Your text to summarize", "max_length": 150, "min_length": 30, "style": "abstractive:default"}}\'')

    # Start the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
