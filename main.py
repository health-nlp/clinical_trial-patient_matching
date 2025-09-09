import os
import json
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import time
import re
from dateutil.parser import parse as parse_date
from datetime import datetime, timedelta

# langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain.docstore.document import Document
import numpy as np

import gc 

# config
try:
    from Config import OPENAI_API_KEY as CFG_KEY  # Prefer env var; fallback to Config
except Exception:
    CFG_KEY = None

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
ENV_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_KEY = (ENV_KEY or (CFG_KEY or "")).strip()
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set. Export OPENAI_API_KEY or define it in Config.OPENAI_API_KEY.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global cache for criteria embeddings 
_criteria_embedding_cache = {}

def get_or_create_criteria_embedding(criterion: str, force_refresh: bool = False):
    """
    Get embedding for a criterion from cache, or compute and cache it if not present.
    
    Args:
        criterion: The eligibility criterion text
        force_refresh: Whether to force recomputation even if cached
        
    Returns:
        numpy array of the embedding vector
    """
    if not force_refresh and criterion in _criteria_embedding_cache:
        logging.info(f"CRITERIA_EMBEDDING_CACHE_HIT | Using cached embedding for: '{criterion[:50]}...'")
        return _criteria_embedding_cache[criterion]
    
    logging.info(f"CRITERIA_EMBEDDING_CACHE_MISS | Computing embedding for: '{criterion[:50]}...'")
    embedding = embeddings.embed_query(criterion)
    _criteria_embedding_cache[criterion] = embedding
    
    return embedding


##currently not used
# (removed unused criteria precompute and cache stats utilities)

# Set up logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console by default
    ]
)

# Set up custom callback handler for LangChain
class DetailedCallbackHandler(BaseCallbackHandler):
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logger.info(f"LLM_START | Model: {serialized.get('name', 'unknown')} | Prompts: {len(prompts)}")
    
    def on_llm_end(self, response, **kwargs):
        token_usage = getattr(response, 'llm_output', {}).get('token_usage', {})
        self.logger.info(f"LLM_END | Tokens: {token_usage}")
    
    def on_llm_error(self, error, **kwargs):
        self.logger.error(f"LLM_ERROR | {error}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        chain_type = serialized.get("name", "unknown")
        self.logger.info(f"CHAIN_START | Type: {chain_type}")
    
    def on_chain_end(self, outputs, **kwargs):
        self.logger.info(f"CHAIN_END | Output keys: {list(outputs.keys())}")
    
    def on_chain_error(self, error, **kwargs):
        self.logger.error(f"CHAIN_ERROR | {error}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.logger.info(f"TOOL_START | Tool: {tool_name} | Input: {input_str[:100]}...")
    
    def on_tool_end(self, output, **kwargs):
        self.logger.info(f"TOOL_END | Output: {str(output)[:100]}...")
    
    def on_tool_error(self, error, **kwargs):
        self.logger.error(f"TOOL_ERROR | {error}")
    
    def on_text(self, text, **kwargs):
        self.logger.info(f"TEXT | {text}")
    
    def on_retriever_start(self, serialized, query, **kwargs):
        self.logger.info(f"RETRIEVER_START | Query: {query}")
    
    def on_retriever_end(self, documents, **kwargs):
        self.logger.info(f"RETRIEVER_END | Retrieved {len(documents)} documents")

# Initialize callback handler
callback_handler = DetailedCallbackHandler()


CURRENT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        logging.info(f"Initializing ChatOpenAI with model={CURRENT_MODEL}")
        _llm = ChatOpenAI(
            temperature=0,
            model=CURRENT_MODEL,
            api_key=OPENAI_API_KEY,
            callbacks=[callback_handler]
        )
    return _llm

def set_model(model_name: str):
    """Set model and recreate LLM instance."""
    global CURRENT_MODEL, _llm
    CURRENT_MODEL = model_name
    _llm = None
    get_llm()

# Use the same API key for embeddings
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY
)

# Function to log embedding details
def log_embedding_details(text, embedding):
    """Log details about an embedding vector"""
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    logging.info(f"EMBEDDING | Text length: {len(text)} chars | Dimensions: {embedding.shape}")
    logging.info(f"EMBEDDING | Norm: {np.linalg.norm(embedding):.4f} | Mean: {np.mean(embedding):.4f} | Std: {np.std(embedding):.4f}")
    logging.info(f"EMBEDDING | Min: {np.min(embedding):.4f} | Max: {np.max(embedding):.4f}")


def log_similarity_scores(query, documents, scores=None):
    """Log details about similarity search results"""
    logging.info(f"SIMILARITY_SEARCH | Query: '{query[:50]}...' | Found {len(documents)} documents")
    
    if scores:
        logging.info(f"SIMILARITY_SCORES | Min: {min(scores):.4f} | Max: {max(scores):.4f} | Mean: {sum(scores)/len(scores):.4f}")
        
        # Log each document with its score
        for i, (doc, score) in enumerate(zip(documents, scores)):
            logging.info(f"DOCUMENT_{i} | Score: {score:.4f} | Content: '{doc.page_content[:50]}...'")
    else:
        # Just log documents without scores
        for i, doc in enumerate(documents):
            logging.info(f"DOCUMENT_{i} | Content: '{doc.page_content[:50]}...'")



def release_vectorstore(vs):
    """
    Aggressively free the FAISS index & docstore in-process and on GPU (if any).
    Avoid collapsing local memory.
    """
    try:
        # Drop FAISS vectors from RAM / GPU
        if hasattr(vs, "index"):
            vs.index.reset()
        # Empty the document store
        if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
            vs.docstore._dict.clear()
    except Exception as e:
        logging.warning(f"Vectorstore cleanup failed: {e}")
    finally:
        del vs
        gc.collect()

def search_relevant_context(vectorstore: FAISS, criterion: str, k: int = 5, callbacks=None, verbose_dir=None, use_mmr=False, lambda_mult=0.0, criterion_embedding: Optional[np.ndarray] = None) -> dict:
    """Search for relevant parts of clinical record for a specific criterion"""
    # Augment query for abdominal criterion with cached keywords
    query_text = criterion
    used_keywords = None
    if "abdominal" in criterion.lower():
        kws = get_or_create_keywords_for_abdominal()
        used_keywords = kws
        query_text = criterion + "\nKeywords: " + ", ".join(kws)
    logging.info(f"RETRIEVAL_START | Criterion: '{criterion}' | k={k} | MMR={use_mmr} | λ={lambda_mult if use_mmr else 'N/A'}")
    start_time = time.time()
    
    if criterion_embedding is None:
        criterion_embedding = get_or_create_criteria_embedding(criterion)
    log_embedding_details(criterion, criterion_embedding)
    
    # Search for relevant chunks
    retrieval_method = None
    if use_mmr:
        try:
            # Prefer vector-based MMR if available
            if hasattr(vectorstore, "max_marginal_relevance_search_by_vector") and criterion_embedding is not None:
                documents = vectorstore.max_marginal_relevance_search_by_vector(
                    embedding=criterion_embedding,
                    k=k,
                    fetch_k=k*2,
                    lambda_mult=lambda_mult
                )
                # No scores available from vector-based MMR; wrap with None scores
                results_with_scores = [(doc, None) for doc in documents]
                logging.info(f"RETRIEVAL_METHOD | Using vector-based MMR search with λ={lambda_mult}")
                retrieval_method = "mmr_vector"
            else:
                # Fallback to string-based MMR
                documents = vectorstore.max_marginal_relevance_search(
                    query=query_text,
                    k=k,
                    fetch_k=k*2,
                    lambda_mult=lambda_mult
                )
                # No scores available from string-based MMR; wrap with None scores
                results_with_scores = [(doc, None) for doc in documents]
                logging.info(f"RETRIEVAL_METHOD | Using string-based MMR search with λ={lambda_mult}")
                retrieval_method = "mmr_string"
        except Exception as e:
            logging.error(f"MMR search failed: {str(e)}")
            # Fall back to regular similarity search
            if hasattr(vectorstore, "similarity_search_by_vector") and criterion_embedding is not None:
                documents = vectorstore.similarity_search_by_vector(criterion_embedding, k=k)
                # No scores from vector-based similarity; wrap with None scores
                results_with_scores = [(doc, None) for doc in documents]
                retrieval_method = "fallback_similarity_vector"
            else:
                results_with_scores = vectorstore.similarity_search_with_score(query_text, k=k)
                retrieval_method = "fallback_similarity_string"
            logging.info(f"RETRIEVAL_METHOD | Fallback to standard search after MMR error: {str(e)}")
    else:
        # Use standard similarity search (prefer vector if available)
        if hasattr(vectorstore, "similarity_search_by_vector") and criterion_embedding is not None:
            documents = vectorstore.similarity_search_by_vector(criterion_embedding, k=k)
            # No scores from vector-based similarity; wrap with None scores
            results_with_scores = [(doc, None) for doc in documents]
            logging.info("RETRIEVAL_METHOD | Using vector-based similarity search")
            retrieval_method = "similarity_vector"
        else:
            results_with_scores = vectorstore.similarity_search_with_score(query_text, k=k)
            logging.info("RETRIEVAL_METHOD | Using string-based similarity search")
            retrieval_method = "similarity_string"
    
    documents = [doc for doc, _ in results_with_scores]
    scores = [score for _, score in results_with_scores]
    scores_to_log = None if any(s is None for s in scores) else scores
    
    # Log similarity scores and document details
    log_similarity_scores(query_text, documents, scores_to_log)
    
    # Combine the chunks into a single context
    relevant_context = "\n\n".join([doc.page_content for doc in documents])
    
    # Prepare detailed chunks for GUI
    chunks_with_details = []
    for i, (doc, score) in enumerate(results_with_scores):
        chunk_info = {
            'content': doc.page_content,
            'score': float(score) if score is not None else None,
            'rank': i + 1,
            'metadata': getattr(doc, 'metadata', {})
        }
        chunks_with_details.append(chunk_info)
    
    # Log retrieval time
    retrieval_time = time.time() - start_time
    logging.info(f"RETRIEVAL_END | Time: {retrieval_time:.2f}s | Context length: {len(relevant_context)} chars")
    
    # Log the retrieved context if verbose directory is provided
    if verbose_dir:
        try:
            os.makedirs(verbose_dir, exist_ok=True)
        except Exception:
            pass
        criterion_filename = criterion[:40].replace(" ", "_").replace("/", "_")
        with open(os.path.join(verbose_dir, f"search_{criterion_filename}.txt"), "w") as f:
            f.write(f"CRITERION: {criterion}\n\n")
            f.write(f"RETRIEVED {len(documents)} DOCUMENTS:\n\n")
            for i, doc in enumerate(documents):
                f.write(f"Document {i+1}:\n{doc.page_content}\n\n")
                f.write("-" * 40 + "\n\n")
    
    return {
        'context': relevant_context,
        'chunks': chunks_with_details,
        'retrieval_method': retrieval_method,
        'used_keywords': used_keywords
    }



def create_patient_vectorstore(clinical_text: str, chunk_size=500, chunk_overlap=50) -> FAISS:
    """Create a vector store from patient clinical record"""

    logging.info(f"VECTORIZATION_START | Text length: {len(clinical_text)} chars | Chunk size: {chunk_size} | Overlap: {chunk_overlap}")
    start_time = time.time()
    
    # Use the standard splitter
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    chunks = text_splitter.split_text(clinical_text)
    
  
    logging.info(f"CHUNKING | Created {len(chunks)} chunks")
    chunk_lengths = [len(chunk) for chunk in chunks]
    logging.info(f"CHUNK_STATS | Min: {min(chunk_lengths)} | Max: {max(chunk_lengths)} | Avg: {sum(chunk_lengths)/len(chunks):.1f} chars")
    
    # Sample and log a few chunks
    if chunks:
        sample_size = min(3, len(chunks))
        sample_indices = [0, len(chunks)//2, len(chunks)-1][:sample_size]
        for i, idx in enumerate(sample_indices):
            logging.info(f"CHUNK_SAMPLE_{i} | Index {idx} | First 100 chars: '{chunks[idx][:100]}...'")
    
    
    embedding_start = time.time()
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    embedding_time = time.time() - embedding_start
    
    # Log vectorization time
    total_time = time.time() - start_time
    logging.info(f"EMBEDDING_TIME | {embedding_time:.2f}s for {len(chunks)} chunks | {embedding_time/len(chunks):.4f}s per chunk")
    logging.info(f"VECTORIZATION_END | Total time: {total_time:.2f}s | Chunks: {len(chunks)}")
    
    return vectorstore

def check_eligibility(
    patient_id,
    clinical_text,
    criteria_list,
    chunk_size=500,
    chunk_overlap=50,
    k_value=3,
    use_medical_mappings=False,
    non_clinical_criteria=None,
    date_aware=True,
    callbacks=None,
    verbose_dir=None,
    force_refresh_vectorstore=False,
    use_mmr=False,
    lambda_mult=0.0,
    criteria_embeddings: Optional[Dict[str, np.ndarray]] = None,
    batch_prep: bool = False,
):
    """Check eligibility using embedding-based search for each criterion with date-aware chunking"""
   
  
    
    # Create or load vector store from patient record with caching
    vectorstore = get_or_create_patient_vectorstore(
        patient_id,
        clinical_text, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        force_refresh=force_refresh_vectorstore
    )

    
    # Extract reference date for temporal reasoning
    retriever = DateAwareRetriever(vectorstore)
    reference_date = retriever._get_reference_date()
  
    results = []
    for i, criterion in enumerate(criteria_list):
        # Check if this is a non-clinical criterion
        is_non_clinical = criterion in non_clinical_criteria if non_clinical_criteria else False
        
        if is_non_clinical:
            
            logging.info(f"SKIPPING DETAILED ANALYSIS FOR NON-CLINICAL CRITERION: {criterion}")
            
            results.append({
                "criterion": criterion,
                "analysis": f"Status: Unclear\nJustification: This is a non-clinical criterion that would typically be determined during patient registration, not from clinical notes.",
                "relevant_context": "Non-clinical criterion - no context retrieved",
                "is_non_clinical": True
            })
            continue
        
        
        logging.info(f"PROCESSING CRITERION {i+1}: {criterion}")
      

        criterion_vec = criteria_embeddings.get(criterion) if criteria_embeddings else None
        if date_aware:
            search_result = search_relevant_context_temporal(
                vectorstore,
                criterion,
                k=k_value,
                callbacks=callbacks,
                verbose_dir=verbose_dir,
                use_mmr=use_mmr,
                lambda_mult=lambda_mult,
                criterion_embedding=criterion_vec,
            )
        else:
            search_result = search_relevant_context(
                vectorstore,
                criterion,
                k=k_value,
                callbacks=callbacks,
                verbose_dir=verbose_dir,
                use_mmr=use_mmr,
                lambda_mult=lambda_mult,
                criterion_embedding=criterion_vec,
            )
        relevant_context = search_result['context']
        retrieval_chunks = search_result['chunks']
        
        # Create focused prompt for this criterion
        prompt_template = """
        Based on the following clinical record excerpt, determine if the patient meets this criterion:
        
        Criterion: {criterion}
        
        Relevant Clinical Context:
        {relevant_context}
        
        Provide your analysis in JSON format with the following structure:
        {{
          "status": "met" or "not met" (must be one of these two values exactly, no other values allowed),
          "justification": "Brief explanation with specific evidence from the clinical context"
        }}
        
        If the information is insufficient, you must still make a determination of either "met" or "not met" based on the available evidence. 

        if the criterion is speaks English, or makes decisions, the default is met and only if we find evidence that lead us to conclude the contrary should we mark it as met. We should not mark it as not met for not finding evidence he patient speaks english or is able to make decisions.
        """
        
        # Add medical term mappings if configured
        if use_medical_mappings:
            prompt_template = """
            Based on the following clinical record excerpt, determine if the patient meets this criterion:
            
            Criterion: {criterion}
            
            Important: if the criterion is speaks English, or makes decisions, the default is met and only if we find evidence that lead us to conclude the contrary should we mark it as met. We should not mark it as not met for not finding evidence he patient speaks english or is able to make decisions. 

            
            Relevant Clinical Context:
            {relevant_context}
            
            Important medical term mappings:
            - "ecasa" is enteric-coated aspirin
            
            
            Provide your analysis in JSON format with the following structure:
            {{
              "status": "met" or "not met" (must be one of these two values exactly, no other values allowed),
              "justification": "Brief explanation with specific evidence from the clinical context"
            }}
            
            If the information is insufficient, you must still make a determination of either "met" or "not met" based on the available evidence.
            """
        
        # Add temporal context for time-based criteria if available
        is_temporal_criterion = any(term in criterion.lower() for term in 
                                   ["past", "last", "recent", "ago", "month", "year", "day", "week", "previous"])
        
        if is_temporal_criterion and reference_date and date_aware:
            prompt_template += f"""
            
            IMPORTANT: When evaluating this time-based criterion, use {reference_date} as the reference date. 
            Any mentions of dates in the clinical context should be compared against this reference date.
            """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["criterion", "relevant_context"]
        )
        
        formatted_prompt = prompt.format(
            criterion=criterion,
            relevant_context=relevant_context
        )
        
        # Get LLM analysis for this criterion
        logging.info(f"ANALYZING CRITERION {i+1}")
        response = None if batch_prep else get_llm().invoke(formatted_prompt)

        # Append structured prompt logging (and response if present)
        if verbose_dir:
            try:
                os.makedirs(verbose_dir, exist_ok=True)
            except Exception:
                pass
            try:
                prompt_log_path = os.path.join(verbose_dir, "prompts.jsonl")
                prompt_log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "patient_id": patient_id,
                    "criterion_index": i + 1,
                    "criterion": criterion,
                    "tag": map_criterion_to_tag(criterion),
                    "prompt": formatted_prompt,
                    "analysis": (response.content if response else None),
                    "retrieval_method": search_result.get('retrieval_method'),
                    "used_keywords": search_result.get('used_keywords'),
                    "reference_date": reference_date.isoformat() if isinstance(reference_date, datetime) else None,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "k_value": k_value,
                    "use_mmr": use_mmr,
                    "lambda_mult": (lambda_mult if use_mmr else None),
                    "date_aware": date_aware,
                    "batch_prep": batch_prep,
                    "model": CURRENT_MODEL,
                }
                with open(prompt_log_path, 'a') as pf:
                    pf.write(json.dumps(prompt_log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.warning(f"Failed to write prompt log: {e}")
        
        # Log the LLM response
        if response:
            logging.info(f"LLM ANALYSIS FOR CRITERION {i+1}:\n{response.content}")
        
        # If verbose logging is enabled, save the full prompt and response
        if verbose_dir:
            criterion_filename = criterion[:40].replace(" ", "_").replace("/", "_")
            with open(os.path.join(verbose_dir, f"check_{criterion_filename}.txt"), "w") as f:
                f.write(f"CRITERION: {criterion}\n\n")
                f.write(f"PROMPT:\n{formatted_prompt}\n\n")
                if response:
                    f.write(f"RESPONSE:\n{response.content}\n\n")
        
        results.append({
            "criterion": criterion,
            "analysis": response.content if response else "",
            "relevant_context": relevant_context,
            "retrieval_chunks": retrieval_chunks,  # Add the detailed chunks for GUI
            "is_non_clinical": False,
            "prompt": formatted_prompt,
            "tag": map_criterion_to_tag(criterion)
        })
    
    # Combine results and provide overall assessment
    final_analysis = format_final_results(results)
    return final_analysis, results, vectorstore

def extract_predicted_tags(eligibility_analysis: str, criteria_list: List[str], non_clinical_criteria=None) -> Dict[str, str]:
    """Extract predicted tags from eligibility analysis"""
    # Define default non-clinical criteria if not provided
    if non_clinical_criteria is None:
        non_clinical_criteria = {}
    
    predicted_tags = {}
    
    # Extract status for each criterion
    for i, criterion in enumerate(criteria_list):
        tag = CRITERIA_TO_TAG.get(criterion)
        if not tag:
            continue
        
        # For non-clinical criteria, mark as "registry_required"
        if criterion in non_clinical_criteria:
            predicted_tags[tag] = "registry_required"
            continue
        
        # Look for JSON response in the analysis
        search_pattern = f"## Criterion {i+1}: {criterion}"
        sections = eligibility_analysis.split("## Criterion")
        
        for section in sections:
            if criterion in section:
                # Try to extract JSON object from the section
                try:
                    # Look for JSON pattern in the text
                    json_pattern = re.compile(r'\{[\s\S]*?\}')
                    matches = json_pattern.findall(section)
                    
                    if matches:
                        for match in matches:
                            try:
                                data = json.loads(match)
                                if 'status' in data:
                                    status = data['status'].lower().strip()
                                    if status == "met":
                                        predicted_tags[tag] = "met"
                                        break
                                    elif status == "not met":
                                        predicted_tags[tag] = "not met"
                                        break
                            except json.JSONDecodeError:
                                continue
                    
                    # Fallback: Check for status line if JSON parsing fails
                    if tag not in predicted_tags:
                        lines = section.split('\n')
                        for line in lines:
                            if "Status:" in line:
                                status_text = line.split(':', 1)[1].strip().lower()
                                if "met" in status_text and "not" not in status_text:
                                    predicted_tags[tag] = "met"
                                    break
                                elif "not met" in status_text:
                                    predicted_tags[tag] = "not met"
                                    break
                
                except Exception as e:
                    logging.error(f"Error extracting JSON for criterion {criterion}: {e}")
                    # Fallback to old text parsing method
                    lines = section.split('\n')
                    for line in lines:
                        if "Status:" in line:
                            status_text = line.split(':', 1)[1].strip().lower()
                            if "met" in status_text and "not" not in status_text:
                                predicted_tags[tag] = "met"
                                break
                            elif "not met" in status_text:
                                predicted_tags[tag] = "not met"
                                break
        
        # If no valid status was found, default to "not met"
        if tag not in predicted_tags:
            logging.warning(f"[FALLBACK] Tag '{tag}' for criterion '{criterion}' defaulted to 'not met'")
            predicted_tags[tag] = "not met"
    
    return predicted_tags

# (removed unused cost/token estimation utilities)


def process_patient(
    patient_id,
    criteria_list,
    chunk_size=500,
    chunk_overlap=50, 
    k_value=3,
    use_medical_mappings=False,
    non_clinical_criteria=None,
    default_registry_values=None,
    text_preprocessor=None, 
    callbacks=None,
    verbose_dir=None,
    force_refresh_vectorstore=False,
    use_mmr=False,
    lambda_mult=0.0,
    date_aware=True,
    criteria_embeddings: Optional[Dict[str, np.ndarray]] = None,
    batch_prep: bool = False,
    patient_file_base_path: Optional[Union[str, Path]] = None,
):
    """Process a patient record against eligibility criteria."""
    vectorstore = None
    try:
        # Load patient record
        logging.info(f"PROCESSING PATIENT {patient_id}")
        
        
        # Define default non-clinical criteria if not provided
        if non_clinical_criteria is None:
            non_clinical_criteria = {
                
            }
            
        # Define default registry values (what we'd expect from registration data)
       
        # Load patient record using provided base path or default
        base_path = Path(patient_file_base_path) if patient_file_base_path else Path("2018n2c2")
        xml_path = base_path / f"{patient_id}.xml"
        logging.info(f"Loading patient record from {xml_path}")
        clinical_text, ground_truth = load_patient_record(str(xml_path))
        
        # Apply text preprocessing if provided
        if text_preprocessor and callable(text_preprocessor):
            original_length = len(clinical_text)
            clinical_text = text_preprocessor(clinical_text)
            new_length = len(clinical_text)
            logging.info(f"Applied text preprocessing: {original_length} chars -> {new_length} chars")
        
        # Log the full clinical text for reference (but keep it manageable)
        logging.info(f"CLINICAL TEXT LENGTH: {len(clinical_text)} characters")
        
        # Log token count estimate (approx 4 chars per token)
        token_estimate = len(clinical_text) / 4
        logging.info(f"ESTIMATED TOKEN COUNT: ~{token_estimate:.0f} tokens")
        
        # Check eligibility with the provided parameters
        # Use temporal version if date_aware is True, otherwise standard version
        if date_aware:
            eligibility_analysis, criterion_results, vectorstore = check_eligibility_temporal(
                patient_id,  # Pass patient_id to enable caching
                clinical_text, 
                criteria_list,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                k_value=k_value,
                use_medical_mappings=use_medical_mappings,
                non_clinical_criteria=non_clinical_criteria,
                callbacks=callbacks,
                verbose_dir=verbose_dir,
                force_refresh_vectorstore=force_refresh_vectorstore,
                use_mmr=use_mmr,
                lambda_mult=lambda_mult,
                criteria_embeddings=criteria_embeddings,
                batch_prep=batch_prep
            )
        else:
            eligibility_analysis, criterion_results, vectorstore = check_eligibility(
                patient_id,  # Pass patient_id to enable caching
                clinical_text, 
                criteria_list,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                k_value=k_value,
                use_medical_mappings=use_medical_mappings,
                non_clinical_criteria=non_clinical_criteria,
                date_aware=date_aware,
                callbacks=callbacks,
                verbose_dir=verbose_dir,
                force_refresh_vectorstore=force_refresh_vectorstore,
                use_mmr=use_mmr,
                lambda_mult=lambda_mult,
                criteria_embeddings=criteria_embeddings,
                batch_prep=batch_prep,
            )
        
        # Extract predicted tags from analysis
        predicted_tags = extract_predicted_tags(
            eligibility_analysis, 
            criteria_list,
            non_clinical_criteria=non_clinical_criteria
        )
        
        # Apply default registry values for non-clinical criteria
        for criterion, tag in non_clinical_criteria.items():
            if tag in predicted_tags and predicted_tags[tag] == "registry_required":
                if tag in default_registry_values:
                    predicted_tags[tag] = default_registry_values[tag]
                    logging.info(f"Applied default registry value for {tag}: {default_registry_values[tag]}")
        
        # Debug logging for  calculation
        logging.info(f"DEBUG | Predicted tags: {json.dumps(predicted_tags, indent=2)}")
        logging.info(f"DEBUG | Ground truth: {json.dumps(ground_truth, indent=2)}")
        
        # Make sure ground truth and predicted tags are not empty
        if not predicted_tags:
            logging.error("ERROR: predicted_tags is empty!")
        if not ground_truth:
            logging.error("ERROR: ground_truth is empty!")
        
        # Check tag format consistency
        for tag in predicted_tags:
            if predicted_tags[tag] not in ["met", "not met", "registry_required"]:
                logging.warning(f"WARNING: Unexpected value for tag {tag}: {predicted_tags[tag]}")
        for tag in ground_truth:
            if ground_truth[tag] not in ["met", "not met"]:
                logging.warning(f"WARNING: Unexpected value for ground truth {tag}: {ground_truth[tag]}")
                
        # Evaluate results
        metrics = evaluate_results(predicted_tags, ground_truth)
        
        # Log detailed comparison of predictions vs ground truth
        logging.info("EVALUATION RESULTS:")
        for criterion, details in metrics["per_criterion"].items():
            status = "✓" if details["correct"] else "✗"
            source = " (from registry)" if default_registry_values and criterion in default_registry_values else ""
            logging.info(f"{status} {criterion}: predicted={details['predicted']}{source}, actual={details['actual']}")
        
        logging.info(f"ACCURACY: {metrics['accuracy']:.2f} ({metrics['correct']}/{metrics['total']} correct)")
        
        return {
            "patient_id": patient_id,
            "eligibility_analysis": eligibility_analysis,
            "predicted_tags": predicted_tags,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "criterion_details": criterion_results,
            "processing_metadata": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "k_value": k_value,
                "use_medical_mappings": use_medical_mappings,
                "clinical_text_length": len(clinical_text),
                "estimated_tokens": token_estimate,
                "non_clinical_criteria": list(non_clinical_criteria.keys()) if non_clinical_criteria else [],
                "text_preprocessor_applied": text_preprocessor is not None,
                "date_aware_chunking": True,
                "use_mmr": use_mmr,
                "lambda_mult": lambda_mult if use_mmr else None
            }
        }
        
    except Exception as e:
        logging.error(f"ERROR: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None
    finally:
        if vectorstore:
            release_vectorstore(vectorstore)
            logging.info(f"Released vectorstore for patient {patient_id} in finally block.")

def format_final_results(results):
    """Format final results of eligibility analysis"""
    formatted = "# Patient Eligibility Analysis\n\n"
    
    # Create a summary table
    formatted += "## Summary of Results\n\n"
    formatted += "| Criterion | Status | Summary |\n"
    formatted += "|-----------|--------|--------|\n"
    
    for i, result in enumerate(results):
        criterion = result["criterion"]
        analysis = result["analysis"]
        is_non_clinical = result.get("is_non_clinical", False)
        
        # Extract status and justification from JSON if possible
        status = "Unknown"
        justification_summary = ""
        
        if is_non_clinical:
            status = "Registry Required"
            justification_summary = "Non-clinical criterion determined during registration"
        else:
            try:
                # Look for JSON pattern in the text
                json_pattern = re.compile(r'\{[\s\S]*?\}')
                matches = json_pattern.findall(analysis)
                if matches:
                    for match in matches:
                        try:
                            data = json.loads(match)
                            if 'status' in data and 'justification' in data:
                                status = data['status'].upper()
                                justification_summary = data['justification'][:50] + "..." if len(data['justification']) > 50 else data['justification']
                                break
                        except:
                            pass
                
                # Fallback to text parsing if JSON not found
                if status == "Unknown":
                    for line in analysis.split("\n"):
                        if "Status:" in line:
                            status = line.split(":", 1)[1].strip()
                            break
            except Exception as e:
                logging.error(f"Error parsing analysis for {criterion}: {e}")
        
        # Add to summary table
        formatted += f"| {criterion} | {status} | {justification_summary} |\n"
    
    # Add detailed analysis sections
    formatted += "\n## Detailed Analysis\n\n"
    for i, result in enumerate(results):
        criterion = result["criterion"]
        analysis = result["analysis"]
        is_non_clinical = result.get("is_non_clinical", False)
        
        formatted += f"### Criterion {i+1}: {criterion}\n\n"
        
        if is_non_clinical:
            formatted += "**Note:** This is a non-clinical criterion typically handled through patient registration.\n\n"
        
        formatted += f"{analysis}\n\n"
    
    return formatted

def load_patient_record(xml_path):
    """Load patient record from XML file"""
    logging.info(f"Loading patient record from {xml_path}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract clinical text
        texts = []
        for text_elem in root.findall('.//TEXT'):
            texts.append(text_elem.text)
        
        clinical_text = '\n\n'.join([t for t in texts if t])
        
        # Extract ground truth tags
        tags_elem = root.find('.//TAGS')
        ground_truth = {}
        if tags_elem is not None:
            for tag in tags_elem:
                tag_name = tag.tag
                tag_met = tag.get('met')
                ground_truth[tag_name] = tag_met
        
        return clinical_text, ground_truth
    
    except Exception as e:
        logging.error(f"Error loading patient record: {e}")
        raise

def evaluate_results(predicted_tags, ground_truth, non_clinical_criteria_tags=None):
    """Evaluate predicted tags against ground truth with separate clinical-only metrics"""
    if non_clinical_criteria_tags is None:
        non_clinical_criteria_tags = ["ENGLISH", "MAKES-DECISIONS"]
    
    # Initialize counters for all criteria
    correct_all = 0
    total_all = 0
    
    # Initialize counters for clinical-only criteria
    correct_clinical = 0
    total_clinical = 0
    
    per_criterion = {}
    
    for tag, predicted in predicted_tags.items():
        if tag in ground_truth:
            is_non_clinical = tag in non_clinical_criteria_tags
            total_all += 1
            if not is_non_clinical:
                total_clinical += 1
                
            actual = ground_truth[tag]
            is_correct = predicted == actual
            
            if is_correct:
                correct_all += 1
                if not is_non_clinical:
                    correct_clinical += 1
            
            per_criterion[tag] = {
                "predicted": predicted,
                "actual": actual,
                "correct": is_correct,
                "is_non_clinical": is_non_clinical
            }
    
    # Calculate accuracy metrics
    accuracy_all = correct_all / total_all if total_all > 0 else 0
    accuracy_clinical = correct_clinical / total_clinical if total_clinical > 0 else 0
    
    return {
        "accuracy": accuracy_all,
        "accuracy_clinical_only": accuracy_clinical,
        "correct": correct_all,
        "total": total_all,
        "correct_clinical_only": correct_clinical,
        "total_clinical_only": total_clinical,
        "per_criterion": per_criterion
    }

def load_eligibility_criteria(criteria_file=None):
    """Load eligibility criteria from a file or use defaults"""
    default_criteria = [
        "Drug abuse, current or past",
        "Current alcohol use over weekly recommended limits",
        "Patient must speak English",
        "Patient must make their own medical decisions",
        "History of intra abdominal surgery, small or large intestine resection or small bowel obstruction",
        "Major diabetes-related complication. Major complication as opposed to minor complication for this purpose is defined as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: Amputation, Kidney damage, Skin conditions, Retinopathy, nephropathy, neuropathy",
        "Advanced cardiovascular disease, as defined by having 2 or more of the following: Taking 2 or more medications to treat CAD, History of myocardial infarction (MI), Currently experiencing angina, Ischemia, past or present",
        "Myocardial infarction in the past 6 months",
        "Diagnosis of ketoacidosis in the past year",
        "Taken a dietary supplement (excluding Vitamin D) in the past 2 months",
        "Use of aspirin to prevent myocardial infarction",
        "Any HbA1c value between 6.5 and 9.5%",
        "Serum creatinine > upper limit of normal"
    ]
    
    # Use provided filename or default
    if criteria_file is None:
        criteria_file = "eligibility_criteria.json"
    
    if os.path.exists(criteria_file):
        try:
            with open(criteria_file, 'r') as f:
                criteria = json.load(f)
                if isinstance(criteria, list) and criteria:
                    logging.info(f"Successfully loaded criteria from {criteria_file}")
                    return criteria
                else:
                    logging.warning(f"{criteria_file} is empty or not a list -- using default criteria.")
        except Exception as e:
            logging.warning(f"Could not parse {criteria_file}: {e} – using default criteria.")
    else:
        if criteria_file != "eligibility_criteria.json":  # Only warn if user specified a custom file
            logging.warning(f"{criteria_file} not found – falling back to built-in criteria.")
        else:
            logging.info(f"Default {criteria_file} not found – using built-in criteria.")
    
    return default_criteria

def get_or_create_patient_vectorstore(patient_id, clinical_text, chunk_size=500, chunk_overlap=50, force_refresh=False):
    """Create or retrieve a vector store from patient clinical record with caching.
    
    Args:
        patient_id: The ID of the patient
        clinical_text: The patient's clinical text
        chunk_size: Size of text chunks for vectorization
        chunk_overlap: Overlap between chunks
        date_aware: Deprecated for standard vectorstore; kept for compatibility (no effect)
        force_refresh: Whether to force recreation of the vectorstore even if cached
        
    Returns:
        FAISS vectorstore with patient record embeddings
    """
    # Create cache directory if it doesn't exist

    tag = "standard"
    idx_name = f"patient_{patient_id}_{tag}_cs{chunk_size}_ov{chunk_overlap}"



    cache_dir = Path("vectorstore_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Define cache path for this patient and configuration
    cache_path = cache_dir / f"{idx_name}.faiss"
    index_path = cache_dir / f"{idx_name}.pkl"
    
    # Check if vectorstore exists in cache and we're not forcing refresh
    if cache_path.exists() and index_path.exists() and not force_refresh:
        logging.info(f"VECTORIZATION_CACHE_HIT | Loading vectorstore from cache for patient {patient_id}")
        try:
            start_time = time.time()
            vectorstore = FAISS.load_local(
                folder_path=str(cache_dir),
                embeddings=embeddings,
                index_name=idx_name,
                allow_dangerous_deserialization=True  # Add this parameter for security bypass
            )
            load_time = time.time() - start_time
            logging.info(f"VECTORIZATION_CACHE_LOAD | Loaded vectorstore in {load_time:.2f}s")
            return vectorstore
        except Exception as e:
            logging.error(f"VECTORIZATION_CACHE_ERROR | Failed to load cached vectorstore: {e}")
            logging.info("Falling back to creating new vectorstore...")
    
    # Create new vectorstore if not cached or cache loading failed
    logging.info(f"VECTORIZATION_CACHE_MISS | Creating new vectorstore for patient {patient_id}")
   

   
    vs = create_patient_vectorstore(
            clinical_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    # Save to cache
    try:
        start_time = time.time()
        vs.save_local(
            folder_path=str(cache_dir),
            index_name=idx_name
        )
        save_time = time.time() - start_time
        logging.info(f"VECTORIZATION_CACHE_SAVE | Saved vectorstore in {save_time:.2f}s")
    except Exception as e:
        logging.error(f"VECTORIZATION_CACHE_ERROR | Failed to save vectorstore to cache: {e}")
    
    return vs

def get_or_create_patient_vectorstore_temporal(
    patient_id,
    clinical_text,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    force_refresh: bool = False,
):
    """Create or retrieve a temporal-aware vectorstore with caching.
    Cache key includes patient_id, mode tag, chunk_size, and chunk_overlap.
    """
    tag = "temporal"
    idx_name = f"patient_{patient_id}_{tag}_cs{chunk_size}_ov{chunk_overlap}"

    cache_dir = Path("vectorstore_cache")
    cache_dir.mkdir(exist_ok=True)

    cache_path = cache_dir / f"{idx_name}.faiss"
    index_path = cache_dir / f"{idx_name}.pkl"

    if cache_path.exists() and index_path.exists() and not force_refresh:
        logging.info(f"TEMPORAL_VECTORIZATION_CACHE_HIT | Loading vectorstore from cache for patient {patient_id}")
        try:
            start_time = time.time()
            vectorstore = FAISS.load_local(
                folder_path=str(cache_dir),
                embeddings=embeddings,
                index_name=idx_name,
                allow_dangerous_deserialization=True
            )
            load_time = time.time() - start_time
            logging.info(f"TEMPORAL_VECTORIZATION_CACHE_LOAD | Loaded vectorstore in {load_time:.2f}s")
            return vectorstore
        except Exception as e:
            logging.error(f"TEMPORAL_VECTORIZATION_CACHE_ERROR | Failed to load cached vectorstore: {e}")
            logging.info("Falling back to creating new temporal vectorstore...")

    logging.info(f"TEMPORAL_VECTORIZATION_CACHE_MISS | Creating new temporal vectorstore for patient {patient_id}")
    vs = create_patient_vectorstore_temporal(
        clinical_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    try:
        start_time = time.time()
        vs.save_local(
            folder_path=str(cache_dir),
            index_name=idx_name
        )
        save_time = time.time() - start_time
        logging.info(f"TEMPORAL_VECTORIZATION_CACHE_SAVE | Saved vectorstore in {save_time:.2f}s")
    except Exception as e:
        logging.error(f"TEMPORAL_VECTORIZATION_CACHE_ERROR | Failed to save temporal vectorstore to cache: {e}")

    return vs

def list_patient_ids(folder: Union[str, Path]) -> List[int]:
    """
    Scan a directory for <patient_id>.xml files and return a list of integer IDs.
    Handles non-integer filenames gracefully by skipping them.
    """
    folder_path = Path(folder) # Convert to Path object
    ids = []
    if not folder_path.is_dir():
        logging.warning(f"Provided patient folder '{folder}' is not a valid directory. Cannot list patient IDs.")
        return ids # Return empty list if not a directory

    for p in folder_path.glob("*.xml"):
        try:
            # Ensure p.stem is a valid integer before appending
            patient_id = int(p.stem)
            ids.append(patient_id)
        except ValueError:
            logging.debug(f"Skipping file '{p.name}' as its stem '{p.stem}' is not a valid integer patient ID.")
            continue # Skip files that don't have an integer stem
    
    if not ids:
        logging.warning(f"No XML files with integer patient IDs found in '{folder_path}'.")
        
    return sorted(ids)

# =================== DATE-AWARE ALTERNATIVES  ===================

def identify_temporal_criteria(criteria_list: List[str]) -> Set[str]:
    """
    Identify which criteria are time/date dependent.
    Only these will use the new temporal filtering.
    """
    temporal_keywords = [
        "past", "last", "recent", "ago", "month", "months", "year", "years", 
        "day", "days", "week", "weeks", "previous", "within", "6 months", "2 months"
    ]
    
    temporal_criteria = set()
    for criterion in criteria_list:
        if any(keyword in criterion.lower() for keyword in temporal_keywords):
            temporal_criteria.add(criterion)
            logging.info(f"TEMPORAL_CRITERION_DETECTED: {criterion}")
    
    return temporal_criteria

class DateAwareRetriever:
    """
    Alternative retriever that adds temporal filtering for date-dependent criteria.
    Fallback to standard search if temporal filtering fails.
    """
    def __init__(self, vectorstore: FAISS):
        self.vs = vectorstore
    
    def _parse_time_window(self, criterion: str) -> Optional[int]:
        """Extract time window in days from criterion text."""
        patterns = [
            r'(?:past|last|previous|within)\s+(\d+)\s*(month|week|day|year)s?',
            r'(\d+)\s*(month|week|day|year)s?\s+(?:ago|prior)',
            r'in\s+the\s+past\s+(\d+)\s*(month|week|day|year)s?'
        ]
        
        for pattern in patterns:
            m = re.search(pattern, criterion, re.IGNORECASE)
            if m:
                n, unit = int(m.group(1)), m.group(2).lower()
                if 'year' in unit:
                    return n * 365
                elif 'month' in unit:
                    return n * 30
                elif 'week' in unit:
                    return n * 7
                elif 'day' in unit:
                    return n
        return None
    
    def _get_reference_date(self) -> Optional[datetime]:
        """Extract the most recent date from vectorstore metadata."""
        dates = []
        try:
            for doc_id in self.vs.docstore._dict:
                doc = self.vs.docstore._dict[doc_id]
                doc_date = doc.metadata.get('note_date')
                if doc_date and isinstance(doc_date, datetime):
                    dates.append(doc_date)
        except Exception as e:
            logging.warning(f"Error extracting reference date: {e}")
            return None
        
        if dates:
            return max(dates)
        return None
    
    def search_with_temporal_filter(self, query: str, k: int = 5) -> List[Tuple]:
        """
        Pure temporal filtering - filter by date but preserve semantic ranking.
        """
        window_days = self._parse_time_window(query)
        reference_date = self._get_reference_date()
        
        if window_days and reference_date:
            logging.info(f"TEMPORAL_FILTER_PURE | Window: {window_days} days, Reference: {reference_date.strftime('%Y-%m-%d')}")
        
            # Get initial documents with standard similarity search
            initial_k = k * 3  # Get extra documents for filtering
            docs_with_scores = self.vs.similarity_search_with_score(query, k=initial_k)
        
            cutoff_date = reference_date - timedelta(days=window_days)
            logging.info(f"TEMPORAL_FILTER_DEBUG | Cutoff date: {cutoff_date.strftime('%Y-%m-%d')} (docs older than this will be excluded)")
        
            # Pure date filtering - NO score changes
            filtered_docs = []
            excluded_docs = []
        
            for i, (doc, score) in enumerate(docs_with_scores):
                doc_date = doc.metadata.get('note_date')
                content_preview = doc.page_content[:100].replace('\n', ' ')
                
                if doc_date and isinstance(doc_date, datetime):
                    if doc_date >= cutoff_date:
                        filtered_docs.append((doc, score))  # Keep ORIGINAL score
                        logging.info(f"TEMPORAL_FILTER_DEBUG | ✅ INCLUDED Doc {i+1}: Date={doc_date.strftime('%Y-%m-%d')}, Score={score:.4f}, Content='{content_preview}...'")
                    else:
                        excluded_docs.append((doc, score))
                        logging.info(f"TEMPORAL_FILTER_DEBUG | ❌ EXCLUDED Doc {i+1}: Date={doc_date.strftime('%Y-%m-%d')}, Score={score:.4f}, Content='{content_preview}...'")
                else:
                    # Keep docs without dates
                    filtered_docs.append((doc, score))
                    logging.info(f"TEMPORAL_FILTER_DEBUG | ✅ INCLUDED Doc {i+1}: Date=None, Score={score:.4f}, Content='{content_preview}...'")
        
            # Log what we're doing
            logging.info(f"TEMPORAL_FILTER_PURE | Kept {len(filtered_docs)} docs within window, excluded {len(excluded_docs)} older docs")
            
            # Log details about excluded docs if any contain MI keywords
            for i, (doc, score) in enumerate(excluded_docs):
                content = doc.page_content.lower()
                if any(term in content for term in ['nstemi', 'myocardial', 'infarction', 'stemi', 'mi']):
                    logging.warning(f"TEMPORAL_FILTER_WARNING | Excluded doc {i+1} contains MI keywords! Date={doc.metadata.get('note_date')}, Content preview: '{doc.page_content[:200]}...'")
        
            # If we have enough filtered docs, use them with ORIGINAL ranking
            if len(filtered_docs) >= k:
                filtered_docs.sort(key=lambda x: x[1])  # Sort by original similarity
                return filtered_docs[:k]
            else:
                # ate proper Document object for "no recent information"
                
                
                no_info_content = f"No recent clinical information found for this time-dependent criterion within the past {window_days} days."
                
                no_info_doc = Document(
                    page_content=no_info_content,
                    metadata={"note_date": None, "section": "TEMPORAL_NO_DATA"}
                )
                
                logging.info(f"TEMPORAL_FILTER_PURE | No recent docs found, returning explicit 'no information' context")
                return [(no_info_doc, 0.0)]  # Return (Document, score) tuple
        else:
            # Standard search for non-temporal criteria
            return self.vs.similarity_search_with_score(query, k=k)


def search_relevant_context_temporal(vectorstore: FAISS, criterion: str, k: int = 5, 
                               callbacks=None, verbose_dir=None, use_mmr=False, 
                               lambda_mult=0.0, criterion_embedding: Optional[np.ndarray] = None) -> dict:
    """
    Alternative search function with temporal awareness.
    Only used for date-dependent criteria, otherwise identical to original.
    """
    logging.info(f"TEMPORAL_RETRIEVAL_START | Criterion: '{criterion}' | k={k}")
    start_time = time.time()

    # Get embedding for the criterion from cache (or use provided precomputed embedding)
    if criterion_embedding is None:
        criterion_embedding = get_or_create_criteria_embedding(criterion)
    log_embedding_details(criterion, criterion_embedding)

    # Create date-aware retriever
    retriever = DateAwareRetriever(vectorstore)
    reference_date = retriever._get_reference_date()
    
    # Augment string queries with abdominal keywords for the non-vector temporal path
    augmented_query_text = criterion
    used_keywords = None
    if "abdominal" in criterion.lower():
        kws = get_or_create_keywords_for_abdominal()
        used_keywords = kws
        augmented_query_text = criterion + "\nKeywords: " + ", ".join(kws)
    
    # Search with potential temporal filtering (prefer vector-based initial search if available)
    retrieval_method = None
    if hasattr(vectorstore, "similarity_search_by_vector") and criterion_embedding is not None:
        initial_docs = vectorstore.similarity_search_by_vector(criterion_embedding, k=k*3)
        # No scores available; wrap with None
        initial_with_scores = [(doc, None) for i, doc in enumerate(initial_docs)]
        # Temporally filter using the same logic as retriever
        retriever_results = []
        reference_date = retriever._get_reference_date()
        window_days = retriever._parse_time_window(criterion)
        if window_days and reference_date:
            cutoff_date = reference_date - timedelta(days=window_days)
            for doc, score in initial_with_scores:
                doc_date = doc.metadata.get('note_date')
                if not doc_date or not isinstance(doc_date, datetime) or doc_date >= cutoff_date:
                    retriever_results.append((doc, score))
            retriever_results = retriever_results[:k] if len(retriever_results) >= k else retriever_results
            results_with_scores = retriever_results if retriever_results else initial_with_scores[:k]
            retrieval_method = "temporal_vector"
        else:
            results_with_scores = initial_with_scores[:k]
            retrieval_method = "temporal_vector_no_window"
    else:
        results_with_scores = retriever.search_with_temporal_filter(augmented_query_text, k=k)
        retrieval_method = "temporal_string"

    documents = [doc for doc, _ in results_with_scores]
    scores = [score for _, score in results_with_scores]

    scores_to_log = None if any(s is None for s in scores) else scores

    log_similarity_scores(augmented_query_text, documents, scores_to_log)

    

    # Combine the chunks into a single context
    relevant_context = "\n\n".join([doc.page_content for doc in documents])
    
    # Prepare detailed chunks for GUI
    chunks_with_details = []
    for i, (doc, score) in enumerate(results_with_scores):
        chunk_info = {
            'content': doc.page_content,
            'score': float(score) if score is not None else None, 
            'rank': i + 1,
            'metadata': getattr(doc, 'metadata', {})
        }
        chunks_with_details.append(chunk_info)

    # Log retrieval time
    retrieval_time = time.time() - start_time
    logging.info(f"TEMPORAL_RETRIEVAL_END | Time: {retrieval_time:.2f}s | Context length: {len(relevant_context)} chars")

    # Log the retrieved context if verbose directory is provided
    if verbose_dir:
        try:
            os.makedirs(verbose_dir, exist_ok=True)
        except Exception:
            pass
        criterion_filename = criterion[:40].replace(" ", "_").replace("/", "_")
        with open(os.path.join(verbose_dir, f"temporal_search_{criterion_filename}.txt"), "w") as f:
            f.write(f"CRITERION: {criterion}\n\n")
            f.write(f"RETRIEVED {len(documents)} DOCUMENTS:\n\n")
            for i, doc in enumerate(documents):
                f.write(f"Document {i+1}:\n{doc.page_content}\n\n")
                f.write("-" * 40 + "\n\n")

    return {
        'context': relevant_context,
        'chunks': chunks_with_details,
        'retrieval_method': retrieval_method,
        'reference_date': reference_date,
        'used_keywords': used_keywords
    }

def create_patient_vectorstore_temporal(clinical_text: str, chunk_size=500, chunk_overlap=50) -> FAISS:
    """
    Alternative vectorstore creation that properly parses dates into datetime objects.
    Uses section-aware chunking with datetime conversion.
    """
    logging.info(f"TEMPORAL_VECTORIZATION_START | Text length: {len(clinical_text)} chars")
    start_time = time.time()

    # 1) split into note/section chunks with datetime parsing
    chunks_meta = split_notes_and_sections_temporal(clinical_text)

    # 2) further split large sections by size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunk_texts, metadatas_list = [], []
    dates_parsed = 0

    for cm in chunks_meta:
        sub_chunks = splitter.split_text(cm['text'])
        for sc in sub_chunks:
            chunk_texts.append(sc)
            metadatas_list.append({
                "note_date": cm["note_date"],  # This will be datetime or None
                "section": cm["section"]
            })
            if cm["note_date"] and isinstance(cm["note_date"], datetime):
                dates_parsed += 1

    logging.info(f"TEMPORAL_CHUNKING | Created {len(chunk_texts)} chunks, {dates_parsed} with parsed dates")

    # 3) batch embed all chunk_texts
    embeddings_list = embeddings.embed_documents(chunk_texts)

    # 4) build FAISS with precomputed embeddings + metadata
    text_embedding_pairs = list(zip(chunk_texts, embeddings_list))

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embeddings,
        metadatas=metadatas_list
    )

    total_time = time.time() - start_time
    logging.info(f"TEMPORAL_VECTORIZATION_END | Total time: {total_time:.2f}s")

    return vectorstore

def split_notes_and_sections_temporal(full_text: str) -> List[Dict[str, Any]]:
    """
    Alternative version that properly converts dates to datetime objects.
    """
    # Use same logic as original but with datetime conversion
    raw_notes = re.split(r"\*{5,}|(?=Record date:)", full_text)
    out_chunks = []
    
    # Same section headers as original
    section_headers = [
        "HPI", "CHIEF COMPLAINT", "CC", "HISTORY OF PRESENT ILLNESS",
        "PHYSICAL EXAM", "EXAM", "LABS", "LABORATORY RESULTS", "LABORATORY DATA","LABORATORY EVALUATION"
        "ASSESSMENT & PLAN", "ASSESSMENT", "PLAN", 
        "MEDICATIONS", "CURRENT MEDICATIONS", 
        "PAST MEDICAL HISTORY", "PMH", "FAMILY HISTORY", "MEDICAL HISTORY", "MEDICAL HISTORY NOTE",
        "REVIEW OF SYSTEMS", "REVIEW OF SYSTEMS NOTE",
        "SOCIAL HISTORY", "DISCHARGE DIAGNOSIS", "IMPRESSION","CLINICAL IMPRESSION",
        "FINAL DIAGNOSIS", "PRIMARY DIAGNOSIS", "SECONDARY DIAGNOSIS",
        "PROCEDURE NOTE", "FOLLOW-UP PLAN", 
        "PHYSICAL EXAMINATION", "DISPOSITION","ALLERGIES", "ADVERSE REACTIONS",
        "RADIOLOGY", "EKG", "CT", "MRI", "XRAY", "IMAGING", "IMAGING STUDIES",
    ]
    hdr_pattern = re.compile(rf"^\s*({'|'.join(section_headers)})[:\s]*$", re.IGNORECASE)
    
    for note in raw_notes:
        note = note.strip()
        if not note:
            continue
            
        # Extract date and convert to datetime
        m = re.search(r"Record date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", note)
        note_date = None
        if m:
            try:
                note_date = parse_date(m.group(1))
                logging.debug(f"TEMPORAL_DATE_PARSED | {m.group(1)} → {note_date}")
            except ValueError as e:
                logging.warning(f"TEMPORAL_DATE_PARSE_ERROR | {m.group(1)}: {e}")
                note_date = None
        
        # Same section splitting logic as original
        buf, current_section = [], "GENERAL"
        for line in note.splitlines():
            if hdr_pattern.match(line.strip()):
                if buf:
                    out_chunks.append({
                        "text": "\n".join(buf).strip(),
                        "note_date": note_date,  # datetime object or None
                        "section": current_section
                    })
                    buf = []
                current_section = line.strip().rstrip(':').upper()
                buf.append(line)
            else:
                buf.append(line)
        
        if buf:
            out_chunks.append({
                "text": "\n".join(buf).strip(),
                "note_date": note_date,  # datetime object or None
                "section": current_section
            })
    
    return out_chunks

def check_eligibility_temporal(patient_id, clinical_text, criteria_list, chunk_size=500, 
                             chunk_overlap=50, k_value=3, use_medical_mappings=False, 
                             non_clinical_criteria=None, callbacks=None, verbose_dir=None, 
                             force_refresh_vectorstore=False, use_mmr=False, lambda_mult=0.0,
                             criteria_embeddings: Optional[Dict[str, np.ndarray]] = None,
                             batch_prep: bool = False):
    """
    Alternative eligibility checker that uses temporal filtering for date-dependent criteria.
    Falls back to standard processing for non-temporal criteria.
    """
    logging.info(f"TEMPORAL_ELIGIBILITY_CHECK | Patient {patient_id}")
    
    # Identify which criteria are temporal
    temporal_criteria = identify_temporal_criteria(criteria_list)
    
    # Create or load temporal-aware vectorstore with caching
    vectorstore = get_or_create_patient_vectorstore_temporal(
        patient_id,
        clinical_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_refresh=force_refresh_vectorstore
    )
    
    results = []
    for i, criterion in enumerate(criteria_list):
        # Check if this is a non-clinical criterion (same as original)
        if non_clinical_criteria and criterion in non_clinical_criteria:
            results.append({
                "criterion": criterion,
                "analysis": f"Status: Unclear\nJustification: This is a non-clinical criterion that would typically be determined during patient registration, not from clinical notes.",
                "relevant_context": "Non-clinical criterion - no context retrieved",
                "is_non_clinical": True
            })
            continue
        
        logging.info(f"PROCESSING CRITERION {i+1}: {criterion}")
        
        # Use temporal search for temporal criteria, standard search for others
        if criterion in temporal_criteria:
            logging.info(f"TEMPORAL_CRITERION | Using temporal search for: {criterion}")
            search_result = search_relevant_context_temporal(
                vectorstore, criterion, k=k_value, callbacks=callbacks, 
                verbose_dir=verbose_dir, use_mmr=use_mmr, lambda_mult=lambda_mult,
                criterion_embedding=(criteria_embeddings.get(criterion) if criteria_embeddings else None)
            )
        else:
            logging.info(f"STANDARD_CRITERION | Using standard search for: {criterion}")
            search_result = search_relevant_context(
                vectorstore, criterion, k=k_value, callbacks=callbacks, 
                verbose_dir=verbose_dir, use_mmr=use_mmr, lambda_mult=lambda_mult,
                criterion_embedding=(criteria_embeddings.get(criterion) if criteria_embeddings else None)
            )
        
        relevant_context = search_result['context']
        retrieval_chunks = search_result['chunks']
        reference_date = search_result.get('reference_date')
        used_keywords = search_result.get('used_keywords')
        
        # Same prompt logic as original
        prompt_template = """
        Based on the following clinical record excerpt, determine if the patient meets this criterion:
        
        Criterion: {criterion}
        
        Relevant Clinical Context:
        {relevant_context}
        
        Provide your analysis in JSON format with the following structure:
        {{
          "status": "met" or "not met" (must be one of these two values exactly, no other values allowed),
          "justification": "Brief explanation with specific evidence from the clinical context"
        }}
        
        If the information is insufficient, you must still make a determination of either "met" or "not met" based on the available evidence.
        """
        
        # Add temporal context note for temporal criteria
        if criterion in temporal_criteria:
            # Get reference date for temporal context
            retriever = DateAwareRetriever(vectorstore)
            reference_date = retriever._get_reference_date()
            reference_date_str = reference_date.strftime('%Y-%m-%d') if reference_date else "unknown"
            
            prompt_template += f"""
            
            IMPORTANT TEMPORAL CONTEXT:
            - This criterion has a time constraint. Pay special attention to dates and timing.
            - The clinical record uses synthetic dates where 2-digit years should be interpreted as 2XXX (e.g., "2/16/51" = "2051-02-16" or "2051-02-16 depending on the case, not "1951-02-16").
            - The reference date (most recent clinical note) is: {reference_date_str}. This is our present moment. 
            - When evaluating time-based criteria, calculate time intervals from events to this reference date.
            - Example: If the reference date is 2151-04-11 and an event occurred on "2/16/51" (2151-02-16), that's about 2 months prior, which IS within "the past 6 months".
            """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["criterion", "relevant_context"]
        )
        
        formatted_prompt = prompt.format(
            criterion=criterion,
            relevant_context=relevant_context
        )
        
     
        response = None if batch_prep else get_llm().invoke(formatted_prompt)
        # Append structured prompt logging (and response if present)
        if verbose_dir:
            try:
                os.makedirs(verbose_dir, exist_ok=True)
            except Exception:
                pass
            try:
                prompt_log_path = os.path.join(verbose_dir, "prompts.jsonl")
                prompt_log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "patient_id": patient_id,
                    "criterion_index": i + 1,
                    "criterion": criterion,
                    "tag": map_criterion_to_tag(criterion),
                    "prompt": formatted_prompt,
                    "analysis": (response.content if response else None),
                    "retrieval_method": search_result.get('retrieval_method'),
                    "used_keywords": used_keywords,
                    "reference_date": reference_date.isoformat() if isinstance(reference_date, datetime) else None,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "k_value": k_value,
                    "use_mmr": use_mmr,
                    "lambda_mult": (lambda_mult if use_mmr else None),
                    "date_aware": True,
                    "batch_prep": batch_prep,
                    "model": CURRENT_MODEL,
                }
                with open(prompt_log_path, 'a') as pf:
                    pf.write(json.dumps(prompt_log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.warning(f"Failed to write prompt log: {e}")
        if response:
            logging.info(f"LLM ANALYSIS FOR CRITERION {i+1}:\n{response.content}")
        
        # Save verbose logs if requested
        if verbose_dir:
            criterion_filename = criterion[:40].replace(" ", "_").replace("/", "_")
            file_prefix = "temporal_" if criterion in temporal_criteria else "standard_"
            with open(os.path.join(verbose_dir, f"{file_prefix}check_{criterion_filename}.txt"), "w") as f:
                f.write(f"CRITERION: {criterion}\n")
                f.write(f"TEMPORAL: {criterion in temporal_criteria}\n\n")
                f.write(f"PROMPT:\n{formatted_prompt}\n\n")
                if response:
                    f.write(f"RESPONSE:\n{response.content}\n\n")
        
        results.append({
            "criterion": criterion,
            "analysis": response.content if response else "",
            "relevant_context": relevant_context,
            "retrieval_chunks": retrieval_chunks,  # Add the detailed chunks for GUI
            "is_non_clinical": False,
            "is_temporal": criterion in temporal_criteria,
            "reference_date": reference_date.strftime('%Y-%m-%d') if reference_date else None,
            "used_keywords": used_keywords,
            "prompt": formatted_prompt,
            "tag": map_criterion_to_tag(criterion)
        })
    
    final_analysis = format_final_results(results)
    return final_analysis, results, vectorstore

# =================== END DATE-AWARE ALTERNATIVES ===================

# Simple in-memory cache for criterion-specific keyword expansion
_keyword_cache: Dict[str, List[str]] = {}

def get_or_create_keywords_for_abdominal() -> List[str]:
    """Return an extensive, curated keyword set for abdominal surgery/obstruction (cached for practicality, initally generated with ChatGPT)"""
    key = "ABDOMINAL_KEYWORDS"
    if key in _keyword_cache:
        return _keyword_cache[key]

    keywords: List[str] = [
        # Surgical approaches
        "laparotomy", "laparoscopic", "laparoscopy", "open abdomen",
        # Resections and bowel terms
        "bowel resection", "small bowel resection", "large bowel resection", "enterectomy", "colectomy",
        "hemicolectomy", "sigmoidectomy", "ileectomy", "ileal resection", "ileocecectomy", "subtotal colectomy",
        "total colectomy", "proctocolectomy", "anastomosis", "re-anastomosis",
        # Ostomies
        "ostomy", "stoma", "ileostomy", "colostomy", "reversal of ostomy",
        # Adhesions and related
        "adhesions", "adhesion", "adhesiolysis", "lysis of adhesions", "adhesive disease",
        # Obstruction variants
        "sbo", "small bowel obstruction", "large bowel obstruction", "intestinal obstruction", "ileus",
        "volvulus", "stricture", "transition point", "obstructive pattern",
        # Common abdominal operations that imply prior surgery
        "appendectomy", "cholecystectomy", "gastrectomy", "duodenectomy", "pancreatectomy", "whipple",
        "hernia repair", "inguinal hernia repair", "ventral hernia repair", "umbilical hernia repair",
        # Bariatric
        "roux-en-y", "rygb", "gastric bypass", "sleeve gastrectomy", "lap band",
        # Imaging and radiographic signs
        "air-fluid levels", "dilated loops", "string of pearls", "step-ladder pattern",
        "kueb", "kub", "abdominal x-ray", "obstruction series", "ct abdomen", "ct a/p", "ct abdo pelvis",
        # Supportive/management terms
        "nasogastric tube", "ng tube", "npo", "bowel rest",
        # Ischemia/complications
        "mesenteric ischemia", "strangulation", "closed-loop obstruction",
    ]

    # Normalize, de-duplicate, and keep order
    seen: Set[str] = set()
    cleaned: List[str] = []
    for term in keywords:
        t = term.lower().strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            cleaned.append(t)

    _keyword_cache[key] = cleaned
    return cleaned

# Map criterion text to tag
CRITERIA_TO_TAG = {
    "Drug abuse, current or past": "DRUG-ABUSE",
    "Current alcohol use over weekly recommended limits": "ALCOHOL-ABUSE",
    "Patient must speak English": "ENGLISH",
    "Patient must make their own medical decisions": "MAKES-DECISIONS",
    "History of intra abdominal surgery, small or large intestine resection or small bowel obstruction": "ABDOMINAL",
    "Major diabetes-related complication. Major complication as opposed to minor complication for this purpose is defined as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: Amputation, Kidney damage, Skin conditions, Retinopathy, nephropathy, neuropathy": "MAJOR-DIABETES",
    "Advanced cardiovascular disease, as defined by having 2 or more of the following: Taking 2 or more medications to treat CAD, History of myocardial infarction (MI), Currently experiencing angina, Ischemia, past or present": "ADVANCED-CAD",
    "Myocardial infarction in the past 6 months": "MI-6MOS",
    "Diagnosis of ketoacidosis in the past year": "KETO-1YR",
    "Taken a dietary supplement (excluding Vitamin D) in the past 2 months": "DIETSUPP-2MOS",
    "Use of aspirin to prevent myocardial infarction": "ASP-FOR-MI",
    "Any HbA1c value between 6.5 and 9.5%": "HBA1C",
    "Serum creatinine > upper limit of normal": "CREATININE"
}

def map_criterion_to_tag(criterion: str) -> Optional[str]:
    return CRITERIA_TO_TAG.get(criterion)
