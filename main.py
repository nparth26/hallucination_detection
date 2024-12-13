from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import json
import autogen
import uvicorn
from pydantic import BaseModel
from typing import List, Dict
import os
import regex as re

#agent classes
base_config = [
    {
        # Let's choose the Mixtral 8x7B model
        # "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "model": "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
        # Provide your Together.AI API key here or put it into the TOGETHER_API_KEY environment variable.
        "api_key": 'Add key here',
        # We specify the API Type as 'together' so it uses the Together.AI client class
        "api_type": "together",
        "stream": False,
    }
]

llm_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": base_config,
    "temperature": 0.7,
}

class ConceptExtractionAgent(autogen.AssistantAgent):
    def __init__(self): 
        super().__init__(
            name="concept_extractor",
            system_message="You are a concept extraction specialist. Your task is to identify and extract key concepts from input queries.",
            llm_config=llm_config
        )
    
    def process_concepts(self, text):
        concepts = self._extract_concepts_llm(text)
        # filtered_concepts = self._filter_common_concepts(concepts)
        # grouped_concepts = self._group_concepts(filtered_concepts)
        # return grouped_concepts
        return concepts
    
    def _extract_concepts_llm(self, text):
        extraction_prompt = f"""Extract only the important key concepts and named entities from the following text. 
For each concept provide:
- The concept word or phrase
- Its position (start and end index) in the text
- The type of entity (e.g., TECH, MEDICAL, PERSON, ORG, etc.)

Text: {text}

Format your response as a list of JSON objects, one per line:
{{"word": "concept", "start": "start_idx", "end": "end_idx", "entity": "type"}}"""

        # Create a human agent for sending the message
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            max_consecutive_auto_reply=0,
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,  # Enable Docker execution
                "timeout": 60,
            }
        )
        
        # Send message and get response
        user_proxy.initiate_chat(
            self,
            
            message=extraction_prompt
        )
        
        # Get the last message from the conversation
        response = self.last_message()["content"]

        
        try:
          concepts = []
          for line in response.strip().split('\n'):
              
              if line and line.strip().startswith('{'):
                line = line.strip()
                # print('hii')
                # print(line)
                concept = json.loads(line.strip(','))
                concepts.append(concept)
          return concepts
        except:
            return []
        
class ConceptDefinitionAgent(autogen.AssistantAgent):
    def __init__(self):
        super().__init__(
            name="concept_definer",
            system_message="You are a concept definition specialist. Your task is to explain concepts clearly without using the original concept terms.",
            llm_config=llm_config
        )
    
    def process_definitions(self, concepts):
        definitions = []
        for concept in concepts:
            # Get definition and mask the concept
            definition = self._get_masked_definition(concept)
            definitions.append({
                "concept": concept["word"],
                "masked_definition": definition
            })
        return definitions
    
    def _get_masked_definition(self, concept):
        definition_prompt = f"""Explain the following concept in short without using the concept term itself or its close synonyms:
Concept: {concept['word']}

Provide a clear, short explanation that would allow someone to understand and identify the concept."""
        
        # Create user proxy for interaction
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            max_consecutive_auto_reply=0,
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,  # Enable Docker execution
                "timeout": 60,
            }
        )
        
        # Get definition
        user_proxy.initiate_chat(
            self,
            message=definition_prompt
        )
        
        # Extract definition from response
        definition = self.last_message(user_proxy)["content"]
        
        # Mask any remaining instances of the concept
        masked_definition = self._mask_concept(definition, concept["word"])
        
        return masked_definition
    
    def _mask_concept(self, text, concept):
        # Simple masking by replacing concept with [MASK]
        # Case insensitive replacement
        pattern = re.compile(re.escape(concept), re.IGNORECASE)
        masked_text = pattern.sub("[MASK]", text)
        return masked_text

class ConceptInferenceAgent(autogen.AssistantAgent):
    def __init__(self):
        super().__init__(
            name="concept_inferrer",
            system_message="You are a concept inference specialist. Your task is to infer original concepts from masked definitions.",
            llm_config=llm_config
        )
    
    def infer_concepts(self, masked_definitions):
        inferred_concepts = []
        for definition in masked_definitions:
            inference = self._infer_concept(definition)
            inferred_concepts.append({
                "original_concept": definition["concept"],
                "inferred_concept": inference
            })
        return inferred_concepts
    
    def _infer_concept(self, definition):
        inference_prompt = f"""Based on the following masked definition, infer what concept is being described:

Definition: {definition['masked_definition']}

Provide a single specific concept that best matches this definition. Only provide the name of the concept in the output and not the explanation"""

        # Create user proxy for interaction
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            max_consecutive_auto_reply=0,
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,  # Enable Docker execution
                "timeout": 60,
            }
        )
        
        # Get inference
        user_proxy.initiate_chat(
            self,
            message=inference_prompt
        )
        
        # Extract inferred concept
        inferred_concept = self.last_message(user_proxy)["content"].strip()
        return inferred_concept

class SimilarityScoreAgent(autogen.AssistantAgent):
    def __init__(self):
        super().__init__(
            name="similarity_scorer",
            system_message="You are a similarity scoring specialist. Your task is to evaluate the similarity between original and inferred concepts and return ONLY a JSON response.",
            llm_config=llm_config
        )
    
    def calculate_similarity(self, concept_pairs):
        similarity_scores = []
        for pair in concept_pairs:
            score = self._get_similarity_score(pair)
            similarity_scores.append({
                "original_concept": pair["original_concept"],
                "inferred_concept": pair["inferred_concept"],
                "similarity_score": score
            })
        return similarity_scores
    
    def _get_similarity_score(self, concept_pair):
        scoring_prompt = f"""Rate the similarity between these concepts' words on a scale of 1 to 10. Most importantly give more weightage to name similarity and little less weightage to semantic similarity:

Original Concept: {concept_pair['original_concept']}
Inferred Concept: {concept_pair['inferred_concept']}

Respond ONLY with a JSON object in this exact format:
{{"similarity_score": <number between 1-10>}}

"""

        # Create user proxy for interaction
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            max_consecutive_auto_reply=0,
            human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,  # Enable Docker execution
                "timeout": 60,
            }
        )
        
        # Get similarity score
        user_proxy.initiate_chat(
            self,
            message=scoring_prompt
        )
        
        # Extract score from JSON response
        response = self.last_message(user_proxy)["content"].strip()
        return self._extract_score(response)
    
    def _extract_score(self, response):
        try:
            # Find the JSON object in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                score_dict = json.loads(json_str)
                score = float(score_dict.get('similarity_score', 1))
                return min(max(score, 1), 10)  # Ensure score is between 1-10
        except:
            pass
        return 1  # Default to lowest score if parsing fails
       

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure API endpoints



@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Define request model
class QueryRequest(BaseModel):
    query: str

@app.post("/analyze")
async def analyze_query(request: QueryRequest):
    # print('hii')
    extractor = ConceptExtractionAgent()
    concepts = extractor.process_concepts(request.query)
    
    # Simplify concepts to only include words
    simplified_concepts = [concept["word"] for concept in concepts]
    # simplified_concepts = 'abc'
    return JSONResponse(content={"concepts": simplified_concepts})

# Request/Response Models
class ConceptResponse(BaseModel):
    concepts: List[str]

class DefinitionResponse(BaseModel):
    concept: str
    masked_definition: str

class InferenceResponse(BaseModel):
    original_concept: str
    inferred_concept: str

class SimilarityResponse(BaseModel):
    original_concept: str
    inferred_concept: str
    similarity_score: float

@app.post("/process_concepts")
async def process_concepts(query: QueryRequest):
    try:
        extractor = ConceptExtractionAgent()
        concepts = extractor.process_concepts(query.query)
        simplified_concepts = [concept["word"] for concept in concepts]
        return ConceptResponse(concepts=simplified_concepts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_definitions")
async def get_definitions(concepts: List[str]):
    try:
        definer = ConceptDefinitionAgent()
        # Convert concepts to format expected by definer
        concept_dicts = [{"word": c} for c in concepts]
        definitions = definer.process_definitions(concept_dicts)
        return [DefinitionResponse(**d) for d in definitions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer_concepts")
async def infer_concepts(definitions: List[DefinitionResponse]):
    try:
        inferrer = ConceptInferenceAgent()
        inferences = inferrer.infer_concepts(definitions)
        return [InferenceResponse(**i) for i in inferences]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_similarity")
async def calculate_similarity(inferences: List[InferenceResponse]):
    try:
        scorer = SimilarityScoreAgent()
        scores = scorer.calculate_similarity(inferences)
        return [SimilarityResponse(**s) for s in scores]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_complete")
async def analyze_complete(query: QueryRequest):
    try:
        # Initialize agents
        extractor = ConceptExtractionAgent()
        definer = ConceptDefinitionAgent()
        inferrer = ConceptInferenceAgent()
        scorer = SimilarityScoreAgent()
        
        # Complete pipeline
        concepts = extractor.process_concepts(query.query)
        simplified_concepts = [{"word": concept["word"]} for concept in concepts]
        definitions = definer.process_definitions(simplified_concepts)
        inferences = inferrer.infer_concepts(definitions)
        scores = scorer.calculate_similarity(inferences)
        
        return {
            "concepts": [c["word"] for c in concepts],
            "definitions": definitions,
            "inferences": inferences,
            "similarity_scores": scores
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the FastAPI application server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()