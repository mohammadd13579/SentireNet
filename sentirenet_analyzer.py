import pandas as pd
import spacy
import spacy.cli
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
import streamlit as st

# --- ROBUST SPACY LOADING LOGIC ---
@st.cache_resource
def load_spacy_model():
    try:
        # 1. Try loading the standard model
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            # 2. If missing, try downloading it on the fly
            print("Downloading 'en_core_web_sm' model...")
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            # 3. Fallback: Use a blank model with a rule-based sentence splitter
            print(f"Model download failed: {e}. Using fallback rule-based pipeline.")
            from spacy.lang.en import English
            nlp = English()
            nlp.add_pipe('sentencizer') 
            return nlp

nlp = load_spacy_model()

# Load Sentence Transformer
@st.cache_resource
def load_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_transformer()

class SentireNetAnalyzer:
    """
    Handles processing text, building the semantic/emotional graph, 
    and simulating rhetorical pathways.
    """
    def __init__(self, df, text_column):
        self.df = df
        self.text_column = text_column
        self.graph = nx.Graph()
        
    def _get_sentiment(self, text):
        """Returns polarity (-1 to 1)."""
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def extract_concepts(self, text):
        """
        Extracts key concepts (nodes) from text.
        Tries to use Noun Chunks (better), falls back to Keywords if parser is missing.
        """
        doc = nlp(text)
        concepts = []
        
        # Check if the model has a parser/tagger (required for noun_chunks)
        has_parser = "parser" in nlp.pipe_names or "tagger" in nlp.pipe_names
        
        if has_parser:
            # Prefer noun chunks (e.g., "artificial intelligence")
            for chunk in doc.noun_chunks:
                clean_chunk = chunk.text.lower().strip()
                if len(clean_chunk) > 2 and clean_chunk not in nlp.Defaults.stop_words:
                    concepts.append(clean_chunk)
        
        # Fallback or supplemental: simple tokens if noun chunks returned nothing
        if not concepts:
            for token in doc:
                if token.is_alpha and not token.is_stop and len(token.text) > 2:
                    concepts.append(token.text.lower())
                    
        return list(set(concepts)) # Deduplicate

    def build_graph(self, similarity_threshold=0.4):
        """
        Main function to build the network.
        """
        all_concepts = []
        
        # 1. Extract Concepts per document
        for text in self.df[self.text_column].dropna():
            concepts = self.extract_concepts(str(text))
            all_concepts.extend(concepts)
            
        # Deduplicate concepts for the graph nodes
        unique_concepts = list(set(all_concepts))
        
        if not unique_concepts:
            return self.graph
            
        # 2. Generate Embeddings for all unique concepts
        embeddings = model.encode(unique_concepts)
        
        # 3. Add Nodes with Sentiment
        for i, concept in enumerate(unique_concepts):
            self.graph.add_node(
                concept, 
                polarity=self._get_sentiment(concept),
                count=all_concepts.count(concept) # Simple frequency
            )
            
        # 4. Create Edges based on Semantic Similarity
        sim_matrix = cosine_similarity(embeddings)
        
        # Iterate through upper triangle of matrix to find connections
        for i in range(len(unique_concepts)):
            for j in range(i + 1, len(unique_concepts)):
                score = sim_matrix[i][j]
                if score > similarity_threshold:
                    self.graph.add_edge(
                        unique_concepts[i],
                        unique_concepts[j],
                        semantic_similarity=score
                    )
                    
        return self.graph

    def simulate_pathway(self, start_idea, steps=5, path_count=3):
        """
        Simulates how a new idea traverses the existing graph.
        """
        if len(self.graph.nodes) == 0:
            return [], "Graph is empty."

        # 1. Embed the start idea
        start_embedding = model.encode([start_idea])
        
        # 2. Find the closest existing node to enter the graph
        node_list = list(self.graph.nodes())
        node_embeddings = model.encode(node_list)
        
        sims = cosine_similarity(start_embedding, node_embeddings)[0]
        start_node_index = np.argmax(sims)
        start_node = node_list[start_node_index]
        
        all_paths = []

        for _ in range(path_count):
            current_node = start_node
            path = [current_node]
            
            for _ in range(steps):
                neighbors = list(self.graph.neighbors(current_node))
                if not neighbors:
                    break
                
                # Get weights
                weights = np.array([self.graph[current_node][neighbor]['semantic_similarity'] for neighbor in neighbors])
                
                # Normalize weights robustly
                if weights.sum() == 0:
                    # If all weights are zero (unlikely), break or choose uniformly
                    break
                
                probs = weights / weights.sum()
                
                # FIX: Renormalize to ensure sum is exactly 1.0 to prevent numpy error
                probs = probs / probs.sum()
                
                next_node = np.random.choice(neighbors, p=probs)
                path.append(next_node)
                current_node = next_node
            
            all_paths.append(path)

        # Generate Summary
        summary = f"Simulation Input: '{start_idea}'\nEntry Point: '{start_node}' (Similarity: {sims[start_node_index]:.2f})\n\n"
        for i, p in enumerate(all_paths):
            path_str = " -> ".join(p)
            end_node = p[-1]
            pol = self.graph.nodes[end_node].get('polarity', 0)
            mood = "Positive" if pol > 0.1 else "Negative" if pol < -0.1 else "Neutral"
            summary += f"Path {i+1}: {path_str} (Ends in {mood} cluster)\n"
            
        return all_paths, summary