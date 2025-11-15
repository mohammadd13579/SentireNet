import pandas as pd
import spacy
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
import re

# Try loading a smaller, faster spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
    # As a fallback, create a blank model (this will limit entity extraction)
    from spacy.lang.en import English
    nlp = English()
    if 'parser' not in nlp.pipe_names:
        nlp.add_pipe('parser', first=True)
    if 'senter' not in nlp.pipe_names:
        nlp.add_pipe('senter', first=True)


# Load the sentence transformer model
# Using a lighter model for faster performance in a demo app
model = SentenceTransformer('all-MiniLM-L6-v2')

class SentireNetAnalyzer:
    """
    This class handles the core logic of processing text, building the semantic
    and emotional graph, and simulating rhetorical pathways.
    """
    def __init__(self, df, text_column):
        self.df = df
        self.text_column = text_column
        self.graph = nx.Graph()
        self.concepts = {} # Stores data about each concept (node)

    def _get_sentiment(self, text):
        """Returns polarity (float -1 to 1) and subjectivity (float 0 to 1)."""
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def _extract_concepts(self, max_concepts=100, min_freq=2):
        """
        Extracts key concepts (nodes) from the text data using NLP.
        We'll use a simple approach: extract nouns, proper nouns, and adjectives.
        """
        concept_docs = {} # {concept: {'count': int, 'original_texts': [str]}}
        
        # First, drop any rows that are *actually* empty
        text_series = self.df[self.text_column].dropna()
        
        # Second, convert everything to a string to handle numeric columns gracefully
        for text in text_series.astype(str):
            try:
                doc = nlp(text)
                
                # Extract entities (like 'AI model', 'artists')
                concepts = [ent.text for ent in doc.ents if ent.label_ in ('NOUN', 'PROPN', 'ORG', 'PRODUCT', 'PERSON')]
                
                # Fallback: extract key nouns and adjectives if no entities found
                if not concepts:
                    concepts = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and not token.is_stop]
                
                # Clean up and add to count
                for concept in set(concepts): # Use set to count once per doc
                    concept_text = concept.lower().strip()
                    if len(concept_text) > 2: # Ignore short/stray tokens
                        if concept_text not in concept_docs:
                            concept_docs[concept_text] = {'count': 0, 'original_texts': []}
                        concept_docs[concept_text]['count'] += 1
                        concept_docs[concept_text]['original_texts'].append(text)
            
            except Exception as e:
                print(f"Error processing text: {text} | Error: {e}")
                # This could happen on very weird input, skip the doc
                continue

        if not concept_docs:
            return pd.DataFrame() # Return empty if no concepts found

        # Filter concepts by frequency
        filtered_concepts = {k: v for k, v in concept_docs.items() if v['count'] >= min_freq}
        
        if not filtered_concepts:
            return pd.DataFrame() # Return empty if no concepts meet freq

        # Convert to DataFrame for easier handling
        concepts_df = pd.DataFrame.from_dict(filtered_concepts, orient='index')
        
        # Limit to max_concepts
        if len(concepts_df) > max_concepts:
            concepts_df = concepts_df.nlargest(max_concepts, 'count')
            
        return concepts_df

    def build_graph(self, max_concepts=100, min_freq=2, min_similarity=0.3):
        """
        Builds the semantic and emotional graph.
        1. Extracts concepts (nodes)
        2. Calculates sentiment for each node
        3. Calculates semantic similarity (edges)
        """
        self.graph = nx.Graph()
        self.concepts = {}
        
        # 1. Extract concepts (nodes)
        concepts_df = self._extract_concepts(max_concepts, min_freq)
        if concepts_df.empty:
            print("No concepts extracted. Graph will be empty.")
            return self.graph

        concept_list = concepts_df.index.tolist()
        
        # 2. Get embeddings and sentiment for each concept
        embeddings = model.encode(concept_list)
        
        for i, concept in enumerate(concept_list):
            data = concepts_df.loc[concept]
            
            # Get average sentiment for the concept based on its first appearance
            # (Averaging all would be slow; this is a good approximation)
            try:
                # Use the *string-converted* original text
                polarity, subjectivity = self._get_sentiment(str(data['original_texts'][0]))
            except Exception as e:
                print(f"Error getting sentiment for {concept}: {e}")
                polarity, subjectivity = 0.0, 0.0

            self.concepts[concept] = {
                'count': data['count'],
                'embedding': embeddings[i],
                'polarity': polarity,
                'subjectivity': subjectivity
            }
            
            # Add node to graph
            self.graph.add_node(
                concept,
                count=data['count'],
                polarity=polarity,
                subjectivity=subjectivity,
                size=5 + (data['count'] * 1.5) # For visualization
            )

        # 3. Calculate semantic similarity (edges)
        if len(concept_list) < 2:
            return self.graph # Not enough nodes to make edges

        sim_matrix = cosine_similarity(embeddings)
        
        for i in range(len(concept_list)):
            for j in range(i + 1, len(concept_list)):
                similarity = sim_matrix[i][j]
                
                if similarity >= min_similarity:
                    node_i = concept_list[i]
                    node_j = concept_list[j]
                    
                    # Calculate emotional resonance (how similar are their sentiments?)
                    pol_i = self.concepts[node_i]['polarity']
                    pol_j = self.concepts[node_j]['polarity']
                    # Simple resonance: 1 is identical, 0 is opposite polarity
                    emotional_resonance = 1 - (abs(pol_i - pol_j) / 2) 
                    
                    self.graph.add_edge(
                        node_i,
                        node_j,
                        semantic_similarity=float(similarity),
                        emotional_resonance=float(emotional_resonance),
                        weight=(float(similarity) * 0.7) + (float(emotional_resonance) * 0.3)
                    )
        
        return self.graph

    def _find_start_node(self, start_idea):
        """Finds the concept node in the graph that is semantically closest to the user's start_idea."""
        if not self.graph.nodes:
            return None

        idea_embedding = model.encode([start_idea])[0]
        
        nodes = list(self.concepts.keys())
        if not nodes:
            return None
            
        node_embeddings = np.array([self.concepts[n]['embedding'] for n in nodes])
        
        # Calculate similarities
        sims = cosine_similarity([idea_embedding], node_embeddings)[0]
        
        # Find the node with the highest similarity
        best_match_index = np.argmax(sims)
        
        return nodes[best_match_index]

    def simulate_pathway(self, start_idea, steps=5, path_count=3):
        """
        Simulates the "path of least resistance" an idea would take through the graph.
        Uses a weighted random walk.
        """
        if not self.graph.nodes:
            return [], "Graph is not built. Please build the graph first."

        start_node = self._find_start_node(start_idea)
        if start_node is None:
            return [], "Could not find a matching concept in the graph to start from."

        all_paths = []
        
        for _ in range(path_count):
            path = [start_node]
            current_node = start_node
            
            for _ in range(steps - 1): # -1 because start_node is already one step
                neighbors = list(self.graph.neighbors(current_node))
                
                if not neighbors:
                    break # Dead end

                # Get weights for all neighbors
                weights = []
                for n in neighbors:
                    # Use the combined 'weight' (semantic + emotional)
                    weight = self.graph.edges[current_node, n].get('weight', 0)
                    weights.append(weight)
                
                if not weights:
                    break # No valid neighbors
                
                # Normalize weights to create a probability distribution
                weights_sum = sum(weights)
                if weights_sum == 0:
                    break # All weights are zero
                
                probabilities = [w / weights_sum for w in weights]
                
                # Choose the next node based on the weighted probabilities
                next_node = np.random.choice(neighbors, p=probabilities)
                path.append(next_node)
                current_node = next_node
                
            all_paths.append(path)

        # Generate a textual summary of the simulation
        summary = f"Simulating rhetorical pathways for: '{start_idea}' (starting at concept: '{start_node}')\n\n"
        summary += f"Found {len(all_paths)} likely paths:\n"
        for i, path in enumerate(all_paths):
            path_str = " -> ".join(path)
            summary += f"  Path {i+1}: {path_str}\n"
            
            final_node = path[-1]
            final_data = self.graph.nodes[final_node]
            polarity = final_data['polarity']
            
            if polarity > 0.2:
                emotion = "a positive cluster."
            elif polarity < -0.2:
                emotion = "a negative cluster."
            else:
                emotion = "a neutral/ambivalent cluster."
                
            summary += f"    ...ending in '{final_node}', which is part of {emotion}\n"

        return all_paths, summary