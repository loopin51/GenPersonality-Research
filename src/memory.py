import chromadb
from chromadb.utils import embedding_functions
import uuid
import numpy as np
from typing import List, Dict, Any
from src.models import MemoryItem
from src.config import VECTOR_DB_PATH, OPENROUTER_API_KEY, LLM_BASE_URL, MEMORY_RETRIEVAL_K

class MemorySystem:
    def __init__(self, collection_name: str = "child_memory"):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        
        # Use OpenAI Embedding via OpenRouter (if supported) or behave like OpenAI
        # Assuming OpenRouter proxies text-embedding-3-small or user has setup
        # Note: ChromaDB's OpenAIEmbeddingFunction might need api_base
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENROUTER_API_KEY,
            api_base=LLM_BASE_URL,
            model_name="text-embedding-3-small"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_memory(self, item: MemoryItem):
        """
        Saves an episodic memory.
        The document content will be a combination of trigger + action + outcome.
        Metadata stores the structured info including emotion impact.
        """
        # Create a rich textual representation for embedding
        text_representation = f"Trigger: {item.trigger} | Action: {item.action} | Outcome: {item.outcome}"
        
        # Metadata must be simple types
        metadata = {
            "episode_id": item.episode_id,
            "trigger": item.trigger,
            "action": item.action,
            "outcome": item.outcome,
            "delta_p": item.emotion_impact[0],
            "delta_a": item.emotion_impact[1],
            "delta_d": item.emotion_impact[2],
            "timestamp": item.timestamp
        }
        
        self.collection.add(
            documents=[text_representation],
            metadatas=[metadata],
            ids=[item.episode_id]
        )

    def retrieve(self, query: str, k: int = MEMORY_RETRIEVAL_K, emotion_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Weighted Retrieval: Similarity + Emotional Intensity.
        Project_receipt.md 3.2: Score = w_sim * Sim + w_emo * |Delta E|
        
        Since Chroma primarily does semantic search, we will:
        1. Fetch 2*K candidates via semantic search.
        2. Re-rank them based on the formula.
        3. Return top K.
        """
        # 1. Fetch Candidates (Fetch more than needed to re-rank)
        results = self.collection.query(
            query_texts=[query],
            n_results=k * 2
        )
        
        if not results['documents']:
            return []

        # Parse results into a list of dicts for easier handling
        candidates = []
        ids = results['ids'][0]
        distances = results['distances'][0] # Cosine distance (lower is better, 0=identical)
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        
        for i in range(len(ids)):
            # Convert distance to similarity (approximate 1 - distance/2 for cosine, restricted to 0-1)
            # Chroma returns distance. For cosine, sim = 1 - dist.
            similarity = 1 - distances[i]
            
            # Calculate Emotional Intensity (L2 Norm of Delta E)
            delta_e = np.array([
                metadatas[i]['delta_p'],
                metadatas[i]['delta_a'],
                metadatas[i]['delta_d']
            ])
            intensity = np.linalg.norm(delta_e)
            
            # Weighted Score (Simple linear combination)
            # Note: receipt says Score = w_sim * Sim + w_emo * |Delta E|
            # We assume w_sim = 1.0 for base, emotion_weight passed as arg.
            score = similarity + (emotion_weight * intensity)
            
            candidates.append({
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
                "similarity": similarity,
                "intensity": intensity,
                "score": score
            })
            
        # 2. Re-rank
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. Return Top K
        return candidates[:k]

    def get_recent_memories(self, start_episode: int, end_episode: int) -> List[Dict[str, Any]]:
        """
        Retrieves memories for a specific range of episodes.
        Used for Reflection (Belief Formation).
        """
        # ChromaDB 'where' filter
        results = self.collection.get(
            where={
                "$and": [
                    {"timestamp": {"$gte": start_episode}},
                    {"timestamp": {"$lte": end_episode}}
                ]
            }
        )
        
        memories = []
        if results['ids']:
            for i in range(len(results['ids'])):
                memories.append({
                    "id": results['ids'][i],
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
                
        return memories

    def clear(self):
        self.client.delete_collection(self.collection.name)
