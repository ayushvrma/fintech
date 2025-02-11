from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
import chromadb
from loguru import logger

class MemoryManager:
    def __init__(self, storage_dir: str = "trading_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB for semantic search of past conversations
        self.chroma_client = chromadb.Client()
        self.conversation_collection = self.chroma_client.create_collection(
            name="trading_conversations",
            metadata={"description": "Trading agent conversations and outcomes"}
        )
        
        # Initialize checkpoint storage
        self.checkpoints_file = self.storage_dir / "checkpoints.json"
        self.conversation_file = self.storage_dir / "conversations.json"
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage files if they don't exist"""
        if not self.checkpoints_file.exists():
            self.checkpoints_file.write_text(json.dumps({"checkpoints": []}))
        if not self.conversation_file.exists():
            self.conversation_file.write_text(json.dumps({"conversations": []}))

    def save_conversation(self, 
                        query: str, 
                        response: Dict, 
                        market_data: Dict,
                        strategies: Dict) -> str:
        """Save conversation with context and return conversation ID"""
        conversation_id = datetime.now().isoformat()
        
        conversation = {
            "id": conversation_id,
            "timestamp": conversation_id,
            "query": query,
            "response": response,
            "market_data": market_data,
            "strategies": strategies
        }
        
        # Save to file
        conversations = json.loads(self.conversation_file.read_text())
        conversations["conversations"].append(conversation)
        self.conversation_file.write_text(json.dumps(conversations, indent=2))
        
        # Add to ChromaDB for semantic search
        self.conversation_collection.add(
            documents=[json.dumps({"query": query, "response": response})],
            metadatas=[{"timestamp": conversation_id}],
            ids=[conversation_id]
        )
        
        return conversation_id

    def create_checkpoint(self, 
                         conversation_id: str, 
                         state: Dict,
                         description: str) -> str:
        """Create a checkpoint of the current trading state"""
        checkpoint_id = f"checkpoint_{datetime.now().isoformat()}"
        
        checkpoint = {
            "id": checkpoint_id,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "description": description
        }
        
        checkpoints = json.loads(self.checkpoints_file.read_text())
        checkpoints["checkpoints"].append(checkpoint)
        self.checkpoints_file.write_text(json.dumps(checkpoints, indent=2))
        
        return checkpoint_id

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations"""
        conversations = json.loads(self.conversation_file.read_text())
        return conversations["conversations"][-limit:]

    def search_conversations(self, query: str, limit: int = 5) -> List[Dict]:
        """Search conversations semantically"""
        results = self.conversation_collection.query(
            query_texts=[query],
            n_results=limit
        )
        return results

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Retrieve a specific checkpoint"""
        checkpoints = json.loads(self.checkpoints_file.read_text())
        for checkpoint in checkpoints["checkpoints"]:
            if checkpoint["id"] == checkpoint_id:
                return checkpoint
        return None

    def get_latest_checkpoint(self) -> Optional[Dict]:
        """Get the most recent checkpoint"""
        checkpoints = json.loads(self.checkpoints_file.read_text())
        if checkpoints["checkpoints"]:
            return checkpoints["checkpoints"][-1]
        return None 