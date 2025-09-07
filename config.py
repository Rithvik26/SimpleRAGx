"""
Configuration management for Enhanced SimpleRAG
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Enhanced default configuration with Graph RAG options
DEFAULT_CONFIG = {
    "gemini_api_key": "",  # Remove os.environ.get
    "claude_api_key": "",  # Remove os.environ.get
    "qdrant_url": "",      # Remove hardcoded URL
    "qdrant_api_key": "",  # Remove os.environ.get
    "collection_name": "simple_rag_docs",  # Keep default name
    "graph_collection_name": "simple_rag_graph",
    "embedding_dimension": 768,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "preferred_llm": "claude",
    "rag_mode": "normal",
    "rate_limit": 60,
    "enable_cache": True,
    "cache_dir": None,
    # Graph RAG specific settings
    "max_entities_per_chunk": 20,
    "relationship_extraction_prompt": "extract_relationships",
    "graph_reasoning_depth": 2,
    "entity_similarity_threshold": 0.8,
    "graph_extraction_timeout": 45,
    "max_chunk_length_for_graph": 2000,
    "enable_agentic_ai": True,
    "agentic_max_iterations": 5,
    "agentic_temperature": 0.1,
     # Neo4j Configuration
    "neo4j_uri": "",
    "neo4j_username": "neo4j", 
    "neo4j_password": "",
    "neo4j_enabled": False,

    
}

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/tmp/simplerag_config.json")  # Use temp file instead

class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_path: str = None, force_fresh_start: bool = False):
        self.config_path = config_path or CONFIG_PATH
        self.force_fresh_start = force_fresh_start  # FIX: Initialize the missing attribute
        self.config = self._load_config()
    
    def _ensure_config_dir_exists(self):
        """Create config directory if it doesn't exist."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        self._ensure_config_dir_exists()
        
        # Start with defaults
        config = DEFAULT_CONFIG.copy()
        
        # NEW: Check for dev_config.json first (for development)
        dev_config_path = 'dev_config.json'
        dev_config = {}  # Initialize empty dict
        if os.path.exists(dev_config_path):
            try:
                with open(dev_config_path, 'r') as f:
                    dev_config = json.load(f)
                    # Merge dev config with existing config
                    config.update(dev_config)
                    # NEW: Mark setup as completed if we have dev config
                    config["setup_completed"] = True
                    logger.info("Loaded development configuration from dev_config.json")
            except Exception as e:
                logger.error(f"Error loading dev_config.json: {e}")
        
        # FIXED: Only load existing config if not forcing fresh start
        # and if setup was previously completed
        if not self.force_fresh_start and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    saved_config = json.load(f)
                    
                    # FIXED: Only merge if setup was completed before
                    if saved_config.get("setup_completed", False) or config.get("setup_completed", False):
                        # Don't overwrite dev_config values
                        for key, value in saved_config.items():
                            if key not in dev_config or key == "setup_completed":
                                config[key] = value
                        logger.info(f"Loaded configuration from {self.config_path}")
                    else:
                        logger.info("Previous setup incomplete, using defaults")
                        # Save fresh defaults
                        self._save_config(config)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                logger.info("Using default configuration")
                self._save_config(config)
        else:
            if self.force_fresh_start:
                logger.info("Force fresh start - using default configuration")
            else:
                logger.info(f"No config file found at {self.config_path}, using defaults")
            # Save the default config
            self._save_config(config)
        
        # Apply environment variable overrides if present
        self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]):
        """Apply environment variable overrides."""
        env_overrides = {
            "gemini_api_key": "GEMINI_API_KEY",
            "claude_api_key": "CLAUDE_API_KEY", 
            "qdrant_api_key": "QDRANT_API_KEY",
            "qdrant_url": "QDRANT_URL",
            "collection_name": "QDRANT_COLLECTION",
            "graph_collection_name": "QDRANT_GRAPH_COLLECTION"
        }
         # Neo4j overrides
        if os.getenv("NEO4J_URI"):
            config["neo4j_uri"] = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USERNAME"):
            config["neo4j_username"] = os.getenv("NEO4J_USERNAME")
        if os.getenv("NEO4J_PASSWORD"):
            config["neo4j_password"] = os.getenv("NEO4J_PASSWORD")
        if os.getenv("NEO4J_ENABLED"):
            config["neo4j_enabled"] = os.getenv("NEO4J_ENABLED").lower() == "true"
        for config_key, env_key in env_overrides.items():
            env_value = os.environ.get(env_key)
            if env_value:  # Only override if env var exists
                config[config_key] = env_value
                logger.info(f"Config override from env: {config_key}")
            
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            self._ensure_config_dir_exists()
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except IOError as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)
    
    def save(self):
        """Save current configuration to file."""
        self._save_config(self.config)
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required API keys
        required_keys = ["gemini_api_key"]
        for key in required_keys:
            if not self.config.get(key):
                validation_result["errors"].append(f"Missing required configuration: {key}")
                validation_result["valid"] = False
        
        # Check optional but recommended keys
        recommended_keys = ["claude_api_key", "qdrant_api_key", "qdrant_url"]
        for key in recommended_keys:
            if not self.config.get(key):
                validation_result["warnings"].append(f"Missing recommended configuration: {key}")
        
        # Validate numeric settings
        numeric_validations = {
            "chunk_size": (100, 5000),
            "chunk_overlap": (0, 1000),
            "top_k": (1, 50),
            "max_entities_per_chunk": (5, 100),
            "graph_reasoning_depth": (1, 10),
            "embedding_dimension": (100, 2000)
        }
        
        for key, (min_val, max_val) in numeric_validations.items():
            value = self.config.get(key)
            if value is not None:
                try:
                    value = int(value)
                    if value < min_val or value > max_val:
                        validation_result["warnings"].append(
                            f"{key} value {value} is outside recommended range [{min_val}, {max_val}]"
                        )
                except (ValueError, TypeError):
                    validation_result["errors"].append(f"Invalid {key}: must be a number")
                    validation_result["valid"] = False
        
        # Validate threshold settings
        threshold_validations = {
            "entity_similarity_threshold": (0.0, 1.0)
        }
        
        for key, (min_val, max_val) in threshold_validations.items():
            value = self.config.get(key)
            if value is not None:
                try:
                    value = float(value)
                    if value < min_val or value > max_val:
                        validation_result["warnings"].append(
                            f"{key} value {value} is outside valid range [{min_val}, {max_val}]"
                        )
                except (ValueError, TypeError):
                    validation_result["errors"].append(f"Invalid {key}: must be a number")
                    validation_result["valid"] = False
        
        # Validate mode settings
        if self.config.get("rag_mode") not in ["normal", "graph", "neo4j", "hybrid_neo4j"]:
            validation_result["errors"].append("rag_mode must be 'normal', 'graph', 'neo4j', or 'hybrid_neo4j'")
            validation_result["valid"] = False
        
        if self.config.get("preferred_llm") not in ["claude", "raw"]:
            validation_result["errors"].append("preferred_llm must be 'claude' or 'raw'")
            validation_result["valid"] = False
        
        return validation_result
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self.config.copy()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = DEFAULT_CONFIG.copy()
        self._apply_env_overrides(self.config)

# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def load_config() -> Dict[str, Any]:
    """Load configuration (backward compatibility)."""
    return get_config_manager().get_all()

def save_config(config: Dict[str, Any]):
    """Save configuration (backward compatibility)."""
    manager = get_config_manager()
    manager.config = config
    manager.save()