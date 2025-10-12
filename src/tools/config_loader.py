"""
Configuration loader for city profiles and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """Load and manage configuration from YAML files and environment."""
    
    CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
    
    @classmethod
    def load_city_profile(cls, profile_name: str = "dense-city") -> Dict[str, Any]:
        """
        Load a city profile configuration.
        
        Args:
            profile_name: Name of the profile (dense-city, suburban, rural)
            
        Returns:
            Dictionary with configuration values
            
        Raises:
            FileNotFoundError: If profile doesn't exist
        """
        profile_path = cls.CONFIG_DIR / f"{profile_name}.yaml"
        
        if not profile_path.exists():
            available = [f.stem for f in cls.CONFIG_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"Profile '{profile_name}' not found. Available profiles: {', '.join(available)}"
            )
        
        with open(profile_path, "r") as f:
            return yaml.safe_load(f)
    
    @classmethod
    def get_profile_from_env(cls) -> Optional[str]:
        """Get city profile name from CITY_PROFILE environment variable."""
        return os.getenv("CITY_PROFILE")
    
    @classmethod
    def load_default_or_env_profile(cls) -> Dict[str, Any]:
        """
        Load city profile from environment variable or use dense-city default.
        
        Returns:
            Configuration dictionary
        """
        profile = cls.get_profile_from_env() or "dense-city"
        return cls.load_city_profile(profile)


def get_config() -> Dict[str, Any]:
    """Convenience function to get current configuration."""
    return ConfigLoader.load_default_or_env_profile()
