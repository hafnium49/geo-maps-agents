"""
Scoring Weights and A/B Testing Module

Manages scoring weight configurations and A/B variant selection.
Implements session-sticky variant chooser using hash-based assignment.
"""

import hashlib
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


@dataclass
class WeightConfig:
    """
    Scoring weight configuration.
    
    All weights should sum to ~1.0 for interpretability, though this
    is not strictly enforced to allow flexibility.
    
    Attributes:
        w_rating: Weight for normalized rating score (0-1)
        w_diversity: Weight for diversity gain score (0-1)
        w_eta: Weight for travel time score (0-1, inverted so lower is better)
        w_open: Weight for open-now bonus (0-1)
        w_crowd: Weight for crowd penalty (0-1, higher crowd = lower score)
        variant_name: Name of the variant (for telemetry tracking)
    """
    w_rating: float = 0.30
    w_diversity: float = 0.25
    w_eta: float = 0.20
    w_open: float = 0.15
    w_crowd: float = 0.10
    variant_name: str = "default"
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for backward compatibility."""
        return {
            "w_rating": self.w_rating,
            "w_diversity": self.w_diversity,
            "w_eta": self.w_eta,
            "w_open": self.w_open,
            "w_crowd": self.w_crowd,
        }
    
    def dict(self) -> Dict[str, float]:
        """Alias for to_dict() for pydantic-like interface."""
        return self.to_dict()


# Default weight configurations
DEFAULT_WEIGHTS = WeightConfig(
    w_rating=0.30,
    w_diversity=0.25,
    w_eta=0.20,
    w_open=0.15,
    w_crowd=0.10,
    variant_name="default"
)

# A/B Test Variants
VARIANT_A = WeightConfig(
    w_rating=0.35,      # Emphasize quality
    w_diversity=0.20,
    w_eta=0.20,
    w_open=0.15,
    w_crowd=0.10,
    variant_name="variant-a"
)

VARIANT_B = WeightConfig(
    w_rating=0.25,
    w_diversity=0.30,   # Emphasize variety
    w_eta=0.20,
    w_open=0.15,
    w_crowd=0.10,
    variant_name="variant-b"
)

VARIANT_C = WeightConfig(
    w_rating=0.30,
    w_diversity=0.25,
    w_eta=0.25,         # Emphasize proximity
    w_open=0.15,
    w_crowd=0.05,
    variant_name="variant-c"
)

# Registry of all variants
WEIGHT_VARIANTS: Dict[str, WeightConfig] = {
    "default": DEFAULT_WEIGHTS,
    "variant-a": VARIANT_A,
    "variant-b": VARIANT_B,
    "variant-c": VARIANT_C,
}


def load_weights_from_yaml(yaml_path: Optional[str] = None) -> Dict[str, WeightConfig]:
    """
    Load weight configurations from YAML file.
    
    Args:
        yaml_path: Path to weights.yaml file. If None, looks in configs/weights.yaml
    
    Returns:
        Dictionary mapping variant names to WeightConfig objects
    
    YAML Format:
        ```yaml
        default:
          w_rating: 0.30
          w_diversity: 0.25
          w_eta: 0.20
          w_open: 0.15
          w_crowd: 0.10
        
        variant-a:
          w_rating: 0.35
          w_diversity: 0.20
          ...
        ```
    """
    if yaml_path is None:
        # Try configs/weights.yaml relative to project root
        yaml_path = Path(__file__).parent.parent.parent / "configs" / "weights.yaml"
    
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        # Return built-in defaults if file doesn't exist
        return WEIGHT_VARIANTS
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    variants = {}
    for variant_name, weights in data.items():
        variants[variant_name] = WeightConfig(
            w_rating=weights.get("w_rating", 0.30),
            w_diversity=weights.get("w_diversity", 0.25),
            w_eta=weights.get("w_eta", 0.20),
            w_open=weights.get("w_open", 0.15),
            w_crowd=weights.get("w_crowd", 0.10),
            variant_name=variant_name,
        )
    
    return variants


def select_ab_variant(
    user_id: Optional[str] = None,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
    variant_names: Optional[list] = None,
) -> WeightConfig:
    """
    Select A/B test variant using session-sticky hash-based assignment.
    
    Uses deterministic hashing to ensure the same user/device/session always
    gets the same variant, enabling consistent A/B testing without server-side
    state management.
    
    Args:
        user_id: User identifier (highest priority)
        device_id: Device identifier (medium priority)
        session_id: Session identifier (lowest priority)
        variant_names: List of variant names to choose from. If None, uses all variants
    
    Returns:
        WeightConfig for the selected variant
    
    Examples:
        >>> # User-based assignment (most stable)
        >>> weights = select_ab_variant(user_id="user_12345")
        
        >>> # Device-based assignment (persistent across sessions)
        >>> weights = select_ab_variant(device_id="device_abc")
        
        >>> # Session-based assignment (least stable, for anonymous users)
        >>> weights = select_ab_variant(session_id="session_xyz")
    """
    # Determine the identifier to use (priority: user > device > session)
    identifier = user_id or device_id or session_id
    
    if identifier is None:
        # No identifier provided, return default
        return DEFAULT_WEIGHTS
    
    # Get available variants
    if variant_names is None:
        variant_names = ["variant-a", "variant-b", "variant-c"]
    
    # Hash the identifier to get a deterministic variant assignment
    hash_value = int(hashlib.sha256(identifier.encode()).hexdigest(), 16)
    variant_index = hash_value % len(variant_names)
    selected_variant = variant_names[variant_index]
    
    return WEIGHT_VARIANTS.get(selected_variant, DEFAULT_WEIGHTS)


def get_variant_by_name(variant_name: str) -> WeightConfig:
    """
    Get weight configuration by variant name.
    
    Args:
        variant_name: Name of the variant ("default", "variant-a", etc.)
    
    Returns:
        WeightConfig for the specified variant, or default if not found
    """
    return WEIGHT_VARIANTS.get(variant_name, DEFAULT_WEIGHTS)


def save_weights_to_yaml(
    variants: Dict[str, WeightConfig],
    yaml_path: Optional[str] = None
) -> str:
    """
    Save weight configurations to YAML file.
    
    Args:
        variants: Dictionary mapping variant names to WeightConfig objects
        yaml_path: Path to save weights.yaml. If None, saves to configs/weights.yaml
    
    Returns:
        Path where the file was saved
    """
    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent.parent / "configs" / "weights.yaml"
    
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict format for YAML
    data = {}
    for variant_name, config in variants.items():
        data[variant_name] = {
            "w_rating": config.w_rating,
            "w_diversity": config.w_diversity,
            "w_eta": config.w_eta,
            "w_open": config.w_open,
            "w_crowd": config.w_crowd,
        }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    return str(yaml_path)
