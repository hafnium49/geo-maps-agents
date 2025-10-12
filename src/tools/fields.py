"""
Centralized FieldMask constants for Google Maps Platform APIs.

This module ensures consistent field selection across all API calls,
minimizing costs by requesting only necessary data.

Reference:
- Places API (New): https://developers.google.com/maps/documentation/places/web-service/place-details
- Routes API: https://developers.google.com/maps/documentation/routes/compute_routes
"""

from typing import Dict, List

# -----------------------------
# Places API FieldMasks
# -----------------------------

# Basic place identification and location
PLACES_BASIC_FIELDS = [
    "places.id",
    "places.displayName",
    "places.location",
]

# Type classification
PLACES_TYPE_FIELDS = [
    "places.primaryType",
    "places.types",
]

# Rating and review data
PLACES_RATING_FIELDS = [
    "places.rating",
    "places.userRatingCount",
]

# Pricing information
PLACES_PRICE_FIELDS = [
    "places.priceLevel",
]

# Operating hours
PLACES_HOURS_FIELDS = [
    "places.currentOpeningHours.openNow",
    "places.currentOpeningHours.periods",
    "places.currentOpeningHours.weekdayDescriptions",
]

# Business status (operational vs permanently closed)
PLACES_STATUS_FIELDS = [
    "places.businessStatus",
]

# Navigation and attribution
PLACES_NAV_FIELDS = [
    "places.googleMapsUri",
]

# Formatted address
PLACES_ADDRESS_FIELDS = [
    "places.formattedAddress",
    "places.addressComponents",
]

# Photos (use sparingly - each photo fetch incurs additional cost)
PLACES_PHOTO_FIELDS = [
    "places.photos",
]

# Reviews (ToS: display max 5, with attribution, no caching)
PLACES_REVIEW_FIELDS = [
    "places.reviews",
]

# -----------------------------
# Composite FieldMasks
# -----------------------------

# Text Search: Cost-effective discovery
PLACES_TEXT_SEARCH_FIELDS = (
    PLACES_BASIC_FIELDS
    + PLACES_TYPE_FIELDS
    + PLACES_RATING_FIELDS
    + PLACES_PRICE_FIELDS
    + ["places.currentOpeningHours.openNow"]  # Only openNow flag, not full hours
    + PLACES_NAV_FIELDS
)

# Place Details: Enrichment with full opening hours
PLACES_DETAILS_FIELDS = (
    PLACES_BASIC_FIELDS
    + PLACES_TYPE_FIELDS
    + PLACES_RATING_FIELDS
    + PLACES_PRICE_FIELDS
    + PLACES_HOURS_FIELDS
    + PLACES_STATUS_FIELDS
    + PLACES_NAV_FIELDS
)

# Nearby Search: Similar to text search
PLACES_NEARBY_SEARCH_FIELDS = PLACES_TEXT_SEARCH_FIELDS

# Full enrichment (use only when necessary - higher cost)
PLACES_FULL_FIELDS = (
    PLACES_BASIC_FIELDS
    + PLACES_TYPE_FIELDS
    + PLACES_RATING_FIELDS
    + PLACES_PRICE_FIELDS
    + PLACES_HOURS_FIELDS
    + PLACES_STATUS_FIELDS
    + PLACES_NAV_FIELDS
    + PLACES_ADDRESS_FIELDS
)

# -----------------------------
# Routes API FieldMasks
# -----------------------------

# computeRoutes: Basic routing response
ROUTES_BASIC_FIELDS = [
    "routes.duration",
    "routes.distanceMeters",
    "routes.polyline.encodedPolyline",
]

# computeRoutes: With traffic and conditions
ROUTES_TRAFFIC_FIELDS = ROUTES_BASIC_FIELDS + [
    "routes.routeLabels",
    "routes.warnings",
    "routes.travelAdvisory",
]

# computeRouteMatrix: Element-level response
ROUTES_MATRIX_FIELDS = [
    "originIndex",
    "destinationIndex",
    "duration",
    "distanceMeters",
    "status",
    "condition",
]

# -----------------------------
# Helper Functions
# -----------------------------

def get_fieldmask_header(fields: List[str]) -> Dict[str, str]:
    """
    Generate X-Goog-FieldMask header from field list.
    
    Args:
        fields: List of field paths (e.g., ["places.id", "places.displayName"])
        
    Returns:
        Dictionary with FieldMask header
        
    Example:
        >>> get_fieldmask_header(PLACES_TEXT_SEARCH_FIELDS)
        {'X-Goog-FieldMask': 'places.id,places.displayName,...'}
    """
    return {"X-Goog-FieldMask": ",".join(fields)}


def get_places_search_mask() -> Dict[str, str]:
    """Get FieldMask header for Places Text/Nearby Search."""
    return get_fieldmask_header(PLACES_TEXT_SEARCH_FIELDS)


def get_places_details_mask() -> Dict[str, str]:
    """Get FieldMask header for Places Details."""
    return get_fieldmask_header(PLACES_DETAILS_FIELDS)


def get_routes_matrix_mask() -> Dict[str, str]:
    """Get FieldMask header for Routes computeRouteMatrix."""
    return get_fieldmask_header(ROUTES_MATRIX_FIELDS)
