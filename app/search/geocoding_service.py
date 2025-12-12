"""
Geocoding Service for Location Standardization
Geocodes locations, stores lat/lon, standardized location_id, and geohash for fast filtering.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib

try:
    import geopy
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    logger.warning("geopy not available, using fallback geocoding")

try:
    import geohash2
    GEOHASH_AVAILABLE = True
except ImportError:
    try:
        import pygeohash as geohash2
        GEOHASH_AVAILABLE = True
    except ImportError:
        GEOHASH_AVAILABLE = False
        logger.warning("geohash library not available")

logger = logging.getLogger(__name__)

@dataclass
class LocationData:
    """Standardized location data"""
    location_id: str
    standardized_name: str
    country: str
    state: Optional[str]
    city: Optional[str]
    latitude: float
    longitude: float
    geohash: str
    timezone: Optional[str] = None
    is_remote: bool = False


class GeocodingService:
    """Service for geocoding and location matching"""
    
    def __init__(self):
        self.geocoder = None
        self.location_cache: Dict[str, LocationData] = {}
        self.geohash_precision = 5  # ~4.9km radius
        
        if GEOPY_AVAILABLE:
            try:
                self.geocoder = Nominatim(user_agent="kempian_search_system", timeout=10)
                logger.info("Geocoding service initialized")
            except Exception as e:
                logger.warning(f"Could not initialize geocoder: {e}")
    
    def geocode_location(self, location_str: str, is_remote: bool = False) -> Optional[LocationData]:
        """
        Geocode a location string to LocationData
        
        Args:
            location_str: Raw location string (e.g., "San Francisco, CA, USA")
            is_remote: Whether this is a remote location
        
        Returns:
            LocationData or None if geocoding fails
        """
        if not location_str or location_str.strip().lower() in ['remote', 'anywhere', 'work from home']:
            return self._create_remote_location()
        
        if is_remote:
            return self._create_remote_location()
        
        # Check cache first
        cache_key = location_str.lower().strip()
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]
        
        # Try to parse and geocode
        try:
            # Normalize location string
            normalized = self._normalize_location_string(location_str)
            
            if not self.geocoder:
                # Fallback: create basic location data without geocoding
                return self._create_fallback_location(normalized)
            
            # Geocode using Nominatim
            location = self.geocoder.geocode(normalized, exactly_one=True, timeout=10)
            
            if not location:
                logger.warning(f"Could not geocode: {location_str}")
                return self._create_fallback_location(normalized)
            
            # Extract location components
            address = location.raw.get('address', {})
            country = address.get('country', '')
            state = address.get('state') or address.get('region') or address.get('county')
            city = address.get('city') or address.get('town') or address.get('village')
            
            # Create location ID
            location_id = self._create_location_id(country, state, city)
            
            # Generate geohash
            geohash = self._generate_geohash(location.latitude, location.longitude)
            
            location_data = LocationData(
                location_id=location_id,
                standardized_name=normalized,
                country=country,
                state=state,
                city=city,
                latitude=location.latitude,
                longitude=location.longitude,
                geohash=geohash,
                is_remote=False
            )
            
            # Cache it
            self.location_cache[cache_key] = location_data
            
            return location_data
            
        except Exception as e:
            logger.error(f"Error geocoding location '{location_str}': {e}")
            return self._create_fallback_location(location_str)
    
    def _normalize_location_string(self, location_str: str) -> str:
        """Normalize location string for geocoding"""
        if not location_str:
            return ''
        
        # Remove extra whitespace
        normalized = ' '.join(location_str.split())
        
        # Common abbreviations
        abbreviations = {
            'usa': 'United States',
            'us': 'United States',
            'uk': 'United Kingdom',
            'uae': 'United Arab Emirates',
            'ca': 'California',
            'ny': 'New York',
            'tx': 'Texas',
            'fl': 'Florida',
            'il': 'Illinois',
            'pa': 'Pennsylvania',
            'az': 'Arizona',
            'ma': 'Massachusetts',
            'wa': 'Washington',
            'co': 'Colorado',
            'nc': 'North Carolina',
            'ga': 'Georgia',
            'mi': 'Michigan',
            'nj': 'New Jersey',
            'va': 'Virginia',
            'tn': 'Tennessee',
            'in': 'Indiana',
            'mo': 'Missouri',
            'md': 'Maryland',
            'wi': 'Wisconsin',
            'or': 'Oregon',
            'sc': 'South Carolina',
            'al': 'Alabama',
            'la': 'Louisiana',
            'ky': 'Kentucky',
            'ut': 'Utah',
            'ia': 'Iowa',
            'ar': 'Arkansas',
            'nv': 'Nevada',
            'ms': 'Mississippi',
            'ks': 'Kansas',
            'nm': 'New Mexico',
            'ne': 'Nebraska',
            'id': 'Idaho',
            'hi': 'Hawaii',
            'nh': 'New Hampshire',
            'me': 'Maine',
            'mt': 'Montana',
            'ri': 'Rhode Island',
            'de': 'Delaware',
            'sd': 'South Dakota',
            'nd': 'North Dakota',
            'ak': 'Alaska',
            'dc': 'District of Columbia',
            'vt': 'Vermont',
            'wy': 'Wyoming',
            'wv': 'West Virginia'
        }
        
        # Replace abbreviations
        for abbr, full in abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            normalized = re.sub(pattern, full, normalized, flags=re.IGNORECASE)
        
        return normalized.strip()
    
    def _create_location_id(self, country: str, state: Optional[str], city: Optional[str]) -> str:
        """Create standardized location ID"""
        parts = []
        if country:
            parts.append(country.lower().replace(' ', '_'))
        if state:
            parts.append(state.lower().replace(' ', '_'))
        if city:
            parts.append(city.lower().replace(' ', '_'))
        
        if not parts:
            return 'unknown'
        
        location_id = '_'.join(parts)
        return hashlib.md5(location_id.encode()).hexdigest()[:16]  # Short hash
    
    def _generate_geohash(self, latitude: float, longitude: float) -> str:
        """Generate geohash for location"""
        if not GEOHASH_AVAILABLE:
            # Fallback: simple encoding
            return f"{latitude:.4f},{longitude:.4f}"
        
        try:
            return geohash2.encode(latitude, longitude, precision=self.geohash_precision)
        except Exception as e:
            logger.error(f"Error generating geohash: {e}")
            return f"{latitude:.4f},{longitude:.4f}"
    
    def _create_remote_location(self) -> LocationData:
        """Create remote location data"""
        return LocationData(
            location_id='remote',
            standardized_name='Remote',
            country='Remote',
            state=None,
            city=None,
            latitude=0.0,
            longitude=0.0,
            geohash='remote',
            is_remote=True
        )
    
    def _create_fallback_location(self, location_str: str) -> LocationData:
        """Create fallback location data when geocoding fails"""
        location_id = hashlib.md5(location_str.lower().encode()).hexdigest()[:16]
        
        return LocationData(
            location_id=location_id,
            standardized_name=location_str,
            country='Unknown',
            state=None,
            city=None,
            latitude=0.0,
            longitude=0.0,
            geohash='unknown',
            is_remote=False
        )
    
    def calculate_distance_km(self, loc1: LocationData, loc2: LocationData) -> float:
        """Calculate distance in kilometers between two locations"""
        if loc1.is_remote or loc2.is_remote:
            return 0.0  # Remote matches any location
        
        if loc1.latitude == 0.0 and loc1.longitude == 0.0:
            return float('inf')  # Unknown location
        
        if loc2.latitude == 0.0 and loc2.longitude == 0.0:
            return float('inf')  # Unknown location
        
        try:
            if GEOPY_AVAILABLE:
                return geodesic(
                    (loc1.latitude, loc1.longitude),
                    (loc2.latitude, loc2.longitude)
                ).kilometers
            else:
                # Haversine formula fallback
                return self._haversine_distance(
                    loc1.latitude, loc1.longitude,
                    loc2.latitude, loc2.longitude
                )
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula"""
        import math
        
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def get_geohash_neighbors(self, geohash: str, radius_km: float = 50.0) -> List[str]:
        """
        Get geohash neighbors within radius
        
        Args:
            geohash: Base geohash
            radius_km: Radius in kilometers
        
        Returns:
            List of geohash prefixes to search
        """
        if not GEOHASH_AVAILABLE or geohash == 'remote' or geohash == 'unknown':
            return [geohash]
        
        try:
            # Decode geohash to get center
            lat, lon = geohash2.decode(geohash)
            
            # Calculate precision needed for radius
            # Rough approximation: precision 5 = ~4.9km, precision 4 = ~39km
            if radius_km <= 5:
                precision = 5
            elif radius_km <= 20:
                precision = 4
            elif radius_km <= 100:
                precision = 3
            else:
                precision = 2
            
            # Get neighbors at appropriate precision
            neighbors = geohash2.get_neighbors(geohash[:precision])
            
            # Include base geohash
            result = [geohash[:precision]] + list(neighbors)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting geohash neighbors: {e}")
            return [geohash]
    
    def calculate_distance_score(self, distance_km: float, sigma: float = 50.0) -> float:
        """
        Calculate distance score using exponential decay
        
        Args:
            distance_km: Distance in kilometers
            sigma: Decay parameter (larger = slower decay)
        
        Returns:
            Score between 0 and 1
        """
        if distance_km == float('inf') or distance_km < 0:
            return 0.0
        
        import math
        score = math.exp(-distance_km / sigma)
        return max(0.0, min(1.0, score))


# Global instance
_geocoding_service = None

def get_geocoding_service() -> GeocodingService:
    """Get or create global geocoding service instance"""
    global _geocoding_service
    if _geocoding_service is None:
        _geocoding_service = GeocodingService()
    return _geocoding_service

def geocode_location(location_str: str, is_remote: bool = False) -> Optional[LocationData]:
    """Convenience function to geocode a location"""
    service = get_geocoding_service()
    return service.geocode_location(location_str, is_remote)

