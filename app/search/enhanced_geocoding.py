"""
Enhanced Geocoding Service with Timezone Support
Production-grade location matching with geohash, distance scoring, and timezone compatibility.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import math

try:
    import geopy
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

try:
    import geohash2
    GEOHASH_AVAILABLE = True
except ImportError:
    try:
        import pygeohash as geohash2
        GEOHASH_AVAILABLE = True
    except ImportError:
        GEOHASH_AVAILABLE = False

try:
    import pytz
    from timezonefinder import TimezoneFinder
    TIMEZONE_AVAILABLE = True
except ImportError:
    TIMEZONE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EnhancedLocationData:
    """Enhanced location data with timezone and geohash"""
    location_id: str
    standardized_name: str
    country: str
    state: Optional[str]
    city: Optional[str]
    latitude: float
    longitude: float
    geohash: str
    timezone: Optional[str] = None
    timezone_offset: Optional[int] = None  # UTC offset in hours
    is_remote: bool = False
    willing_to_relocate: bool = False
    relocation_radius_km: Optional[float] = None


class EnhancedGeocodingService:
    """Enhanced geocoding with timezone and advanced scoring"""
    
    def __init__(self):
        self.geocoder = None
        self.timezone_finder = None
        self.location_cache: Dict[str, EnhancedLocationData] = {}
        self.geohash_precision = 5  # ~4.9km radius
        
        if GEOPY_AVAILABLE:
            try:
                self.geocoder = Nominatim(user_agent="kempian_search_system", timeout=10)
            except Exception as e:
                logger.warning(f"Could not initialize geocoder: {e}")
        
        if TIMEZONE_AVAILABLE:
            try:
                self.timezone_finder = TimezoneFinder()
            except Exception as e:
                logger.warning(f"Could not initialize timezone finder: {e}")
    
    def geocode_location(
        self,
        location_str: str,
        is_remote: bool = False,
        willing_to_relocate: bool = False,
        relocation_radius_km: Optional[float] = None
    ) -> Optional[EnhancedLocationData]:
        """Geocode location with full metadata"""
        if not location_str or location_str.strip().lower() in ['remote', 'anywhere', 'work from home']:
            return self._create_remote_location(willing_to_relocate, relocation_radius_km)
        
        if is_remote:
            return self._create_remote_location(willing_to_relocate, relocation_radius_km)
        
        # Check cache
        cache_key = f"{location_str.lower().strip()}_{is_remote}_{willing_to_relocate}"
        if cache_key in self.location_cache:
            return self.location_cache[cache_key]
        
        try:
            normalized = self._normalize_location_string(location_str)
            
            if not self.geocoder:
                return self._create_fallback_location(normalized, willing_to_relocate, relocation_radius_km)
            
            # Geocode
            location = self.geocoder.geocode(normalized, exactly_one=True, timeout=10)
            
            if not location:
                return self._create_fallback_location(normalized, willing_to_relocate, relocation_radius_km)
            
            # Extract components
            address = location.raw.get('address', {})
            country = address.get('country', '')
            state = address.get('state') or address.get('region') or address.get('county')
            city = address.get('city') or address.get('town') or address.get('village')
            
            # Get timezone
            timezone, timezone_offset = self._get_timezone(location.latitude, location.longitude)
            
            # Create location ID
            location_id = self._create_location_id(country, state, city)
            
            # Generate geohash
            geohash = self._generate_geohash(location.latitude, location.longitude)
            
            location_data = EnhancedLocationData(
                location_id=location_id,
                standardized_name=normalized,
                country=country,
                state=state,
                city=city,
                latitude=location.latitude,
                longitude=location.longitude,
                geohash=geohash,
                timezone=timezone,
                timezone_offset=timezone_offset,
                is_remote=False,
                willing_to_relocate=willing_to_relocate,
                relocation_radius_km=relocation_radius_km
            )
            
            # Cache
            self.location_cache[cache_key] = location_data
            
            return location_data
            
        except Exception as e:
            logger.error(f"Error geocoding location '{location_str}': {e}")
            return self._create_fallback_location(location_str, willing_to_relocate, relocation_radius_km)
    
    def _get_timezone(self, latitude: float, longitude: float) -> Tuple[Optional[str], Optional[int]]:
        """Get timezone for coordinates"""
        if not self.timezone_finder:
            return None, None
        
        try:
            timezone_str = self.timezone_finder.timezone_at(lat=latitude, lng=longitude)
            
            if timezone_str and pytz:
                tz = pytz.timezone(timezone_str)
                utc_offset = tz.utcoffset(datetime.now()).total_seconds() / 3600
                return timezone_str, int(utc_offset)
            
            return timezone_str, None
            
        except Exception as e:
            logger.error(f"Error getting timezone: {e}")
            return None, None
    
    def calculate_distance_score(
        self,
        distance_km: float,
        sigma: float = 50.0,
        is_remote: bool = False,
        willing_to_relocate: bool = False,
        relocation_radius: Optional[float] = None
    ) -> float:
        """
        Calculate distance score with exponential decay
        
        Formula: distance_score = exp(-distance_km / sigma)
        
        Args:
            distance_km: Distance in kilometers
            sigma: Decay parameter (larger = slower decay)
            is_remote: Whether candidate is remote
            willing_to_relocate: Whether candidate is willing to relocate
            relocation_radius: Maximum relocation radius in km
        """
        if is_remote:
            return 1.0  # Perfect score for remote
        
        if distance_km == float('inf') or distance_km < 0:
            return 0.0
        
        # If willing to relocate, check if within radius
        if willing_to_relocate and relocation_radius:
            if distance_km <= relocation_radius:
                return 1.0  # Within relocation radius
            else:
                # Penalty for outside radius
                penalty = (distance_km - relocation_radius) / sigma
                return max(0.0, math.exp(-penalty))
        
        # Standard exponential decay
        score = math.exp(-distance_km / sigma)
        return max(0.0, min(1.0, score))
    
    def check_timezone_compatibility(
        self,
        loc1: EnhancedLocationData,
        loc2: EnhancedLocationData,
        max_offset_hours: int = 3
    ) -> Tuple[bool, float]:
        """
        Check timezone compatibility
        
        Returns: (is_compatible, compatibility_score)
        """
        if loc1.is_remote or loc2.is_remote:
            return (True, 1.0)
        
        if not loc1.timezone_offset or not loc2.timezone_offset:
            return (True, 0.8)  # Unknown timezone, assume compatible
        
        offset_diff = abs(loc1.timezone_offset - loc2.timezone_offset)
        
        is_compatible = offset_diff <= max_offset_hours
        compatibility_score = max(0.0, 1.0 - (offset_diff / 12.0))  # Normalize to 0-1
        
        return (is_compatible, compatibility_score)
    
    def get_geohash_neighbors(self, geohash: str, radius_km: float = 50.0) -> List[str]:
        """Get geohash neighbors within radius"""
        if not GEOHASH_AVAILABLE or geohash in ['remote', 'unknown']:
            return [geohash]
        
        try:
            # Calculate precision needed
            if radius_km <= 5:
                precision = 6  # ~1.2km
            elif radius_km <= 20:
                precision = 5  # ~4.9km
            elif radius_km <= 100:
                precision = 4  # ~39km
            else:
                precision = 3  # ~156km
            
            # Get neighbors
            neighbors = geohash2.get_neighbors(geohash[:precision])
            
            return [geohash[:precision]] + list(neighbors)
            
        except Exception as e:
            logger.error(f"Error getting geohash neighbors: {e}")
            return [geohash]
    
    def calculate_distance_km(
        self,
        loc1: EnhancedLocationData,
        loc2: EnhancedLocationData
    ) -> float:
        """Calculate distance in kilometers"""
        if loc1.is_remote or loc2.is_remote:
            return 0.0
        
        if loc1.latitude == 0.0 and loc1.longitude == 0.0:
            return float('inf')
        
        if loc2.latitude == 0.0 and loc2.longitude == 0.0:
            return float('inf')
        
        try:
            if GEOPY_AVAILABLE:
                return geodesic(
                    (loc1.latitude, loc1.longitude),
                    (loc2.latitude, loc2.longitude)
                ).kilometers
            else:
                return self._haversine_distance(
                    loc1.latitude, loc1.longitude,
                    loc2.latitude, loc2.longitude
                )
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance calculation"""
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _normalize_location_string(self, location_str: str) -> str:
        """Normalize location string"""
        # Same as before but enhanced
        normalized = ' '.join(location_str.split())
        
        # State abbreviations (same as before)
        abbreviations = {
            'usa': 'United States', 'us': 'United States',
            'uk': 'United Kingdom', 'uae': 'United Arab Emirates',
            # ... (same abbreviations as before)
        }
        
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
        return hashlib.md5(location_id.encode()).hexdigest()[:16]
    
    def _generate_geohash(self, latitude: float, longitude: float) -> str:
        """Generate geohash"""
        if not GEOHASH_AVAILABLE:
            return f"{latitude:.4f},{longitude:.4f}"
        
        try:
            return geohash2.encode(latitude, longitude, precision=self.geohash_precision)
        except Exception as e:
            logger.error(f"Error generating geohash: {e}")
            return f"{latitude:.4f},{longitude:.4f}"
    
    def _create_remote_location(
        self,
        willing_to_relocate: bool,
        relocation_radius: Optional[float]
    ) -> EnhancedLocationData:
        """Create remote location data"""
        return EnhancedLocationData(
            location_id='remote',
            standardized_name='Remote',
            country='Remote',
            state=None,
            city=None,
            latitude=0.0,
            longitude=0.0,
            geohash='remote',
            timezone=None,
            timezone_offset=None,
            is_remote=True,
            willing_to_relocate=willing_to_relocate,
            relocation_radius_km=relocation_radius
        )
    
    def _create_fallback_location(
        self,
        location_str: str,
        willing_to_relocate: bool,
        relocation_radius: Optional[float]
    ) -> EnhancedLocationData:
        """Create fallback location"""
        location_id = hashlib.md5(location_str.lower().encode()).hexdigest()[:16]
        
        return EnhancedLocationData(
            location_id=location_id,
            standardized_name=location_str,
            country='Unknown',
            state=None,
            city=None,
            latitude=0.0,
            longitude=0.0,
            geohash='unknown',
            timezone=None,
            timezone_offset=None,
            is_remote=False,
            willing_to_relocate=willing_to_relocate,
            relocation_radius_km=relocation_radius
        )


# Global instance
_enhanced_geocoding = None

def get_enhanced_geocoding() -> EnhancedGeocodingService:
    """Get or create global enhanced geocoding instance"""
    global _enhanced_geocoding
    if _enhanced_geocoding is None:
        _enhanced_geocoding = EnhancedGeocodingService()
    return _enhanced_geocoding

