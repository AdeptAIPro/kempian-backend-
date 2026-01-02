from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


cors = CORS()
# More lenient rate limiting - 1000 requests per minute for general use
# Health checks and essential endpoints will have higher limits
limiter = Limiter(key_func=get_remote_address, default_limits=["1000/minute"])


