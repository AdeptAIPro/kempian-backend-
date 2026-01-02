from typing import Optional
import os
from jose import jwt, JWTError


def verify_token(token: str) -> Optional[dict]:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        return None
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError:
        return None


