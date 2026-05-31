"""
Simple JWT (JSON Web Token) helper for AI RecruitX.

Built using only Python standard library:
  - hmac       : creates the cryptographic signature
  - hashlib    : SHA-256 hashing algorithm
  - base64     : Base64Url encoding/decoding
  - json       : payload serialisation
  - time       : expiry timestamp

How a JWT works (explain in viva):
  A JWT has three parts separated by dots:
    HEADER.PAYLOAD.SIGNATURE

  - HEADER   : tells which algorithm is used (HS256)
  - PAYLOAD  : the actual data (username, expiry time)
  - SIGNATURE: HMAC-SHA256(HEADER + "." + PAYLOAD, secret_key)
               proves the token has not been tampered with
"""

import hmac
import hashlib
import base64
import json
import time


def _base64url_encode(data: bytes) -> str:
    """Encodes bytes to URL-safe Base64 string (no padding)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _base64url_decode(text: str) -> bytes:
    """Decodes URL-safe Base64 string back to bytes."""
    # Add back padding if missing
    padding = 4 - len(text) % 4
    if padding != 4:
        text += "=" * padding
    return base64.urlsafe_b64decode(text)


def generate_jwt(username: str, secret_key: str, expiry_hours: int = 2) -> str:
    """
    Creates a signed JWT token for the given username.

    Args:
        username   : The recruiter's username to embed in the token.
        secret_key : The app's secret key used to sign the token.
        expiry_hours: How many hours until the token expires (default 2).

    Returns:
        A JWT string in format: header.payload.signature
    """
    # Header: tells decoder which algorithm we used
    header = {"alg": "HS256", "typ": "JWT"}

    # Payload: the actual data stored inside the token
    payload = {
        "username": username,
        "iat": int(time.time()),                         # issued at
        "exp": int(time.time()) + (expiry_hours * 3600)  # expiry time
    }

    # Encode header and payload to Base64Url strings
    header_encoded  = _base64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_encoded = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode())

    # Create the signing message: header.payload
    signing_input = f"{header_encoded}.{payload_encoded}"

    # Sign with HMAC-SHA256
    signature = hmac.new(
        secret_key.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256
    ).digest()

    signature_encoded = _base64url_encode(signature)

    # Return complete JWT: header.payload.signature
    return f"{signing_input}.{signature_encoded}"


def verify_jwt(token: str, secret_key: str):
    """
    Verifies a JWT token and returns the payload if valid.

    Args:
        token      : The JWT string received from the browser cookie.
        secret_key : The same secret key used to sign the token.

    Returns:
        dict  : The decoded payload if the token is valid and not expired.
        None  : If the token is invalid, tampered, or expired.
    """
    try:
        # Split the token into its three parts
        parts = token.split(".")
        if len(parts) != 3:
            return None  # Not a valid JWT format

        header_encoded, payload_encoded, signature_encoded = parts

        # Re-create the signature from header + payload using the secret key
        signing_input = f"{header_encoded}.{payload_encoded}"
        expected_signature = hmac.new(
            secret_key.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256
        ).digest()

        expected_encoded = _base64url_encode(expected_signature)

        # Compare signatures safely (constant-time comparison prevents timing attacks)
        if not hmac.compare_digest(expected_encoded, signature_encoded):
            return None  # Token signature does not match — tampered!

        # Decode the payload
        payload = json.loads(_base64url_decode(payload_encoded).decode("utf-8"))

        # Check if the token has expired
        if payload.get("exp", 0) < int(time.time()):
            return None  # Token is expired

        return payload  # All checks passed — return the payload

    except Exception:
        return None  # Any decode/format error means invalid token
