#!/usr/bin/env python3
"""
Debug script to inspect JobVite API responses.
Run this to see exactly what the API returns.

Usage:
    python debug_jobvite_api.py
"""

import os
import sys
import time
import hmac
import base64
import requests
import json
from hashlib import sha256

# Get credentials from environment or set here
API_KEY = os.getenv("JOBVITE_API_KEY", "YOUR_API_KEY")
API_SECRET = os.getenv("JOBVITE_API_SECRET", "YOUR_API_SECRET")
COMPANY_ID = os.getenv("JOBVITE_COMPANY_ID", "YOUR_COMPANY_ID")
BASE_URL = os.getenv("JOBVITE_BASE_URL", "https://api.jvistg2.com/api/v2")

def try_legacy_auth(url):
    """Try legacy header authentication"""
    headers = {
        "x-jvi-api": API_KEY,
        "x-jvi-sc": API_SECRET
    }
    print(f"\n=== Trying Legacy Auth (x-jvi-api / x-jvi-sc) ===")
    print(f"URL: {url}")
    print(f"Headers: {list(headers.keys())}")
    
    try:
        r = requests.get(url, headers=headers, timeout=20)
        print(f"Status Code: {r.status_code}")
        print(f"Response Headers: {dict(r.headers)}")
        return r
    except Exception as e:
        print(f"Error: {e}")
        return None

def try_hmac_auth(url):
    """Try HMAC authentication"""
    epoch = str(int(time.time()))
    to_hash = f"{API_KEY}|{epoch}"
    sig = base64.b64encode(
        hmac.new(API_SECRET.encode(), to_hash.encode(), sha256).digest()
    ).decode()
    
    headers = {
        "X-JVI-API": API_KEY,
        "X-JVI-SIGN": sig,
        "X-JVI-EPOCH": epoch
    }
    print(f"\n=== Trying HMAC Auth (X-JVI-API / X-JVI-SIGN / X-JVI-EPOCH) ===")
    print(f"URL: {url}")
    print(f"Headers: {list(headers.keys())}")
    
    try:
        r = requests.get(url, headers=headers, timeout=20)
        print(f"Status Code: {r.status_code}")
        print(f"Response Headers: {dict(r.headers)}")
        return r
    except Exception as e:
        print(f"Error: {e}")
        return None

def inspect_response(r):
    """Inspect and print response details"""
    if not r:
        print("No response received")
        return
    
    print(f"\n=== Response Inspection ===")
    print(f"Status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('Content-Type', 'unknown')}")
    print(f"Content-Length: {len(r.text)} bytes")
    
    # Check if JSON
    try:
        data = r.json()
    except Exception as e:
        print(f"\n❌ Non-JSON Response (first 1000 chars):")
        print(r.text[:1000])
        return
    
    print(f"\n✅ Valid JSON Response")
    print(f"Top-level keys: {list(data.keys())}")
    
    # Check for list-like keys
    print(f"\n=== Checking for Job Lists ===")
    for k in ("jobs", "requisitions", "items", "data", "job"):
        if k in data:
            val = data[k]
            if isinstance(val, list):
                print(f"✓ Found key '{k}': list with {len(val)} items")
            elif isinstance(val, dict):
                print(f"✓ Found key '{k}': dict with keys {list(val.keys())}")
            else:
                print(f"✓ Found key '{k}': {type(val).__name__} = {val}")
    
    # Show total if exists
    if 'total' in data:
        print(f"\nTotal: {data['total']}")
    
    # Find the actual job list
    candidates = None
    list_key = None
    for k in ("jobs", "requisitions", "items", "data"):
        if k in data and isinstance(data[k], list):
            candidates = data[k]
            list_key = k
            break
    
    if candidates:
        print(f"\n=== First 5 Items from '{list_key}' ===")
        for i, item in enumerate(candidates[:5]):
            print(f"\nItem {i}:")
            print(f"  Keys: {list(item.keys())}")
            print(f"  ID: {item.get('id') or item.get('jobId') or item.get('requisitionId') or 'N/A'}")
            print(f"  Title: {item.get('title') or item.get('name') or 'N/A'}")
            print(f"  Status: {item.get('status') or 'N/A'}")
            print(f"  Type: {type(item).__name__}")
    else:
        print(f"\n❌ No list-like job payload found")
        print(f"\nFull payload preview (first 2000 chars):")
        print(json.dumps(data, indent=2)[:2000])
    
    # Show full response structure (truncated)
    print(f"\n=== Full Response Structure (first 3000 chars) ===")
    print(json.dumps(data, indent=2)[:3000])
    if len(json.dumps(data)) > 3000:
        print(f"\n... (truncated, total length: {len(json.dumps(data))} chars)")

def main():
    print("=" * 80)
    print("JobVite API Debug Script")
    print("=" * 80)
    
    # Check credentials
    if API_KEY == "YOUR_API_KEY" or API_SECRET == "YOUR_API_SECRET":
        print("\n⚠️  WARNING: Please set API credentials!")
        print("   Set environment variables:")
        print("   export JOBVITE_API_KEY='your_key'")
        print("   export JOBVITE_API_SECRET='your_secret'")
        print("   export JOBVITE_COMPANY_ID='your_company_id'")
        print("   export JOBVITE_BASE_URL='https://api.jvistg2.com/api/v2'")
        print("\n   Or edit this script and set values directly.")
        sys.exit(1)
    
    # Test endpoint
    url = f"{BASE_URL}/job?start=0&count=100"
    
    print(f"\nTesting endpoint: {url}")
    print(f"Company ID: {COMPANY_ID}")
    print(f"API Key (first 10 chars): {API_KEY[:10]}...")
    
    # Try legacy auth first
    r = try_legacy_auth(url)
    
    if not r or r.status_code != 200:
        print("\n⚠️  Legacy auth failed, trying HMAC...")
        r = try_hmac_auth(url)
    
    if not r:
        print("\n❌ Both authentication methods failed")
        sys.exit(1)
    
    if r.status_code != 200:
        print(f"\n❌ Request failed with status {r.status_code}")
        print(f"Response: {r.text[:1000]}")
        sys.exit(1)
    
    # Inspect response
    inspect_response(r)
    
    # Also try meta endpoint if available
    print(f"\n\n{'=' * 80}")
    print("Testing /meta endpoint (if available)")
    print("=" * 80)
    meta_url = f"{BASE_URL}/meta"
    r_meta = try_legacy_auth(meta_url)
    if r_meta and r_meta.status_code == 200:
        try:
            meta_data = r_meta.json()
            print(f"Meta response: {json.dumps(meta_data, indent=2)}")
        except:
            print(f"Meta response (text): {r_meta.text[:500]}")

if __name__ == "__main__":
    main()

