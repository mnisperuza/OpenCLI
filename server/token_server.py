"""
Token server 

By Matias Nisperuza 2026
══════════════════════════════════════════════════════════════════════════════
"""

import os
import hmac
import hashlib
import secrets
import string
from datetime import datetime, date, timedelta
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

# SECURITY: Set these in environment variables!
TOKEN_HMAC_SECRET = os.environ.get('TOKEN_HMAC_SECRET', 'CHANGE_ME_IN_PRODUCTION')
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://localhost/bert_tokens')

WEEKLY_TOKEN_LIMIT = 20000
RATE_LIMIT_CLAIMS_PER_HOUR = 5

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

import psycopg2
from psycopg2.extras import RealDictCursor

def get_db():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    """Initialize database tables"""
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            token VARCHAR(50) UNIQUE NOT NULL,
            week_number INTEGER NOT NULL,
            year INTEGER NOT NULL,
            tokens_used INTEGER DEFAULT 0,
            tokens_remaining INTEGER DEFAULT 20000,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            UNIQUE(email, week_number, year)
        );
        
        CREATE INDEX IF NOT EXISTS idx_token ON tokens(token);
        CREATE INDEX IF NOT EXISTS idx_email_week ON tokens(email, week_number, year);
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id SERIAL PRIMARY KEY,
            token_id INTEGER REFERENCES tokens(id),
            tokens_used INTEGER NOT NULL,
            endpoint VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS claim_attempts (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            ip_address VARCHAR(50),
            success BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_claims_email ON claim_attempts(email, created_at);
    """)
    
    conn.commit()
    cur.close()
    conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_current_week():
    """Get current ISO week number and year"""
    today = date.today()
    week = today.isocalendar()[1]
    year = today.isocalendar()[0]
    return week, year

def generate_token_string():
    """Generate random token segment"""
    chars = string.ascii_uppercase.replace('O', '').replace('I', '').replace('L', '') + '23456789'
    return ''.join(secrets.choice(chars) for _ in range(4))

def sign_token(token_base, week, year):
    """Create HMAC signature for token"""
    data = f"{token_base}:{week}:{year}"
    sig = hmac.new(
        TOKEN_HMAC_SECRET.encode(),
        data.encode(),
        hashlib.sha256
    ).hexdigest()[:4].upper()
    return sig

def create_token():
    """Create a new signed token"""
    week, year = get_current_week()
    
    # Generate token parts
    p1 = generate_token_string()
    p2 = generate_token_string()
    p3 = generate_token_string()
    
    # Create base token
    token_base = f"BERT-{p1}-{p2}-{p3}"
    
    # Sign it
    sig = sign_token(token_base, week, year)
    
    # Format: BERT-XXXX-XXXX-XXXX-WWSS (Week + Signature)
    week_str = f"{week:02d}"
    full_token = f"{token_base}-{week_str}{sig}"
    
    return full_token, week, year

def verify_token_signature(token):
    """Verify token signature is valid"""
    try:
        parts = token.upper().split('-')
        if len(parts) != 5 or parts[0] != 'BERT':
            return False, "Invalid format"
        
        # Extract week and signature from last part
        last_part = parts[4]
        if len(last_part) != 6:
            return False, "Invalid token length"
        
        week_str = last_part[:2]
        sig = last_part[2:]
        
        try:
            week = int(week_str)
        except ValueError:
            return False, "Invalid week"
        
        # Reconstruct base token
        token_base = f"BERT-{parts[1]}-{parts[2]}-{parts[3]}"
        
        # Get current year
        _, year = get_current_week()
        
        # Verify signature
        expected_sig = sign_token(token_base, week, year)
        
        if not hmac.compare_digest(sig, expected_sig):
            # Try previous year (for week 52/1 transitions)
            expected_sig_prev = sign_token(token_base, week, year - 1)
            if not hmac.compare_digest(sig, expected_sig_prev):
                return False, "Invalid signature"
        
        return True, week
        
    except Exception as e:
        return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════

def check_rate_limit(email, ip):
    """Check if email/IP is rate limited"""
    conn = get_db()
    cur = conn.cursor()
    
    one_hour_ago = datetime.now() - timedelta(hours=1)
    
    cur.execute("""
        SELECT COUNT(*) as count FROM claim_attempts
        WHERE (email = %s OR ip_address = %s)
        AND created_at > %s
    """, (email, ip, one_hour_ago))
    
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    return result['count'] < RATE_LIMIT_CLAIMS_PER_HOUR

def log_claim_attempt(email, ip, success):
    """Log a claim attempt"""
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO claim_attempts (email, ip_address, success)
        VALUES (%s, %s, %s)
    """, (email, ip, success))
    
    conn.commit()
    cur.close()
    conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "ok", "service": "bert-tokens"})

@app.route('/api/claim', methods=['POST'])
def claim_token():
    """
    Claim a new token for the current week.
    Requires email. One token per email per week.
    """
    data = request.get_json() or {}
    email = data.get('email', '').lower().strip()
    
    if not email or '@' not in email:
        return jsonify({"error": "Valid email required"}), 400
    
    ip = request.remote_addr
    
    # Rate limit check
    if not check_rate_limit(email, ip):
        log_claim_attempt(email, ip, False)
        return jsonify({"error": "Too many attempts. Try again later."}), 429
    
    week, year = get_current_week()
    
    conn = get_db()
    cur = conn.cursor()
    
    # Check if email already has token this week
    cur.execute("""
        SELECT token, tokens_remaining FROM tokens
        WHERE email = %s AND week_number = %s AND year = %s AND is_active = TRUE
    """, (email, week, year))
    
    existing = cur.fetchone()
    
    if existing:
        log_claim_attempt(email, ip, True)
        cur.close()
        conn.close()
        return jsonify({
            "success": True,
            "token": existing['token'],
            "tokens_remaining": existing['tokens_remaining'],
            "message": "You already have a token for this week",
            "week": week
        })
    
    # Create new token
    token, _, _ = create_token()
    
    try:
        cur.execute("""
            INSERT INTO tokens (email, token, week_number, year, tokens_remaining)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, token, tokens_remaining
        """, (email, token, week, year, WEEKLY_TOKEN_LIMIT))
        
        result = cur.fetchone()
        conn.commit()
        
        log_claim_attempt(email, ip, True)
        
        cur.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "token": result['token'],
            "tokens_remaining": result['tokens_remaining'],
            "week": week,
            "message": f"Token created! {WEEKLY_TOKEN_LIMIT:,} tokens available this week."
        })
        
    except Exception as e:
        conn.rollback()
        cur.close()
        conn.close()
        log_claim_attempt(email, ip, False)
        return jsonify({"error": "Failed to create token"}), 500

@app.route('/api/validate', methods=['POST'])
def validate_token():
    """
    Validate a token and return remaining balance.
    Called by CLI on startup and periodically.
    """
    data = request.get_json() or {}
    token = data.get('token', '').upper().strip()
    
    if not token:
        return jsonify({"valid": False, "error": "Token required"}), 400
    
    # Verify signature first
    sig_valid, week_or_error = verify_token_signature(token)
    if not sig_valid:
        return jsonify({"valid": False, "error": week_or_error}), 400
    
    current_week, current_year = get_current_week()
    
    conn = get_db()
    cur = conn.cursor()
    
    # Look up token
    cur.execute("""
        SELECT id, email, tokens_used, tokens_remaining, week_number, year, is_active
        FROM tokens WHERE token = %s
    """, (token,))
    
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    if not result:
        return jsonify({"valid": False, "error": "Token not found"}), 404
    
    if not result['is_active']:
        return jsonify({"valid": False, "error": "Token deactivated"}), 403
    
    # Check if token is for current week
    if result['week_number'] != current_week or result['year'] != current_year:
        return jsonify({
            "valid": False,
            "error": "Token expired",
            "token_week": result['week_number'],
            "current_week": current_week
        }), 403
    
    return jsonify({
        "valid": True,
        "tokens_used": result['tokens_used'],
        "tokens_remaining": result['tokens_remaining'],
        "week": current_week
    })

@app.route('/api/use', methods=['POST'])
def use_tokens():
    """
    Record token usage from CLI.
    Called after each generation.
    """
    data = request.get_json() or {}
    token = data.get('token', '').upper().strip()
    count = data.get('count', 0)
    
    if not token:
        return jsonify({"error": "Token required"}), 400
    
    if not isinstance(count, int) or count < 0:
        return jsonify({"error": "Invalid count"}), 400
    
    current_week, current_year = get_current_week()
    
    conn = get_db()
    cur = conn.cursor()
    
    # Update token usage atomically
    cur.execute("""
        UPDATE tokens
        SET tokens_used = tokens_used + %s,
            tokens_remaining = tokens_remaining - %s,
            last_used_at = CURRENT_TIMESTAMP
        WHERE token = %s 
          AND week_number = %s 
          AND year = %s
          AND is_active = TRUE
          AND tokens_remaining >= %s
        RETURNING id, tokens_used, tokens_remaining
    """, (count, count, token, current_week, current_year, count))
    
    result = cur.fetchone()
    
    if not result:
        cur.close()
        conn.close()
        return jsonify({"error": "Token invalid or limit reached"}), 403
    
    # Log usage
    cur.execute("""
        INSERT INTO usage_log (token_id, tokens_used, endpoint)
        VALUES (%s, %s, %s)
    """, (result['id'], count, 'generate'))
    
    conn.commit()
    cur.close()
    conn.close()
    
    return jsonify({
        "success": True,
        "tokens_used": result['tokens_used'],
        "tokens_remaining": result['tokens_remaining']
    })

@app.route('/api/status', methods=['GET'])
def status():
    """Get server status and stats"""
    week, year = get_current_week()
    
    conn = get_db()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            COUNT(*) as total_tokens,
            COUNT(CASE WHEN week_number = %s AND year = %s THEN 1 END) as active_tokens,
            SUM(tokens_used) as total_usage
        FROM tokens
    """, (week, year))
    
    stats = cur.fetchone()
    cur.close()
    conn.close()
    
    return jsonify({
        "week": week,
        "year": year,
        "active_tokens": stats['active_tokens'] or 0,
        "total_tokens_issued": stats['total_tokens'] or 0,
        "total_usage": stats['total_usage'] or 0
    })

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("🔐 Bert Token Server")
    print(f"📅 Current week: {get_current_week()[0]}")
    
    # Initialize database
    try:
        init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"⚠️  Database init failed: {e}")
    
    # Run server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', False))
