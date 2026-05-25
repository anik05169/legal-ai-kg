import socket
import ssl
import urllib.request
import os
from dotenv import load_dotenv
load_dotenv()
def run_diagnostics():
    print("==================================================")
    print(" 🔍 GROQ API CONNECTION DIAGNOSTIC TOOL  🔍")
    print("==================================================")
    
    host = "api.groq.com"
    port = 443
    
    # --- 1. DNS Resolution Check ---
    print("\n1. [DNS] Resolving Hostname 'api.groq.com'...")
    try:
        ip_addresses = socket.getaddrinfo(host, port)
        resolved_ips = list(set([ip[4][0] for ip in ip_addresses]))
        print(f"   ✅ SUCCESS: Resolved to IPs: {resolved_ips}")
    except Exception as e:
        print(f"   ❌ FAILED: DNS Resolution Error: {e}")
        print("   💡 Suggestion: Your DNS server might be blocking or unable to resolve 'api.groq.com'. Try changing your DNS settings to Google DNS (8.8.8.8) or Cloudflare (1.1.1.1).")
        return
    # --- 2. TCP Socket Connection Check ---
    print(f"\n2. [TCP] Opening socket connection to {host}:{port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((resolved_ips[0], port))
        sock.close()
        print("   ✅ SUCCESS: Connected to port 443.")
    except Exception as e:
        print(f"   ❌ FAILED: Connection Error: {e}")
        print("   💡 Suggestion: A firewall or network proxy is blocking outgoing TCP traffic on port 443 to 'api.groq.com'.")
        return
    # --- 3. SSL Handshake Check ---
    print("\n3. [SSL/TLS] Checking SSL Handshake and Certificate...")
    try:
        context = ssl.create_default_context()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        ssl_sock = context.wrap_socket(sock, server_hostname=host)
        ssl_sock.connect((resolved_ips[0], port))
        cert = ssl_sock.getpeercert()
        ssl_sock.close()
        print("   ✅ SUCCESS: SSL/TLS handshake complete. Certificate verified.")
    except ssl.SSLCertVerificationError as e:
        print(f"   ❌ FAILED: SSL Verification Error: {e}")
        print("   💡 Suggestion: Your SSL certificate validation failed. This usually happens if you are behind a corporate proxy/VPN (like Zscaler) that intercepts SSL traffic. You may need to install the proxy's root CA certificate or bypass the proxy.")
        return
    except Exception as e:
        print(f"   ❌ FAILED: SSL Handshake Error: {e}")
        return
    # --- 4. Simple HTTP request check ---
    print("\n4. [HTTP] Testing API Endpoint reachability...")
    try:
        url = f"https://{host}/openai/v1/models"
        req = urllib.request.Request(url)
        # Check if key is available
        key = os.getenv("GROQ_API_KEY")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
            
        with urllib.request.urlopen(req, timeout=5.0) as response:
            status = response.getcode()
            print(f"   ✅ SUCCESS: Received HTTP {status}")
    except urllib.error.HTTPError as e:
        # 401/403 means auth failure but the connection itself was successful!
        if e.code in [401, 403]:
            print(f"   ✅ SUCCESS (Network-wise): Connection established, but received HTTP {e.code} (Auth/API Key issue).")
            print("   💡 Suggestion: Verify that your GROQ_API_KEY in '.env' is valid.")
        else:
            print(f"   ❌ FAILED: HTTP Error {e.code}: {e.reason}")
    except Exception as e:
        print(f"   ❌ FAILED: HTTP Request Error: {e}")
        
    print("\n==================================================")
if __name__ == "__main__":
    run_diagnostics()