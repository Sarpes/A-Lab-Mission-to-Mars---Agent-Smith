#####################################################################
# ws_check.py - WebSocket el sikismasi (handshake) kontrolu
#
# NE: Verilen ws:// veya wss:// adresine HTTP Upgrade istegi yollar,
#     durum satirini yazdirir. 101 = WebSocket kabul edildi.
# NEDEN: Funnel arkasinda Streamlit WebSocket'i (/_stcore/stream) gecmeli;
#        tarayici olmadan, yalnizca standart kutuphane ile kanitlamak icin.
# Kullanim: python tools/ws_check.py wss://host/_stcore/stream
#####################################################################
import base64
import os
import socket
import ssl
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
guvenli = url.scheme == "wss"
port = url.port or (443 if guvenli else 80)

# Rastgele anahtar: sunucu bunu dogrulayip 101 ile yanit verir
anahtar = base64.b64encode(os.urandom(16)).decode()
istek = (
    f"GET {url.path or '/'} HTTP/1.1\r\n"
    f"Host: {url.hostname}\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    f"Sec-WebSocket-Key: {anahtar}\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    f"Origin: {'https' if guvenli else 'http'}://{url.hostname}\r\n\r\n"
)

ham = socket.create_connection((url.hostname, port), timeout=15)
soket = ssl.create_default_context().wrap_socket(ham, server_hostname=url.hostname) if guvenli else ham
soket.sendall(istek.encode())
yanit = soket.recv(4096).decode(errors="replace")
print(yanit.split("\r\n")[0])
sys.exit(0 if " 101 " in yanit.split("\r\n")[0] else 1)
