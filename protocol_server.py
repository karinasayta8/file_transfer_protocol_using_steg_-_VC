# protocol_server.py - Run this on receiving server
import cv2
import numpy as np
import socket
import hashlib
import threading
from datetime import datetime
from adaptive_lsb import AdaptiveLSB
from visual_crypto import VisualCryptography
from blockchain import Blockchain, Block

class FileTransferServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.lsb = AdaptiveLSB()
        self.vc = VisualCryptography()
        self.blockchain = Blockchain()
        self.setup_server()

    def setup_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Server listening on {self.host}:{self.port}")

    def handle_client(self, conn):
        try:
            # Receive shares
            share1 = self.receive_image(conn)
            share2 = self.receive_image(conn)
        
        # Reconstruct and extract
            reconstructed = self.vc.combine_shares(share1, share2)
            cv2.imwrite('reconstructed.png', reconstructed)
        
            data_length = int.from_bytes(conn.recv(4), byteorder='big')
            extracted_data = self.lsb.extract(reconstructed, data_length)
        
        # Force blockchain reload from disk
            self.blockchain.load_chain()
        
        # Debug: Save extracted data
            with open('extracted.bin', 'wb') as f:
                f.write(extracted_data)
            
            file_hash = hashlib.sha256(extracted_data).hexdigest()
            print(f"\n🔍 Verification Hash: {file_hash}")
            print("📦 Blockchain Contents:")
            for idx, block in enumerate(self.blockchain.chain):
                if isinstance(block.data, dict) and block.data.get('type') == 'transfer':
                    print(f" Block {idx}: {block.data.get('file_hash')}")
        
            verified = self.blockchain.verify_file(file_hash)
        
            if verified:
                conn.send(b"SUCCESS")
                print("✅ Verification Successful")
            else:
                conn.send(b"FAIL")
                print("❌ Verification Failed - Hash not found in blockchain")

        except Exception as e:
            print(f"⚠️ Error: {e}")
        finally:
            conn.close()

    def receive_image(self, conn):
        size_bytes = conn.recv(4)
        if not size_bytes:
            return None
        size = int.from_bytes(size_bytes, byteorder='big')
        img_data = b''
        while len(img_data) < size:
            packet = conn.recv(size - len(img_data))
            if not packet:
                return None
            img_data += packet
        return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    def start(self):
        while True:
            conn, addr = self.sock.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=self.handle_client, args=(conn,)).start()

if __name__ == "__main__":
    server = FileTransferServer()
    server.start()