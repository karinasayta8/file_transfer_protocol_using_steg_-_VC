import cv2
import socket
import hashlib
import os
import sys
from datetime import datetime
from blockchain import Blockchain
from adaptive_lsb import AdaptiveLSB
from visual_crypto import VisualCryptography

class FileTransferClient:
    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.lsb = AdaptiveLSB()
        self.vc = VisualCryptography()
        self.blockchain = Blockchain()

    def send_file(self, image_path, file_path):
        # Verify files exist with clear error messages
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"🚨 Cover image not found: {image_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"🚨 Secret file not found: {file_path}")

        # Read files with validation
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"❌ Invalid image format: {image_path}")
            
        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Embed data and generate shares
        stego_image = self.lsb.embed(image, file_data)
        share1, share2 = self.vc.generate_shares(stego_image)

        # Network communication
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.server_host, self.server_port))
            
            # Send shares
            self.send_image(sock, share1)
            self.send_image(sock, share2)
            
            # Send data length
            sock.send(len(file_data).to_bytes(4, byteorder='big'))
            
            # Blockchain record
            file_hash = hashlib.sha256(file_data).hexdigest()
            print(f"\n📤 Client Hash: {file_hash}")
    
            self.blockchain.add_block({
                'type': 'transfer',
                'file_hash': file_hash,
                'timestamp': str(datetime.now()),
                'status': 'sent'
            })
    
            # Force immediate blockchain save
            self.blockchain.save_chain()
            
            # Get verification
            response = sock.recv(1024)
            return response.decode()

    def send_image(self, sock, image):
        _, img_encoded = cv2.imencode('.png', image)
        sock.send(len(img_encoded).to_bytes(4, byteorder='big'))
        sock.sendall(img_encoded.tobytes())

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("🔥 Usage: python protocol_client.py <server_ip> <port> <cover_image> <secret_file>")
        print("Example: python protocol_client.py localhost 5000 cover.png secret.txt")
        sys.exit(1)

    server_ip = sys.argv[1]
    port = int(sys.argv[2])
    cover_image = sys.argv[3]
    secret_file = sys.argv[4]

    client = FileTransferClient(server_ip, port)
    try:
        print(f"🚀 Starting transfer: {cover_image} + {secret_file}")
        result = client.send_file(cover_image, secret_file)
        print(f"✅ Transfer result: {result}")
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        sys.exit(1)