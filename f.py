import cv2
import numpy as np
import hashlib
import os
import pickle
from datetime import datetime

# ==================== Adaptive LSB Encoding ====================
class AdaptiveLSB:
    def __init__(self, variance_threshold=50):
        self.variance_threshold = variance_threshold

    def embed(self, image, data, block_size=8):
        data_bin = ''.join(format(byte, '08b') for byte in data)
        h, w, c = image.shape
        data_ptr = 0
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = image[y:y+block_size, x:x+block_size]
                variance = np.var(block)
                bits = 3 if variance > self.variance_threshold else 1
                
                for i in range(block.shape[0]):
                    for j in range(block.shape[1]):
                        for channel in range(3):
                            for k in range(bits):
                                if data_ptr >= len(data_bin):
                                    return image
                                block[i,j,channel] = (block[i,j,channel] & ~(1 << k)) | ((int(data_bin[data_ptr]) << k))
                                data_ptr += 1
        return image

    def extract(self, image, data_length, block_size=8):
        h, w, c = image.shape
        data_bin = []
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = image[y:y+block_size, x:x+block_size]
                variance = np.var(block)
                bits = 3 if variance > self.variance_threshold else 1
                
                for i in range(block.shape[0]):
                    for j in range(block.shape[1]):
                        for channel in range(3):
                            for k in range(bits):
                                data_bin.append(str((block[i,j,channel] >> k) & 1))
        
        byte_strings = [''.join(data_bin[i:i+8]) for i in range(0, len(data_bin), 8)]
        valid_bytes = [int(b, 2) for b in byte_strings if len(b) == 8]
        return bytes(valid_bytes[:data_length])

# ==================== Visual Cryptography ====================
class VisualCryptography:
    def generate_shares(self, image):
        share1 = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        share2 = cv2.bitwise_xor(image, share1)
        return share1, share2

    def combine_shares(self, share1, share2):
        return cv2.bitwise_xor(share1, share2)

# ==================== Blockchain ====================
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data_str = str(self.data) if isinstance(self.data, dict) else self.data
        return hashlib.sha256(
            f"{self.index}{self.timestamp}{data_str}{self.previous_hash}".encode()
        ).hexdigest()

class Blockchain:
    def __init__(self, chain_file='blockchain.dat'):
        self.chain_file = chain_file
        self.chain = []
        self.load_chain()

    def create_genesis_block(self):
        return Block(0, datetime.now(), {'type': 'genesis'}, "0")

    def add_block(self, data):
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), datetime.now(), data, previous_block.hash)
        self.chain.append(new_block)
        self.save_chain()

    def verify_integrity(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            if current.hash != current.calculate_hash():
                return False
            if current.previous_hash != previous.hash:
                return False
        return True

    def save_chain(self):
        with open(self.chain_file, 'wb') as f:
            pickle.dump(self.chain, f)

    def load_chain(self):
        if os.path.exists(self.chain_file):
            with open(self.chain_file, 'rb') as f:
                self.chain = pickle.load(f)
        else:
            self.chain = [self.create_genesis_block()]
            self.save_chain()

# ==================== Secure File Transfer Protocol ====================
class SecureFileTransfer:
    def __init__(self, blockchain_file='shared_blockchain.dat'):
        self.lsb = AdaptiveLSB()
        self.vc = VisualCryptography()
        self.blockchain = Blockchain(blockchain_file)

    def send_file(self, image_path, file_data, output_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image file: {image_path}")
        
        stego_image = self.lsb.embed(image, file_data)
        share1, share2 = self.vc.generate_shares(stego_image)
        
        cv2.imwrite(f"{output_path}_share1.png", share1)
        cv2.imwrite(f"{output_path}_share2.png", share2)
        
        file_hash = hashlib.sha256(file_data).hexdigest()
        self.blockchain.add_block({
            'type': 'transfer',
            'file_hash': file_hash,
            'timestamp': str(datetime.now()),
            'status': 'sent'
        })
        
        return True

    def receive_file(self, share1_path, share2_path, data_length):
        share1 = cv2.imread(share1_path)
        share2 = cv2.imread(share2_path)
        if share1 is None or share2 is None:
            raise ValueError("Could not read share files")
        
        reconstructed = self.vc.combine_shares(share1, share2)
        extracted_data = self.lsb.extract(reconstructed, data_length)
        
        # Force blockchain reload
        self.blockchain.load_chain()
        
        file_hash = hashlib.sha256(extracted_data).hexdigest()
        
        for block in self.blockchain.chain:
            if isinstance(block.data, dict) and block.data.get('type') == 'transfer':
                if block.data.get('file_hash') == file_hash:
                    print("File integrity verified in blockchain")
                    return extracted_data
        
        print("File not found in blockchain records!")
        return None

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Configuration
    BLOCKCHAIN_FILE = "transfer_blockchain.dat"
    COVER_IMAGE = "cover_image.png"
    SECRET_FILE = "secret.txt"
    OUTPUT_PREFIX = "output_share"

    # Clean previous files for testing
    if os.path.exists(BLOCKCHAIN_FILE):
        os.remove(BLOCKCHAIN_FILE)
    if os.path.exists(COVER_IMAGE):
        os.remove(COVER_IMAGE)
    if os.path.exists(SECRET_FILE):
        os.remove(SECRET_FILE)

    # Create test files
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite(COVER_IMAGE, test_image)
    
    with open(SECRET_FILE, "wb") as f:
        f.write(b"This is a super secret message! 1234567890")

    # Initialize transfer protocol
    sender = SecureFileTransfer(BLOCKCHAIN_FILE)
    
    # Perform file transfer
    with open(SECRET_FILE, "rb") as f:
        secret_data = f.read()
    
    if sender.send_file(COVER_IMAGE, secret_data, OUTPUT_PREFIX):
        print("File sent successfully")
        
        # Initialize receiver AFTER sending
        receiver = SecureFileTransfer(BLOCKCHAIN_FILE)
        
        # Receive and verify file
        received_data = receiver.receive_file(
            f"{OUTPUT_PREFIX}_share1.png",
            f"{OUTPUT_PREFIX}_share2.png",
            len(secret_data)
        )
        
        if received_data:
            with open("received_secret.txt", "wb") as f:
                f.write(received_data)
            print("File transfer completed successfully!")
            print("Original hash:", hashlib.sha256(secret_data).hexdigest())
            print("Received hash:", hashlib.sha256(received_data).hexdigest())
        else:
            print("File transfer failed!")