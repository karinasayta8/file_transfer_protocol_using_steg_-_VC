# blockchain.py
import cv2
import numpy as np
import hashlib
import os
import pickle
from datetime import datetime


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
        """Force-save the blockchain"""
        with open(self.chain_file, 'wb') as f:
            pickle.dump(self.chain, f)

    def load_chain(self):
        if os.path.exists(self.chain_file):
            with open(self.chain_file, 'rb') as f:
                self.chain = pickle.load(f)
        else:
            self.chain = [self.create_genesis_block()]
            self.save_chain()

    def verify_file(self, file_hash):
        """Enhanced verification with debug"""
        print("\n🔎 Blockchain Verification Process:")
        for idx, block in enumerate(self.chain):
            if isinstance(block.data, dict) and block.data.get('type') == 'transfer':
                print(f" Checking Block {idx}: {block.data.get('file_hash')}")
                if block.data.get('file_hash') == file_hash:
                    return True
        return False

    def print_chain(self):
        """Display all blockchain transactions"""
        print("\n🔗 Blockchain Transactions:")
        for block in self.chain:
            print(f"\nBlock #{block.index}")
            print(f"Timestamp: {block.timestamp}")
            print(f"Hash: {block.hash[:16]}...")
            
            if isinstance(block.data, dict):
                print("Type:", block.data.get('type', 'unknown'))
                if block.data.get('type') == 'transfer':
                    print("File Hash:", block.data.get('file_hash'))
                    print("Status:", block.data.get('status'))
            else:
                print("Data:", block.data[:50] + "...")
                
            print(f"Previous Hash: {block.previous_hash[:16]}...")