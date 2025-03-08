import cv2
import numpy as np
import hashlib
import os
import pickle
from datetime import datetime

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