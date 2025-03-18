import cv2
import numpy as np
import time
import hashlib
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

class AdaptiveLSB:
    def __init__(self, threshold=30):
        self.threshold = threshold

    def _pixel_complexity(self, img_block):
        if img_block.shape != (3, 3, 3):
            return 0
        return np.std(img_block)

    def embed(self, cover, secret_data):
        binary_data = ''.join(format(byte, '08b') for byte in secret_data)
        data_index = 0
        height, width, _ = cover.shape
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                if data_index >= len(binary_data):
                    break
                block = cover[i-1:i+2, j-1:j+2]
                if self._pixel_complexity(block) > self.threshold:
                    for channel in range(3):
                        if data_index >= len(binary_data):
                            break
                        cover[i,j,channel] = (cover[i,j,channel] & 0xFE) | int(binary_data[data_index])
                        data_index += 1
        return cover

    def extract(self, stego, data_length):
        binary_str = ''
        height, width, _ = stego.shape
        extracted = 0
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                if extracted >= data_length * 8:
                    break
                for channel in range(3):
                    if extracted >= data_length * 8:
                        break
                    binary_str += str(stego[i,j,channel] & 1)
                    extracted += 1
                    
        bytes_data = bytearray()
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i+8].ljust(8, '0')
            bytes_data.append(int(byte, 2))
        return bytes(bytes_data)

class StegMetrics:
    def __init__(self, original_path, stego_path, secret_path, extracted_path):
        try:
            self.original = cv2.imread(original_path)
            self.stego = cv2.imread(stego_path)
            if self.original is None or self.stego is None:
                raise ValueError("Error loading images. Check the file paths.")

            with open(secret_path, 'rb') as f:
                self.secret_data = f.read()
            with open(extracted_path, 'rb') as f:
                self.extracted_data = f.read()
        except Exception as e:
            raise ValueError(f"File loading error: {e}")

        self.encoding_time = 0
        self.decoding_time = 0
        self.rs_analysis_value = self.rs_analysis()

    def calculate_all_metrics(self):
        return {
            "Quality Metrics": self.image_quality(),
            "Performance Metrics": self.performance(),
            "Capacity Metrics": self.capacity(),
            "Security Metrics": self.security_analysis(),
            "Integrity Metrics": self.file_integrity()
        }

    def image_quality(self):
        return {
            "PSNR (dB)": self.psnr(),
            "SSIM": self.ssim(),
            "MSE": self.mse(),
            "NPCR (%)": self.npcr(),
            "UACI (%)": self.uaci()
        }

    def performance(self):
        return {
            "Encoding Time (s)": self.encoding_time,
            "Decoding Time (s)": self.decoding_time
        }

    def capacity(self):
        return {
            "Payload Capacity (bpp)": self.bpp(),
            "Max Theoretical Capacity (bits)": self.max_capacity(),
            "Utilization (%)": self.capacity_utilization()
        }

    def security_analysis(self):
        return {
            "RS Analysis (Δ)": self.rs_analysis_value,
            "Bit Error Rate": self.ber(),
            "Visual Detection Risk": self.visual_risk()
        }

    def file_integrity(self):
        return {
            "Hash Match": self.hash_match(),
            "File Size Match": self.size_match(),
            "Content Match": self.content_match()
        }

    def psnr(self):
        mse = self.mse()
        return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    def ssim(self):
        return ssim(cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(self.stego, cv2.COLOR_BGR2GRAY))

    def mse(self):
        return np.mean((self.original - self.stego) ** 2)

    def npcr(self):
        return np.mean(self.original != self.stego) * 100

    def uaci(self):
        return np.mean(np.abs(self.original.astype(int) - self.stego.astype(int))) / 255 * 100

    def bpp(self):
        pixels = self.original.shape[0] * self.original.shape[1]
        return (len(self.secret_data) * 8) / pixels if pixels > 0 else 0

    def max_capacity(self):
        return self.original.shape[0] * self.original.shape[1] * 3

    def capacity_utilization(self):
        max_cap = self.max_capacity()
        return (len(self.secret_data) * 8 / max_cap) * 100 if max_cap > 0 else 0

    def ber(self):
        if len(self.secret_data) != len(self.extracted_data):
            return 1.0
        xor_result = np.bitwise_xor(
            np.frombuffer(self.secret_data, dtype=np.uint8),
            np.frombuffer(self.extracted_data, dtype=np.uint8))
        return np.sum(np.unpackbits(xor_result)) / (len(self.secret_data) * 8)

    def visual_risk(self):
        diff = cv2.absdiff(self.original, self.stego)
        return np.mean(diff) / 255

    def hash_match(self):
        return hashlib.sha256(self.secret_data).hexdigest() == hashlib.sha256(self.extracted_data).hexdigest()

    def size_match(self):
        return len(self.secret_data) == len(self.extracted_data)

    def content_match(self):
        return self.secret_data == self.extracted_data

    def rs_analysis(self):
        mask1 = np.array([[[0,1,0]] * self.stego.shape[1]] * self.stego.shape[0], dtype=np.uint8)
        mask2 = np.array([[[1,0,1]] * self.stego.shape[1]] * self.stego.shape[0], dtype=np.uint8)
        flipped1 = self.stego ^ mask1
        flipped2 = self.stego ^ mask2
        delta = (np.abs(np.mean(self.stego) - np.mean(flipped1)) + 
                np.abs(np.mean(self.stego) - np.mean(flipped2))) / 2
        return float(delta)

    def plot_all_metrics(self):
        metrics = self.calculate_all_metrics()
        plt.figure(figsize=(22, 18))
        plt.suptitle("Comprehensive Steganography Analysis", fontsize=18, y=1.02)
        plt.subplots_adjust(hspace=0.5, wspace=0.4)

        # 1. Quality Metrics Radar Chart
        plt.subplot(231, polar=True)
        quality_labels = ["PSNR (dB)", "SSIM", "MSE", "NPCR (%)", "UACI (%)"]
        quality_values = [
            metrics["Quality Metrics"]["PSNR (dB)"] / 100,  # Assuming PSNR <= 100 dB
            metrics["Quality Metrics"]["SSIM"],
            1 - (metrics["Quality Metrics"]["MSE"] / 255**2),  # Normalize MSE to 0-1
            metrics["Quality Metrics"]["NPCR (%)"] / 100,
            metrics["Quality Metrics"]["UACI (%)"] / 100
        ]
        angles = np.linspace(0, 2 * np.pi, len(quality_labels), endpoint=False).tolist()
        angles += angles[:1]
        quality_values += quality_values[:1]
        plt.polar(angles, quality_values, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.fill(angles, quality_values, alpha=0.25)
        plt.title("Image Quality Radar", pad=25)
        plt.xticks(angles[:-1], quality_labels, fontsize=10, rotation=30)

        # 2. Time Efficiency Comparison
        plt.subplot(232)
        times = [
            metrics["Performance Metrics"]["Encoding Time (s)"],
            metrics["Performance Metrics"]["Decoding Time (s)"]
        ]
        bars = plt.bar(["Encoding", "Decoding"], times, color=['#2ecc71', '#e74c3c'])
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.title("Processing Time Comparison", pad=20)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom')

        # 3. Security Analysis
        plt.subplot(233)
        security_metrics = [
            metrics["Security Metrics"]["RS Analysis (Δ)"],
            metrics["Security Metrics"]["Bit Error Rate"],
            metrics["Security Metrics"]["Visual Detection Risk"]
        ]
        plt.barh(["RS Analysis", "Bit Errors", "Visual Risk"], security_metrics, color='#3498db')
        plt.xlim(0, 1)
        plt.title("Security Vulnerability Profile", pad=20)
        for i, v in enumerate(security_metrics):
            plt.text(v + 0.02, i, f"{v:.4f}", color='black', va='center')

        # 4. Capacity Utilization
        plt.subplot(234)
        utilization = metrics["Capacity Metrics"]["Utilization (%)"]
        plt.pie([utilization, 100 - utilization], 
                labels=["Used", "Available"], 
                colors=['#9b59b6', '#ecf0f1'], 
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        plt.title("Capacity Utilization Ratio", pad=20)

        # 5. Error Components
        plt.subplot(235)
        errors = [
            metrics["Security Metrics"]["Bit Error Rate"],
            metrics["Security Metrics"]["Visual Detection Risk"],
            metrics["Security Metrics"]["RS Analysis (Δ)"]
        ]
        bars = plt.bar(["Bit Errors", "Visual Risk", "RS Analysis"], errors, color='#e67e22')
        plt.ylim(0, 1)
        plt.title("Error Component Analysis", pad=20)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom')

        # 6. Data Integrity Check
        plt.subplot(236)
        integrity = [
            int(metrics["Integrity Metrics"]["Hash Match"]),
            int(metrics["Integrity Metrics"]["File Size Match"]),
            int(metrics["Integrity Metrics"]["Content Match"])
        ]
        colors = ['#2ecc71' if x else '#e74c3c' for x in integrity]
        bars = plt.bar(["Hash", "Size", "Content"], integrity, color=colors)
        plt.ylim(0, 1)
        plt.title("Data Integrity Checks", pad=20)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     "Pass" if height == 1 else "Fail", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Load cover image and secret data
    cover_image = cv2.imread("cover_image.png")
    with open("secret.txt", "rb") as f:
        secret_data = f.read()
    
    # Embed the secret data and save the stego image
    lsb = AdaptiveLSB()
    
    # Time encoding and save the new stego
    start = time.perf_counter()
    stego_image = lsb.embed(cover_image.copy(), secret_data)
    encoding_time = time.perf_counter() - start
    cv2.imwrite("reconstructed.png", stego_image)
    
    # Initialize metrics AFTER embedding (loads fresh stego image)
    metrics = StegMetrics("cover_image.png", "reconstructed.png", "secret.txt", "extracted.bin")
    metrics.encoding_time = encoding_time
    
    # Time decoding using the new stego image
    start = time.perf_counter()
    extracted_data = lsb.extract(stego_image, len(secret_data))
    metrics.decoding_time = time.perf_counter() - start
    
    # Save extracted data for integrity checks
    with open("extracted.bin", "wb") as f:
        f.write(extracted_data)
    
    # Generate report and plot
    results = metrics.calculate_all_metrics()
    
    print("\nComplete Steganography Analysis Report")
    print("======================================")
    for category, values in results.items():
        print(f"\n{category}:")
        for metric, value in values.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, (float, int)) else f"  {metric}: {value}")
    
    metrics.plot_all_metrics()