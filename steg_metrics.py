import cv2
import numpy as np
import time
import hashlib
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from adaptive_lsb import AdaptiveLSB

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
            return 1.0  # 100% error if lengths differ
        xor_result = np.bitwise_xor(
            np.frombuffer(self.secret_data, dtype=np.uint8),
            np.frombuffer(self.extracted_data, dtype=np.uint8)
        )
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

    def time_encoding(self, embed_func):
        start = time.perf_counter()
        test_stego = embed_func(self.original.copy(), self.secret_data)
        self.encoding_time = time.perf_counter() - start

    def time_decoding(self, extract_func):
        start = time.perf_counter()
        extract_func(self.stego.copy(), len(self.secret_data))
        self.decoding_time = time.perf_counter() - start

    def rs_analysis(self):
        # Corrected RS Analysis with proper mask dimensions
        mask = np.random.randint(0, 2, self.stego.shape, dtype=np.uint8)
        flipped = self.stego ^ mask
        return float(np.abs(np.mean(self.stego) - np.mean(flipped)))

    def plot_all_metrics(self):
        metrics = self.calculate_all_metrics()
        plt.figure(figsize=(20, 15))
        plt.suptitle("Comprehensive Steganography Analysis", fontsize=16, y=1.02)
        
        # 1. Quality Metrics Radar Chart
        plt.subplot(231, polar=True)
        quality_labels = ["PSNR (dB)", "SSIM", "MSE", "NPCR (%)", "UACI (%)"]
        quality_values = [
            metrics["Quality Metrics"]["PSNR (dB)"] / 100,
            metrics["Quality Metrics"]["SSIM"],
            1 - metrics["Quality Metrics"]["MSE"],
            metrics["Quality Metrics"]["NPCR (%)"] / 100,
            metrics["Quality Metrics"]["UACI (%)"] / 100
        ]
        angles = np.linspace(0, 2 * np.pi, len(quality_labels), endpoint=False).tolist()
        angles += angles[:1]
        quality_values += quality_values[:1]
        plt.polar(angles, quality_values, marker='o')
        plt.fill(angles, quality_values, alpha=0.25)
        plt.title("Image Quality Radar", pad=20)
        plt.xticks(angles[:-1], quality_labels)

        # 2. Time Efficiency Comparison
        plt.subplot(232)
        times = [
            metrics["Performance Metrics"]["Encoding Time (s)"],
            metrics["Performance Metrics"]["Decoding Time (s)"]
        ]
        plt.bar(["Encoding", "Decoding"], times, color=['#2ecc71', '#e74c3c'])
        plt.ylabel("Time (seconds)")
        plt.title("Processing Time Comparison")

        # 3. Security Analysis
        plt.subplot(233)
        security_metrics = [
            metrics["Security Metrics"]["RS Analysis (Δ)"],
            metrics["Security Metrics"]["Bit Error Rate"],
            metrics["Security Metrics"]["Visual Detection Risk"]
        ]
        plt.barh(["RS Analysis", "Bit Errors", "Visual Risk"], security_metrics, color='#3498db')
        plt.xlim(0, 1)
        plt.title("Security Vulnerability Profile")

        # 4. Capacity Utilization
        plt.subplot(234)
        utilization = metrics["Capacity Metrics"]["Utilization (%)"]
        plt.pie([utilization, 100 - utilization], 
                labels=["Used", "Available"],
                colors=['#9b59b6', '#ecf0f1'],
                autopct='%1.1f%%',
                startangle=90)
        plt.title("Capacity Utilization Ratio")

        # 5. Error Components
        plt.subplot(235)
        errors = [
            metrics["Security Metrics"]["Bit Error Rate"],
            metrics["Security Metrics"]["Visual Detection Risk"],
            metrics["Security Metrics"]["RS Analysis (Δ)"]
        ]
        plt.bar(["Bit Errors", "Visual Risk", "RS Analysis"], errors, color='#e67e22')
        plt.ylim(0, 1)
        plt.title("Error Component Analysis")

        # 6. Data Integrity Check
        plt.subplot(236)
        integrity = [
            int(metrics["Integrity Metrics"]["Hash Match"]),
            int(metrics["Integrity Metrics"]["File Size Match"]),
            int(metrics["Integrity Metrics"]["Content Match"])
        ]
        plt.bar(["Hash", "Size", "Content"], integrity, 
                color=['#2ecc71' if x else '#e74c3c' for x in integrity])
        plt.ylim(0, 1)
        plt.title("Data Integrity Checks")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    metrics = StegMetrics("cover_image.png", "reconstructed.png", "secret.txt", "extracted.bin")
    lsb = AdaptiveLSB()
    
    # Time operations
    metrics.time_encoding(lsb.embed)
    metrics.time_decoding(lsb.extract)
    
    # Generate report
    results = metrics.calculate_all_metrics()
    
    print("\nComplete Steganography Analysis Report")
    print("======================================")
    for category, values in results.items():
        print(f"\n{category}:")
        for metric, value in values.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, (float, int)) else f"  {metric}: {value}")
    
    metrics.plot_all_metrics()