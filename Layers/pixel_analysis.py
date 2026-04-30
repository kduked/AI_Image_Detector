import cv2
import numpy as np

def analyze_noise(image_gray):
    """Analyze noise patterns in the image"""
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    noise = cv2.subtract(image_gray, blurred)
    noise = noise.astype(np.float32)
    
    noise_variance = np.var(noise)
    noise_std_dev = np.std(noise)
    
    return {
        'variance': float(noise_variance),
        'std_dev': float(noise_std_dev),
        'suspicious': noise_std_dev < 10.0   # stricter
    }


def analyze_frequency(image_gray):
    """Analyze frequency domain patterns"""
    fft = np.fft.fft2(image_gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    
    high_freq_region = magnitude.copy()
    high_freq_region[center_h-30:center_h+30, center_w-30:center_w+30] = 0
    
    high_freq_energy = np.sum(high_freq_region)
    total_energy = np.sum(magnitude)
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
    
    return {
        'high_freq_ratio': float(high_freq_ratio),
        'suspicious': high_freq_ratio < 0.9   # stricter
    }


def analyze_edges(image_gray):
    """Analyze edge consistency and patterns"""
    edges = cv2.Canny(image_gray, 100, 200)
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = edge_pixels / total_pixels
    
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    edge_mean = np.mean(gradient_magnitude)
    edge_std = np.std(gradient_magnitude)
    
    return {
        'density': float(edge_density),
        'mean': float(edge_mean),
        'std': float(edge_std),
        'suspicious': edge_density < 0.1 or edge_std < 20   # stricter
    }


def analyze_texture(image_gray):
    """Analyze texture uniformity and repetition"""
    block_size = 32
    h, w = image_gray.shape
    block_variances = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = image_gray[i:i+block_size, j:j+block_size]
            block_variances.append(np.var(block))
    
    texture_variance = np.var(block_variances) if block_variances else 0
    texture_mean = np.mean(block_variances) if block_variances else 0
    
    return {
        'variance': float(texture_variance),
        'mean': float(texture_mean),
        'suspicious': texture_variance < 500000   # stricter
    }


def analyze_color(image_color):
    """Analyze color distribution and entropy"""
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    
    r_channel = image_rgb[:, :, 0]
    g_channel = image_rgb[:, :, 1]
    b_channel = image_rgb[:, :, 2]
    
    r_std = np.std(r_channel)
    g_std = np.std(g_channel)
    b_std = np.std(b_channel)
    
    avg_color_std = (r_std + g_std + b_std) / 3
    
    hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])
    
    hist_r = hist_r / np.sum(hist_r)
    hist_g = hist_g / np.sum(hist_g)
    hist_b = hist_b / np.sum(hist_b)
    
    entropy_r = -np.sum(hist_r * np.log2(hist_r + 1e-10))
    entropy_g = -np.sum(hist_g * np.log2(hist_g + 1e-10))
    entropy_b = -np.sum(hist_b * np.log2(hist_b + 1e-10))
    avg_entropy = (entropy_r + entropy_g + entropy_b) / 3
    
    return {
        'avg_std': float(avg_color_std),
        'avg_entropy': float(avg_entropy),
        'suspicious': avg_color_std < 70 or avg_entropy < 7.5   # stricter
    }

def get_pixel_analysis_results(image_path):
   
    image_color = cv2.imread(image_path)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image_color is None or image_gray is None:
        return None
    
    results = {
        'noise': analyze_noise(image_gray),
        'frequency': analyze_frequency(image_gray),
        'edges': analyze_edges(image_gray),
        'texture': analyze_texture(image_gray),
        'color': analyze_color(image_color)
    }
    
    # Calculate overall suspicion
    suspicious_count = sum([
        results['noise']['suspicious'],
        results['frequency']['suspicious'],
        results['edges']['suspicious'],
        results['texture']['suspicious'],
        results['color']['suspicious']
    ])
    
    results['suspicion_score'] = suspicious_count / 5.0
    
    # Keep main logic the same — still considers 3 or more suspicious as likely AI
    results['likely_ai'] = suspicious_count >= 3
    
    return results
