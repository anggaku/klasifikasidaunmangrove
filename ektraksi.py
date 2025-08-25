import cv2
import numpy as np
import math
from skimage.feature import graycomatrix, graycoprops

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    # --- Shape Features ---
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    leaf_contour = max(contours, key=cv2.contourArea)
    points = leaf_contour[:, 0, :]

    # Calculate maximum length (major axis)
    max_length = 0
    tip = base = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_length:
                max_length = dist
                tip, base = points[i], points[j]

    # Calculate width (minor axis) perpendicular to length
    dy = base[1] - tip[1]
    dx = base[0] - tip[0]
    length_slope = dy / dx if dx != 0 else np.inf
    width_slope = -1 / length_slope if length_slope not in [0, np.inf] else np.inf

    max_width = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            slope = dy / dx if dx != 0 else np.inf
            if abs(slope - width_slope) < 0.2:
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_width:
                    max_width = dist

    # Calculate all shape features
    area = cv2.contourArea(leaf_contour)
    perimeter = cv2.arcLength(leaf_contour, True)
    aspect_ratio = max_length / max_width if max_width != 0 else 0
    circularity = (4 * math.pi * area) / (perimeter**2) if perimeter != 0 else 0
    rectangularity = area / (max_length * max_width) if (max_length * max_width) != 0 else 0
    diameter = max([np.linalg.norm(p1 - p2) for p1 in points for p2 in points])
    # narrow_factor = diameter / max_length if max_length != 0 else 0
    # ratio_perim_diam = perimeter / diameter if diameter != 0 else 0
    # ratio_perim_lenwidth = perimeter / (max_length + max_width) if (max_length + max_width) != 0 else 0

    # --- Color Features ---
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    leaf_pixels = hsv[mask > 0]
    if len(leaf_pixels) == 0:
        return None

    mean_hsv = np.mean(leaf_pixels, axis=0)  # 3 features (H,S,V)
    std_hsv = np.std(leaf_pixels, axis=0)    # 3 features (H,S,V)
    hist_hue, _ = np.histogram(leaf_pixels[:, 0], bins=8, range=(0, 180))  # 16 features
    hist_hue = hist_hue / hist_hue.sum()
    color_features = np.concatenate([mean_hsv, std_hsv, hist_hue])  # 22 color features total

    # --- Texture Features ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (128, 128))
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_resized, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    texture_features = [np.mean(graycoprops(glcm, p)[0]) for p in props]  # 4 features
    glcm_sum = np.sum(glcm, axis=3)
    glcm_prob = glcm_sum / np.sum(glcm_sum)
    entropy = -np.sum(glcm_prob * np.log2(glcm_prob + 1e-10))  # 1 feature
    texture_features.append(entropy)  # 5 texture features total

    # Combine all features (9 shape + 22 color + 5 texture = 36 features)
    features = np.concatenate([
        [area, perimeter, aspect_ratio, circularity, rectangularity, 
         diameter],
        color_features,
        texture_features
    ])
    
    return features