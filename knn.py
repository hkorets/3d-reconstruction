import numpy as np

def knn_match_manual(desc1, desc2, kp1, kp2, ratio_thresh=0.75):
    """
    Manual implementation of KNN matching + Lowe's ratio test.

    :param desc1: numpy array of descriptors from image 1 (shape: Nx128)
    :param desc2: numpy array of descriptors from image 2 (shape: Mx128)
    :param kp1: list of (x, y) coordinates for keypoints in image 1
    :param kp2: list of (x, y) coordinates for keypoints in image 2
    :param ratio_thresh: Lowe's ratio threshold (default: 0.75)
    :return: list of matched point pairs as (x1, y1, x2, y2)
    """
    matches = []

    for i, d1 in enumerate(desc1):
        # Compute L2 distances to all descriptors in image 2
        distances = np.linalg.norm(desc2 - d1, axis=1)

        # Find the two closest descriptors
        idx_sorted = np.argsort(distances)
        best_idx, second_idx = idx_sorted[0], idx_sorted[1]
        best_dist, second_dist = distances[best_idx], distances[second_idx]

        # Apply Lowe's ratio test
        if best_dist < ratio_thresh * second_dist:
            pt1 = kp1[i]
            pt2 = kp2[best_idx]
            matches.append((pt1[0], pt1[1], pt2[0], pt2[1]))

    return matches
