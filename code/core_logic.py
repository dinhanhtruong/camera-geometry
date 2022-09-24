import numpy as np
import cv2
import random

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix by setting up a system of
    equations using the corresponding 2D and 3D points. 

    

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in the scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)  # [n, 2]
    points3d = np.array(points3d)  # [n, 3]


    A = np.zeros((2*points2d.shape[0], 11)) # [2n, 11]
    i = 0
    for XYZ, uv in zip(points3d, points2d):
        # set first row for current point
        A[i][0] = XYZ[0]
        A[i][1] = XYZ[1]
        A[i][2] = XYZ[2]
        A[i][3] = 1
        A[i][8] = -XYZ[0]*uv[0]
        A[i][9] = -XYZ[1]*uv[0]
        A[i][10] = -XYZ[2]*uv[0]
        # second row
        A[i + 1][4] = XYZ[0]
        A[i + 1][5] = XYZ[1]
        A[i + 1][6] = XYZ[2]
        A[i + 1][7] = 1
        A[i + 1][8] = -XYZ[0]*uv[1]
        A[i + 1][9] = -XYZ[1]*uv[1]
        A[i + 1][10] = -XYZ[2]*uv[1]
        i += 2
    b = np.reshape(points2d, (-1, 1)) # [2n, 1]

    M = np.linalg.lstsq(A, b)[0]
    # add scale factor 1 to last element of M
    M = np.concatenate([M, [[1]]], axis=0)
    M = np.reshape(M, (3, 4))

    return M


def normalize_coordinates(points):
    pass


def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """

    # This is an intentionally incorrect Fundamental matrix placeholder
    n = points1.shape[0]  # should be ~8
    A = np.zeros((n, 9)) 
    for i, (uv1, uv2) in enumerate(zip(points1, points2)):
        u1, v1 = uv1
        u2, v2 = uv2
        # fill in i-th row
        A[i] = np.array([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1])
    # SVD
    V = np.linalg.svd(A)[2]
    f = V[-1] # last row of V
    F_matrix = np.reshape(f, (3, 3))
    # make F rank 2
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh 

    return F_matrix


def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Run RANSAC for num_iters.

    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points from two pictures. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    best_Fmatrix is the [3 x 3] fundamental matrix, inliers1 and inliers2 are
    the [M x 2] corresponding points (some subset of matches1 and matches2) that
    are inliners with respect to best_Fmatrix

    For this section, use RANSAC to find the best fundamental matrix by randomly
    sampling interest points. 

    :return: best_Fmatrix, inliers1, inliers2
    """
    random.seed(0)
    np.random.seed(0)
    

    best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    inliers_a = matches1[0:29, :]
    inliers_b = matches2[0:29, :]
    
    num_matches = matches1.shape[0]
    max_num_inliers = 0
    threshold = 0.025
    i = 0
    while i < num_iters:
        if i % 100 == 0:
            print("iter: " ,i)
        rand_indices = np.random.choice(num_matches, size=12, replace=False) # 8 random ints in [0,...,N-1]
        subset1 = matches1[rand_indices]
        subset2 = matches2[rand_indices]
        # F_matrix, _ = cv2.findFundamentalMat(subset1, subset2, cv2.FM_8POINT, 1e10, 0, 1)
        F_matrix = estimate_fundamental_matrix(subset1, subset2)
        # count inliers out of all possible correspondences
        # dist metric = val of xTFx' (from GT=0) for homogeneous x/x'   [1,3] [3,3] [3,1]
        matches1_homogenous = np.concatenate([matches1, np.ones((num_matches, 1))], axis=1)
        matches2_homogenous = np.concatenate([matches2, np.ones((num_matches, 1))], axis=1)
        distances = []
        for x1, x2 in zip(matches1_homogenous, matches2_homogenous):
            x1_T = np.reshape(x1, (1, 3))
            x2 = np.reshape(x2, (3,1))
            distances.append(x1_T @ F_matrix @ x2)
        distances = np.squeeze(np.array(distances))
        num_inliers = np.size(distances[np.absolute(distances) < threshold])
        # update best F and corresponding inliers if necessary
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            best_Fmatrix = F_matrix
            inliers_a = matches1[np.absolute(distances) < threshold]
            inliers_b = matches2[np.absolute(distances) < threshold]
            print("new max inliers %: ", 100*max_num_inliers/num_matches)
        i += 1
    print("final max inliers %: ", 100*max_num_inliers/num_matches)
    return best_Fmatrix, inliers_a, inliers_b


def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, solves
    for the ground-truth 3D points using np.linalg.lstsq(). 

    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image2
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] list of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    points3d = []
    n = points1.shape[0]
    A = np.zeros((4, 3)) # [4, 3]  => 4 rows/eqs for each example
    b = np.zeros((4, 1)) # [4, 1]
    for uv1, uv2 in zip(points1, points2):
        u1, v1 = uv1
        u2, v2 = uv2
        # set first row for current point
        A[0][0] = M1[2][0]*u1 - M1[0][0]
        A[0][1] = M1[2][1]*u1 - M1[0][1]
        A[0][2] = M1[2][2]*u1 - M1[0][2]
        b[0] = M1[0, 3] - M1[2, 3]*u1
        # second row
        A[1][0] = M1[2][0]*v1 - M1[1][0]
        A[1][1] = M1[2][1]*v1 - M1[1][1]
        A[1][2] = M1[2][2]*v1 - M1[1][2]
        b[1] = M1[1, 3] - M1[2, 3]*v1
        # third row
        A[2][0] = M2[2][0]*u2 - M2[0][0]
        A[2][1] = M2[2][1]*u2 - M2[0][1]
        A[2][2] = M2[2][2]*u2 - M2[0][2]
        b[2] = M2[0, 3] - M2[2, 3]*u2
        # second row
        A[3][0] = M2[2][0]*v2 - M2[1][0]
        A[3][1] = M2[2][1]*v2 - M2[1][1]
        A[3][2] = M2[2][2]*v2 - M2[1][2]
        b[3] = M2[1, 3] - M2[2, 3]*v2
        points3d.append(np.squeeze(np.linalg.lstsq(A, b)[0])) # append this point [3,1] to points3d
        
    return points3d
