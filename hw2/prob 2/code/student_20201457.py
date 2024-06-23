import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage import convolve
from skimage.feature import peak_local_max
from student_proj1 import my_gauss_2D #과제1에서 구현한 함수를 재활용했습니다.

def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    Harris corner detector를 구현하세요. 수업시간에 배운 간단한 형태의 알고리즘만 구현해도 됩니다.
    스케일scale, 방향성orientation 등은 추후에 고민해도 됩니다.
    Implement the Harris corner detector (See Szeliski 4.1.1).
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    원한다면 다른 종류의 특징점 정합 기법을 구현해도 됩니다.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    만약 영상의 에지 근처에서 잘못된 듯한 특징점이 도출된다면 에지 근처의 특징점을 억제해 버리는 코드를 추가해도 됩니다.
    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    유용한 함수: 제시해 드리는 모든 함수를 꼭 활용해야 하는 건 아닙니다만, 이중 일부가 여러분에게 유용할 수 있습니다.
    각 함수는 웹에서 제공되는 documentation을 참고해서 사용해 보세요.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. 

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :입력 인자params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width of the range of pixels that effect the interst point detection
                    (size of the local gaussian filtering)

    :반환값returns:
    :kpts: an np array of dimension k x 2 where k is the number of interest points. 
        The first column is the x coordinates and the second column is the y coordinates.

    :옵션으로 추가할 수 있는 반환값optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''
    # TODO: Your implementation here! See block comments and the project webpage for instructions
    kerdx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kerdy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    ix = convolve(image, kerdx)
    iy = convolve(image, kerdy)

    #가우시안 s
    s = 1.7

    #Harris 파라미터
    k = 0.04 # ~0.04 to 0.06
    
    gauss_filter_2D = my_gauss_2D(s, feature_width)
    
    ixx = convolve(ix**2, gauss_filter_2D)
    iyy = convolve(iy**2, gauss_filter_2D)
    ixy = convolve(ix * iy, gauss_filter_2D)

    det = (ixx * iyy) - (ixy**2)
    trace = ixx + iyy
    C = det - k * trace

    thresh = 0.01 * np.max(C) #gaudi 이미지에서는 0.08로 둬야함
    corner_candidates = C > thresh
    coordinates = peak_local_max(C, min_distance=1, threshold_abs=thresh)
    # These are placeholders - replace with the coordinates of your interest points!
    kpts = np.zeros((1,2),dtype=int)

    # End of placeholders
    kpts = coordinates

    #x, y 변환
    for i in range(len(kpts)):
        kpts[i][0], kpts[i][1] = kpts[i][1], kpts[i][0]
    
    return kpts



def match_features_nn(im1_features, im2_features):
    '''
    가장 까까운 이웃 거리 비율 테스트를 구현하여 두 이미지의 feature point들 간에 대응점 쌍 후보들을 찾도록 하세요.
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Szeliski. section 4.1.3에 있는 "가장 가까운 거리 비율 테스트(NNDR)" 방정식을 구현하십시오.
    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.
    
    ######################################################################
    
    1. 특징 벡터의 Euclidean distance를 구하고 정렬한다.
    2. 가장 작은 거리를 찾는다.
    3. 두 거리의 비율을 계산하고 threshold를 넘는지 확인한다.
    
    ######################################################################
    
    :params:
    :im1_features: an np array of SIFT features for interest points in image1
    :im2_features: an np array of SIFT features for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    matches     = np.zeros((1,2))
    confidences = np.zeros(1)
    
    num_feat1 = im1_features.shape[0]
    num_feat2 = im2_features.shape[0]
    
    dis = np.empty((num_feat1, num_feat2))

    for i in range(num_feat1):
        for j in range(num_feat2):
            dis[i, j] = np.sqrt(np.sum((im1_features[i] - im2_features[j])**2))
            
    sorted_ind = np.argsort(dis, axis=1)
    nearest_ind = sorted_ind[:, 0]
    second_nearest_ind = sorted_ind[:, 1]

    # 거리 비율 계산
    ratio = dis[np.arange(num_feat1), nearest_ind] / dis[np.arange(num_feat1), second_nearest_ind]

    # 임계값 설정 
    threshold = 0.8 #보고서에서는 기본으로 0.8설정, 실험값으로 1.6으로 둠
    valid_matches = ratio < threshold

    # 유효한 매치와 신뢰도 계산
    matches = np.column_stack((np.arange(num_feat1)[valid_matches], nearest_ind[valid_matches]))
    confidences = 1 - ratio[valid_matches]

    # 만약 매치가 없다면, 빈 배열을 반환
    if matches.size == 0:
        return np.zeros((0, 2)), np.zeros(0)

    # End of placeholders

    return matches, confidences


def get_homography(kpts1, kpts2, matches, sample_idxs):
    '''
    Corresponding keypoint pair들을 이용하여 Homography를 계산하는 코드를 구현하시오. 

    ######################################################################
    
    1. Ax = b 형식의 수식에 대응점 쌍 좌표들을 이용하여 행렬 A와 b를 정의하고 원소 값을 대입함 
      (강의08 슬라이드 38페이지 참고)
    2. 선형방정식 풀이법을 적절하게 이용하여 homography H를 계산함
    
    ######################################################################
    
    :params:
    :kpts1: an np array of dimension k x 2 where k is the number of interest points in image 1. 
        The first column is the x coordinates and the second column is the y coordinates.
    :kpts2: an np array of dimension k x 2 where k is the number of interest points in image 2. 
        The first column is the x coordinates and the second column is the y coordinates.
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into kpts1 and the second column is an index into kpts2
    :sample_idxs: an array of random indices to sample corresponding keypoint pairs

    :returns:
    :H: the homography computed from the matched points
    '''
    
    # placeholder - replace with your H!
    H = np.eye(3)
    
    # TODO: Your implementation here! See block comments and the project webpage for instructions
    pts1 = kpts1[matches[sample_idxs, 0]]
    pts2 = kpts2[matches[sample_idxs, 1]]
    num_matches = len(sample_idxs)

    A = np.zeros((2 * num_matches, 9))

    for i in range(num_matches):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]

    # SVD를 사용하여 Ax = 0 문제 해결
    U, s, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))  # 마지막 특이 벡터를 3x3 행렬로 변환

    return H




def match_features_ransac(kpts1, kpts2, matches):
    '''
    Homography를 기반으로 하는 RANSAC 알고리즘을 구현하시오. 

    ######################################################################
    
    1. 4개의 특징점 대응 쌍을 임의로 샘플링함
    2. 샘플링된 대응점 쌍을 이용하여 homography H를 계산함 - get_homography() 함수 구현 및 호출
    3. 계산된 H를 이용하여 전체 대응점 쌍을 inlier와 outlier로 구분하고 inlier 개수 측정함
    4. 이 과정을 RANSAC parameter 계산 수식으로 도출된 횟수만큼 반복함
    
    ######################################################################
    
    :params:
    :kpts1: an np array of dimension k x 2 where k is the number of interest points in image 1. 
        The first column is the x coordinates and the second column is the y coordinates.
    :kpts2: an np array of dimension k x 2 where k is the number of interest points in image 2. 
        The first column is the x coordinates and the second column is the y coordinates.
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into kpts1 and the second column is an index into kpts2

    :returns:
    :matches_in: an np array of dimension k x 2 where k is the number of INLIER matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :H: the homography computed from the matched points
    '''
    
    # These are placeholders - replace with your matches and confidences!
    matches_in    = np.zeros((1,2))
    H = np.eye(3)

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    num_iter = 1000
    threshold = 5.0
    best_inliers = []
    best_homography = None

    for _ in range(num_iter):
        # 4개의 매칭 샘플 선택
        sample_idxs = np.random.choice(len(matches), 4, replace=False)
        H = get_homography(kpts1, kpts2, matches, sample_idxs)

        if H is None:
            continue

        inliers = []

        # 각 매치에 대해 인라이어 검사 수행
        for i in range(len(matches)):
            pt1 = np.append(kpts1[matches[i, 0]], 1)
            pt2 = kpts2[matches[i, 1]]
            proj_pt1 = np.dot(H, pt1)
            proj_pt1 /= proj_pt1[2]  # 정규화

            # 거리 계산
            if np.linalg.norm(proj_pt1[:2] - pt2[:2]) < threshold:
                inliers.append(matches[i])

        # 최대 인라이어 수 업데이트
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_homography = H

    # 최대 인라이어 매칭과 최적의 호모그래피 반환
    matches_in = np.array(best_inliers)
    
    return matches_in, best_homography


def get_aligned_image(image1, image2, H):
    '''계산된 Homography를 이용해서 두 영상을 정렬하여 이어붙인 영상을 생성하시오. 

    ######################################################################
    
    1. 우선 각 영상별로 네 꼭지점들의 정렬 이후의 좌표를 도출하여 영상의 전체 크기를 도출하고 그에 맞된 출력영상 ndarray 변수를 생성함
    2. 생성된 출력영상의 각 픽셀별로 H의 inverse mapping 등을 통해 대응되는 원본 영상의 대응 픽셀을 찾아 해당 픽셀의 컬러를 심어옴
    3. 컬러를 심어올 때 bilinear interpolation 등 기법 적용 시 가산점 부여 (+15% 까지)
    
    ######################################################################
    
    :params:
    :image1: first image of input image pair
    :image2: second image of input image pair
    :H: the homography computed from the matched points

    :returns:
    :aligned_image: the image generated by aligning and stitching the input image pairs
    '''
    # These are placeholders - replace with your matches and confidences!
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    H_inv = np.linalg.inv(H)

    corners1 = np.array([
        [0, 0, 1],
        [w1 - 1, 0, 1],
        [w1 - 1, h1 - 1, 1],
        [0, h1 - 1, 1]
    ]).T
    transformed_corners = np.dot(H, corners1)
    transformed_corners /= transformed_corners[2]

    x_min, y_min = transformed_corners[:2, :].min(axis=1)
    x_max, y_max = transformed_corners[:2, :].max(axis=1)

    x_min = min(x_min, 0)
    y_min = min(y_min, 0)
    x_max = max(x_max, w2)
    y_max = max(y_max, h2)

    out_height = int(np.ceil(y_max - y_min))
    out_width = int(np.ceil(x_max - x_min))

    aligned_image = np.zeros((out_height, out_width, 3), dtype=image1.dtype)

    translation_matrix = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    for y in range(out_height):
        for x in range(out_width):
            xy_homog = np.dot(np.linalg.inv(translation_matrix).dot(H_inv), [x, y, 1])
            xy_homog /= xy_homog[2]
            xi, yi = xy_homog[:2]

            if 0 <= xi < w1 and 0 <= yi < h1:
                aligned_image[y, x] += bilinear_interpolation(image1, xi, yi) * 0.5
            if 0 <= x + x_min < w2 and 0 <= y + y_min < h2:
                aligned_image[y, x] += image2[int(y + y_min), int(x + x_min)] * 0.5

    
    return aligned_image

def bilinear_interpolation(img, x, y):
    x_floor, y_floor = int(x), int(y)
    x_ceil, y_ceil = min(x_floor + 1, img.shape[1] - 1), min(y_floor + 1, img.shape[0] - 1)

    # 픽셀 값 가져오기
    bl = img[y_floor, x_floor]  # Bottom left
    br = img[y_floor, x_ceil]   # Bottom right
    tl = img[y_ceil, x_floor]   # Top left
    tr = img[y_ceil, x_ceil]    # Top right

    # 가중치 계산
    top = (x_ceil - x) * tl + (x - x_floor) * tr
    bottom = (x_ceil - x) * bl + (x - x_floor) * br
    pixel = (y_ceil - y) * bottom + (y - y_floor) * top

    return pixel