import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, transform, img_as_int
from skimage.measure import regionprops
from skimage.color import rgb2gray


def load_folder_imgs(folder_name, img_size):
    """
    Load all images in folder and resize
    :param folder_name: string of path to database root folder
    :param img_size: tuple of size of image (to resize)

    :return data: an ndarray of images of size m x h x w, 
                  where m = number of images and h, w are width and height of images
    """
    ### your code here ###
    # get all image paths in folder and store in list
    path_list = []

    n = len(path_list)

    # initialize data
    data = np.zeros((n,img_size[0],img_size[1]))

    path_list = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    n = len(path_list)

    # 이미지 데이터를 저장할 배열을 초기화합니다.
    data = np.zeros((n, img_size[1], img_size[0]))  # (m, height, width, channels)

    # 각 이미지 파일에 대하여
    for i, path in enumerate(path_list):
        img = io.imread(path)
        img = rgb2gray(img)# 이미지를 읽습니다.
        img = transform.resize(img, (img_size[1], img_size[0]), anti_aliasing=True)  # 크기를 조정합니다.
        img = img / 255  # 픽셀 값의 범위를 [0, 1]로 조정합니다.
        data[i] = img  # 데이터 배열에 이미지를 추가합니다.

    return data

def get_integral_imgaes(imgs):
    """
    Compute integral image for all images in ndarray
    :param imgs: ndarray of images of size m x h x w, 
                 where m = number of images and h, w are width and height of images 

    :return iimgs: an ndarray of integral images of size m x h x w, 
                   where m = number of images and h, w are width and height of images
    """
    ### your code here ###
    iimgs = np.zeros_like(imgs)

    for n in range(imgs.shape[0]):
        img = imgs[n]
        h = imgs[n].shape[0]
        w = imgs[n].shape[1]
        
        ii = np.zeros_like(imgs[n])
        si = np.zeros_like(imgs[n])
        
        for j in range(h):
            for i in range(w):
                si[j, i] = si[j, i - 1] + img[j, i]
                ii[j, i] = ii[j - 1, i] + si[j, i]
                iimgs[n] = ii
                
    # return integral images
    return iimgs


def get_feature_pos_sz_2h(hlf_sz):
    """
    Compute all positions and sizes of type 2h haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_w, img_h = hlf_sz
    
    features = []
    
    for x1 in range(1, img_w, 2):
        for y1 in range(img_h):
            for x2 in range(x1, img_w):
                for y2 in range(y1, img_h):
                    if x2 - x1 > 4 and y2 - y1 > 4:
                        w = x2 - x1 + 1
                        h = y2 - y1 + 1
                        features.append([x1, y1, w, h])
    
    
    ps = np.array(features)
    return ps


def get_feature_pos_sz_2v(hlf_sz):
    """
    Compute all positions and sizes of type 2v haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    
    img_w, img_h = hlf_sz
    
    features = []
    
    for x1 in range(img_w):
        for y1 in range(1, img_h, 2):
            for x2 in range(x1, img_w):
                for y2 in range(y1, img_h):
                    if x2 - x1 > 4 and y2 - y1 > 4:
                        w = x2 - x1 + 1
                        h = y2 - y1 + 1
                        features.append([x1, y1, w, h])
    
    
    ps = np.array(features)
    return ps


def get_feature_pos_sz_3h(hlf_sz):
    """
    Compute all positions and sizes of type 3h haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_w, img_h = hlf_sz
    
    features = []
    
    for x1 in range(1, img_w, 3):
        for y1 in range(img_h):
            for x2 in range(x1, img_w):
                for y2 in range(y1, img_h):
                    if x2 - x1 > 4 and y2 - y1 > 4:
                        w = x2 - x1 + 1
                        h = y2 - y1 + 1
                        features.append([x1, y1, w, h])
    
    
    ps = np.array(features)
    return ps



def get_feature_pos_sz_3v(hlf_sz):
    """
    Compute all positions and sizes of type 3v haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_w, img_h = hlf_sz
    
    features = []
    
    for x1 in range(img_w):
        for y1 in range(1, img_h, 3):
            for x2 in range(x1, img_w):
                for y2 in range(y1, img_h):
                    if x2 - x1 > 4 and y2 - y1 > 4:
                        w = x2 - x1 + 1
                        h = y2 - y1 + 1
                        features.append([x1, y1, w, h])
    
    
    ps = np.array(features)
    return ps


def get_feature_pos_sz_4(hlf_sz):
    """
    Compute all positions and sizes of type 4 haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_w, img_h = hlf_sz
    
    features = []
    
    for x1 in range(1, img_w, 2):
        for y1 in range(1, img_h, 2):
            for x2 in range(x1, img_w):
                for y2 in range(y1, img_h):
                    if x2 - x1 > 4 and y2 - y1 > 4:
                        w = x2 - x1 + 1
                        h = y2 - y1 + 1
                        features.append([x1, y1, w, h])
    
    
    ps = np.array(features)
    return ps



def compute_features_2h(ps, iimg):
    """
    Compute all positions and sizes of type 2h haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    
    feats = np.zeros((iimg.shape[0], ps.shape[0]))
    
    for i in range(iimg.shape[0]):
        for j in range(ps.shape[0]):
            x, y, w, h = ps[j]
            A = iimg[i, y, x] if y > 0 and x > 0 else 0
            B = iimg[i, y, x + w//2] if x + w//2 < iimg.shape[2] and y > 0 else 0
            C = iimg[i, y, x + w] if x + w < iimg.shape[2] and y > 0 else 0
            D = iimg[i, y + h, x] if y + h < iimg.shape[1] and x > 0 else 0
            E = iimg[i, y + h, x + w//2] if  y + h < iimg.shape[1] and x + w//2 < iimg.shape[2] else 0
            F = iimg[i, y + h, x + w] if y + h < iimg.shape[1] and x + w < iimg.shape[2] else 0
            
            black = E - D - B + A
            white = F - E - C + B
            
            value = white - black

            feats[i, j] = value
            
    return feats 


def compute_features_2v(ps, iimg):
    """
    Compute all positions and sizes of type 2v haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    feats = np.zeros((iimg.shape[0], ps.shape[0]))
    
    for i in range(iimg.shape[0]):
        for j in range(ps.shape[0]):
            x, y, w, h = ps[j]
            A = iimg[i, y, x] if y > 0 and x > 0 else 0
            B = iimg[i, y, x + w] if x + w < iimg.shape[2] and y > 0 else 0
            C = iimg[i, y + h//2, x] if y + h//2 < iimg.shape[1] and x > 0 else 0
            D = iimg[i, y + h//2, x + w] if y + h//2 < iimg.shape[1] and x + w < iimg.shape[2] else 0
            E = iimg[i, y + h, x] if  y + h < iimg.shape[1] and x > 0 else 0
            F = iimg[i, y + h, x + w] if y + h < iimg.shape[1] and x + w < iimg.shape[2] else 0
            
            black = D - C - B + A
            white = F - E - D + C
            
            value = white - black

            feats[i, j] = value
            
    return feats  


def compute_features_3h(ps, iimg):
    """
    Compute all positions and sizes of type 3h haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    feats = np.zeros((iimg.shape[0], ps.shape[0]))
    
    for i in range(iimg.shape[0]):
        for j in range(ps.shape[0]):
            x, y, w, h = ps[j]
            A = iimg[i, y, x] if y > 0 and x > 0 else 0
            B = iimg[i, y, x + w//3] if x + w//3 < iimg.shape[2] and y > 0 else 0
            C = iimg[i, y, x + 2 * (w//3)] if x + 2 * (w//3) < iimg.shape[2] and y > 0 else 0
            D = iimg[i, y, x + w] if x + w < iimg.shape[2] and y > 0 else 0
            E = iimg[i, y + h, x] if y + h < iimg.shape[1] and x + w//3 < iimg.shape[2] else 0
            F = iimg[i, y + h, x + w//3] if  y + h < iimg.shape[1] and x + w//2 < iimg.shape[2] else 0
            G = iimg[i, y + h, x + 2 * (w//3)] if y + h < iimg.shape[1] and x + 2 * (w//3) < iimg.shape[2] else 0
            H = iimg[i, y + h, x + w] if x + w < iimg.shape[2] and y + h < iimg.shape[1] else 0
            black = (F - E - B + A) + (H - G - D + C)
            white = G - F - C + B
            
            value = white - black

            feats[i, j] = value
            
    return feats


def compute_features_3v(ps, iimg):
    """
    Compute all positions and sizes of type 3v haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    feats = np.zeros((iimg.shape[0], ps.shape[0]))
    
    for i in range(iimg.shape[0]):
        for j in range(ps.shape[0]):
            x, y, w, h = ps[j]
            A = iimg[i, y, x] if y > 0 and x > 0 else 0
            B = iimg[i, y , x + w] if x + w < iimg.shape[2] and y > 0 else 0
            C = iimg[i, y + h//3, x] if x > 0 and y + h//3 < iimg.shape[1] else 0
            D = iimg[i, y + h//3, x + w] if x + w < iimg.shape[2] and y + h//3 < iimg.shape[1] else 0
            E = iimg[i, y + 2 * (h//3), x] if y + 2 * (h//3) < iimg.shape[1] and x > 0 else 0
            F = iimg[i, y + 2 * (h//3), x + w] if  y + 2 * (h//3) < iimg.shape[1] and x + w < iimg.shape[2] else 0
            G = iimg[i, y + h, x] if y + h < iimg.shape[1] and x > 0 else 0
            H = iimg[i, y + h, x + w] if x + w < iimg.shape[2] and y + h < iimg.shape[1] else 0
            black = (D - C - B + A) + (H - G - F + E)
            white = F - E - D + C
            
            value = white - black

            feats[i, j] = value
            
    return feats


def compute_features_4(ps, iimg):
    """
    Compute all positions and sizes of type 4 haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    feats = np.zeros((iimg.shape[0], ps.shape[0]))
    
    for i in range(iimg.shape[0]):
        for j in range(ps.shape[0]):
            x, y, w, h = ps[j]
            A = iimg[i, y, x] if y > 0 and x > 0 else 0
            B = iimg[i, y, x + w//2] if x + w < iimg.shape[2] and y > 0 else 0
            C = iimg[i, y, x + w] if x + w < iimg.shape[2] and y > 0 else 0
            D = iimg[i, y + h//2, x] if x > 0 and y + h//2 < iimg.shape[1] else 0
            E = iimg[i, y + h//2, x + w//2] if y + h//2 < iimg.shape[1] and x + w//2 < iimg.shape[2] else 0
            F = iimg[i, y + h//2, x + w] if y + h//2 < iimg.shape[1] and x + w < iimg.shape[2] else 0
            G = iimg[i, y + h, x] if y + h < iimg.shape[1] and x > 0 else 0
            H = iimg[i, y + h, x + w//2] if x + w//2 < iimg.shape[2] and y + h < iimg.shape[1] else 0
            I = iimg[i, y + h, x + w] if x + w < iimg.shape[2] and y + h < iimg.shape[1] else 0
            black = (E - D - B + A) + (I - H - F + E)
            white = (F - E - C + B) + (H - G - E + D)
            
            value = white - black

            feats[i, j] = value
            
    return feats


def get_weak_classifiers(feats, labels, weights):
    """
    Compute all positions and sizes of type 4 haar-like-features
    :param feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    :param labels: an ndarray of shape (n_img) with pos 1/neg 0 labels of all input imags

    :return thetas: an ndarray of weak classifier threshold values of shape (n_feat)
    :return signs: an ndarray of weak classifier sign values (either +1 or -1) of shape (n_feat) 
    :return errors: an ndarray of weak classifier total error values over all images of shape (n_feat)  
    """
    ### your code here ###
    num_feat = feats.shape[1]
    thetas = np.zeros([num_feat])
    signs = np.zeros([num_feat])
    errors = np.zeros([num_feat])
    
    for j in range(num_feat):
        feature_values = feats[:, j]
        
        # 가능한 모든 임계값을 찾기 위해 특징 값을 정렬
        sorted_indices = np.argsort(feature_values)
        sorted_features = feature_values[sorted_indices]
        sorted_labels = labels[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # 누적 합계를 통해 오류 계산
        pos_cumsum = np.cumsum(sorted_weights * (sorted_labels == 1))
        neg_cumsum = np.cumsum(sorted_weights * (sorted_labels == 0))

        # 오류 계산
        total_pos = pos_cumsum[-1]
        total_neg = neg_cumsum[-1]
        
        # 양수/음수 에러
        error_pos = neg_cumsum + (total_pos - pos_cumsum)
        error_neg = pos_cumsum + (total_neg - neg_cumsum)
        
        # 최소 오류와 최적의 임계값 찾기
        min_error_pos = np.min(error_pos)
        min_error_neg = np.min(error_neg)
        
        if min_error_pos < min_error_neg:
            errors[j] = min_error_pos
            signs[j] = 1
            thetas[j] = sorted_features[np.argmin(error_pos)]
        else:
            errors[j] = min_error_neg
            signs[j] = -1
            thetas[j] = sorted_features[np.argmin(error_neg)]

    return thetas, signs, errors



def visualize_haar_feature(hlf_sz, x,y,w,h,type):
    """
    Visualize haar-like feature
    :param hlf_sz: tuple (w, h) of size of haar-like feature box, 
    :param x, y, w, h, type: position x,y, size w,h, and type of haar-like feature

    :return hlf_img: image visualizing particular haar-like-feature
    """
    ### your code here ###
    # hlf_img = np.ones(hlf_sz)
    hlf_img = np.ones(hlf_sz, dtype=np.uint8) * 255  # Initialize image with white background
    
    if type == 1:  # Two-rectangle feature (horizontal)
        # Left half - black, right half - white
        hlf_img[y:y+h, x:x+w//2] = 0  # Left half black
        hlf_img[y:y+h, x+w//2:x+w] = 1  # Right half white
        
    elif type == 2:  # Two-rectangle feature (vertical)
        # Top half - black, bottom half - white
        hlf_img[y:y+h//2, x:x+w] = 0  # Top half black
        hlf_img[y+h//2:y+h, x:x+w] = 1  # Bottom half white
        
    elif type == 3:  # Three-rectangle feature (horizontal)
        # Left third - black, middle third - white, right third - black
        third_w = w // 3
        hlf_img[y:y+h, x:x+third_w] = 0  # Left third black
        hlf_img[y:y+h, x+third_w:x+2*third_w] = 1  # Middle third white
        hlf_img[y:y+h, x+2*third_w:x+w] = 0  # Right third black
        
    elif type == 4:  # Three-rectangle feature (vertical)
        # Top third - black, middle third - white, bottom third - black
        third_h = h // 3
        hlf_img[y:y+third_h, x:x+w] = 0  # Top third black
        hlf_img[y+third_h:y+2*third_h, x:x+w] = 1  # Middle third white
        hlf_img[y+2*third_h:y+h, x:x+w] = 0  # Bottom third black

    elif type == 5:  # Four-rectangle feature (checkerboard)
        # Top left - black, top right - white, bottom left - white, bottom right - black
        half_w = w // 2
        half_h = h // 2
        hlf_img[y:y+half_h, x:x+half_w] = 0  # Top left black
        hlf_img[y:y+half_h, x+half_w:x+w] = 1  # Top right white
        hlf_img[y+half_h:y+h, x:x+half_w] = 1  # Bottom left white
        hlf_img[y+half_h:y+h, x+half_w:x+w] = 0  # Bottom right black

    return hlf_img

def get_best_weak_classifier(errors, num_feat_per_type, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps):
    """
    Get Haar-like feature parameters of best weak classifier
    :param errors: error values for all weak classifiers
    :param num_feat_per_type: number of features per feature type
    :param feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps: ndarry of all positions and sizes of haar-like-features 

    :return x,y,w,h,type: position, size and type of Haar-like feature of best weak classifier
    """
    ### your code here ###
    idx = np.argmin(errors)

    if(0 <= idx <= feat2h_ps.shape[0]):
        type = 1
        x, y, w, h = feat2h_ps[idx]

    elif(feat2h_ps.shape[0] < idx <= feat2h_ps.shape[0] + feat2v_ps.shape[0]):
        type = 2
        x, y, w, h = feat2v_ps[idx - int(feat2h_ps.shape[0])]

    elif(feat2h_ps.shape[0] + feat2v_ps.shape[0] < idx <= feat2h_ps.shape[0] + feat2v_ps.shape[0] + feat3h_ps.shape[0]):
        type = 3
        x, y, w, h = feat3h_ps[idx - int(feat2h_ps.shape[0] + feat2v_ps.shape[0])]

    elif(feat2h_ps.shape[0] + feat2v_ps.shape[0] + feat3h_ps.shape[0] < idx <= feat2h_ps.shape[0] + feat2v_ps.shape[0] + feat3h_ps.shape[0] + feat3v_ps.shape[0]):
        type = 4
        x, y, w, h = feat3v_ps[idx - int(feat2h_ps.shape[0] + feat2v_ps.shape[0] + feat3h_ps.shape[0])]
    else:
        type = 5
        x, y, w, h = feat4_ps[idx - int(feat2h_ps.shape[0] + feat2v_ps.shape[0] + feat3h_ps.shape[0] + feat3v_ps.shape[0])]
    return x, y, w, h, type

def overlay_haar_feature(hlf_sz, x,y,w,h,type, image):
    """
    Visualize haar-like feature
    :param hlf_sz: tuple (w, h) of size of haar-like feature box, 
    :param x, y, w, h, type: position x,y, size w,h, and type of haar-like feature
    :param image: image to overlay haar-like feature

    :return hlf_img: image visualizing particular haar-like-feature
    """
    ### your code here ###
    hlf_img = image.copy()

    if type == 1:  # 두 사각형 특징 (수평)
        # 왼쪽 절반 - 검정, 오른쪽 절반 - 흰색
        hlf_img[y:y+h, x:x+w//2] = 0  # 왼쪽 절반 검정
        hlf_img[y:y+h, x+w//2:x+w] = 1  # 오른쪽 절반 흰색
        
    elif type == 2:  # 두 사각형 특징 (수직)
        # 위쪽 절반 - 검정, 아래쪽 절반 - 흰색
        hlf_img[y:y+h//2, x:x+w] = 0  # 위쪽 절반 검정
        hlf_img[y+h//2:y+h, x:x+w] = 1  # 아래쪽 절반 흰색
        
    elif type == 3:  # 세 사각형 특징 (수평)
        # 왼쪽 3분의 1 - 검정, 가운데 3분의 1 - 흰색, 오른쪽 3분의 1 - 검정
        third_w = w // 3
        hlf_img[y:y+h, x:x+third_w] = 0  # 왼쪽 3분의 1 검정
        hlf_img[y:y+h, x+third_w:x+2*third_w] = 1  # 가운데 3분의 1 흰색
        hlf_img[y:y+h, x+2*third_w:x+w] = 0  # 오른쪽 3분의 1 검정
        
    elif type == 4:  # 세 사각형 특징 (수직)
        # 위쪽 3분의 1 - 검정, 가운데 3분의 1 - 흰색, 아래쪽 3분의 1 - 검정
        third_h = h // 3
        hlf_img[y:y+third_h, x:x+w] = 0  # 위쪽 3분의 1 검정
        hlf_img[y+third_h:y+2*third_h, x:x+w] = 1  # 가운데 3분의 1 흰색
        hlf_img[y+2*third_h:y+h, x:x+w] = 0  # 아래쪽 3분의 1 검정

    elif type == 5:  # 네 사각형 특징 (체커보드)
        # 왼쪽 위 - 검정, 오른쪽 위 - 흰색, 왼쪽 아래 - 흰색, 오른쪽 아래 - 검정
        half_w = w // 2
        half_h = h // 2
        hlf_img[y:y+half_h, x:x+half_w] = 0  # 왼쪽 위 검정
        hlf_img[y:y+half_h, x+half_w:x+w] = 1  # 오른쪽 위 흰색
        hlf_img[y+half_h:y+h, x:x+half_w] = 1  # 왼쪽 아래 흰색
        hlf_img[y+half_h:y+h, x+half_w:x+w] = 0  # 오른쪽 아래 검정
    
    return hlf_img