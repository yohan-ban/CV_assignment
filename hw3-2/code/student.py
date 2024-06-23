import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, transform, img_as_int
from skimage.measure import regionprops
from skimage.color import rgb2gray
from tqdm import tqdm
from PIL import Image
import cv2

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

def get_integral_images(imgs):
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
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
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
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
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
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
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
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
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
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
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
    p = 0

    #특징값 정렬
    s_feat = np.sort(feats, axis = 1)
    
    # thr 후보 구하기
    thr = (s_feat[0:len(s_feat) - 1, :] + s_feat[1:, :]) / 2
    sign_list = []
    error_list = []

    for i in tqdm(range(thr.shape[0])):
        
        feature_val = feats
        thr_val = thr[i, :]
        
        pred_val = feature_val < thr_val
        
        comp = (pred_val == labels)
        
        #부호 정하기
        acc = np.sum(comp, axis = 0) / feats.shape[0]
        valid = acc < 0.5
                    
        sign_cand = np.where(valid, 1, -1)
        
        sign_list.append(sign_cand)
        stack_sign = np.vstack(sign_list)
        
        #에러 구하기
        error = weights * (1 - comp)
        error_cand = np.sum(error, axis = 0)
        error_list.append(error_cand)
        stack_error = np.vstack(error_list)
        
        best_idx = np.argmin(stack_error, axis = 0)
        
        thetas = thr[best_idx, np.arange(thr.shape[1])]
        signs = stack_sign[best_idx, np.arange(thr.shape[1])]
        errors = stack_error[best_idx, np.arange(thr.shape[1])]
    
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

def get_best_weak_classifier(errors, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps):
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
    

class haar_weak_classifier:
    """
    class for haar-like feature-based weak classifier
    :define class attributes(class variables) to include 
      -position, size, type of Haar-like feature
      -threshold and polarity
    :define methods(class functions) as needed
    """
    def __init__(self, x, y, w, h, feature_type): # <-- add class attributes
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.feature_type = feature_type
        self.theta = 0
        self.sign = 0
        self.alpha = 0


def get_strong_classifier(feats, labels, weights, num_feat_per_type, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps, num_weak_cls):
    """
    Train strong classifier
    :param feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    :param labels: an ndarray of shape (n_img) with pos 1/neg -1 labels of all input images
    :param weights: an ndarray of shape (n_img) with weight values of all input images 
    :param num_feat_per_type: number of features per feature type
    :param feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps: ndarray of all positions and sizes of haar-like-features 
    :param num_weak_cls: number of weak classifiers that comprise the strong classifier 

    :return hwc_list: list of haar-like feature based weak classifiers
    """
    hwc_list = []

    n_img = feats.shape[0]

    for j in tqdm(range(num_weak_cls)):

        weights /= np.sum(weights)

        thetas, signs, errors = get_weak_classifiers(feats, labels, weights)
        idx = np.argmin(errors)
        x, y, w, h, feat_type = get_best_weak_classifier(errors, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps)

        ej = errors[idx]
        beta_j = (ej / (1.0 - ej))
        alpha_j = -np.log(beta_j)
        
        hwc = haar_weak_classifier(x, y, w, h, feat_type)
        hwc.sign = signs[idx]
        hwc.theta = thetas[idx]
        hwc.alpha = alpha_j
    
        
        if hwc.sign == 1:
            e_ij = np.where(feats[:, idx] < hwc.theta, 1, 0)
        else:
            e_ij = np.where(feats[:, idx] > hwc.theta, 1, 0)


        weight_update = np.zeros_like(weights)
        
        for i in range(n_img):
            weight_update[i] = weights[i] * (beta_j ** e_ij[i])
            
        weights = weight_update
       
        hwc_list.append(hwc)    
    return hwc_list


def apply_strong_classifier_training_iimgs(iimgs, hwc_list):
    """
    apply strong classifier to training images
    :param iimgs: training set integral images 
    :param hwc_list: list of Haar-like wavelet coefficients
    :return cls_list: list of classification results (classification result = 1 if face, 0 if not)
    """
    import numpy as np  # np 모듈을 import해야 합니다.

    cls_list = np.zeros(len(iimgs))

    for i, iimg in enumerate(iimgs):
        score = 0
        for hwc in hwc_list:
            x = hwc.x
            y = hwc.y
            w = hwc.w
            h = hwc.h

            # Integral image dimensions
            img_h, img_w = iimg.shape

            # Check if the feature is within bounds
            if y + h >= img_h or x + w >= img_w:
                continue

            if hwc.feature_type == 1:
                if y + h >= img_h or x + w//2 >= img_w:
                    continue
                black = iimg[y+h, (x+w//2)] - iimg[y+h, x] - iimg[y, (x+w//2)] + iimg[y, x]
                withe = iimg[y+h, x+w] - iimg[y+h, (x+w//2)] - iimg[y, x+w] + iimg[y, (x+w//2)]
                val = withe - black

            elif hwc.feature_type == 2:
                if y + h//2 >= img_h or x + w >= img_w:
                    continue
                black = iimg[(y+h//2), x+w] - iimg[(y+h//2), x] - iimg[y, x+w] + iimg[y, x]
                withe = iimg[y+h, x+w] - iimg[y+h, x] - iimg[(y+h//2), x+w] + iimg[(y+h//2), x]
                val = withe - black

            elif hwc.feature_type == 3:
                if y + h >= img_h or x + w//3 >= img_w or 2*(x+w//3) >= img_w:
                    continue
                black1 = iimg[y+h, (x+w//3)] - iimg[y+h, x] - iimg[y, (x+w//3)] + iimg[y, x]
                withe = iimg[y+h, 2 * (x+w//3)] - iimg[y+h, (x+w//3)] - iimg[y, 2 * (x+w//3)] + iimg[y, (x+w//3)]
                black2 = iimg[y+h, x+w] - iimg[y+h, 2 * (x+w//3)] - iimg[y, x+w] + iimg[y, 2 * (x+w//3)]
                val = withe - (black1 + black2)

            elif hwc.feature_type == 4:
                if y + h >= img_h or y + h//3 >= img_h or 2*(y+h//3) >= img_h or x + w >= img_w:
                    continue
                black1 = iimg[(y+h//3), x+w] - iimg[(y+h//3), x] - iimg[y, x+w] + iimg[y, x]
                withe = iimg[2 * (y+h//3), x+w] - iimg[2 * (y+h//3), x] - iimg[(y+h//3), x+w] + iimg[(y+h//3), x]
                black2 = iimg[y+h, x+w] - iimg[y+h, x] - iimg[2 * (y+h//3), x+w] + iimg[2 * (y+h//3), x]
                val = withe - (black1 + black2)

            elif hwc.feature_type == 5:
                if y + h >= img_h or y + h//2 >= img_h or x + w >= img_w or x + w//2 >= img_w:
                    continue
                black1 = iimg[(y+h//2), (x+w//2)] - iimg[(y+h//2), x] - iimg[y, (x+w//2)] + iimg[y, x]
                withe1 = iimg[y+h, (x+w//2)] - iimg[y+h, x] - iimg[(y+h//2), (x+w//2)] + iimg[(y+h//2), x]
                black2 = iimg[y+h, x+w] - iimg[y+h, (x+w//2)] - iimg[(y+h//2), x+w] + iimg[(y+h//2), (x+w//2)]
                withe2 = iimg[(y+h//2), x+w] - iimg[(y+h//2), (x+w//2)] - iimg[y, x+w] + iimg[y, (x+w//2)]
                val = (withe1 + withe2) - (black1 + black2)

            if hwc.sign == 1:
                pre = 1 if val < hwc.theta else -1
            else:
                pre = 1 if val > hwc.theta else -1

            score += hwc.alpha * pre

        if score > 0:
            cls_list[i] = 1
        elif score < 0:
            cls_list[i] = -1
        else:
            cls_list[i] = 0

    return cls_list



def get_classification_correctnet(labels, cls_list):
    """
    check correctness of classification results
    :param labels: an ndarray of shape (n_img) with pos 1/neg 0 labels of all input imags 
    :param cls_list: an ndarray of shape (n_img) with class estimatations

    :return cls_list: list of True/False results for class estimation input
    """
    ### YOUR CODE HERE ###
    correctness = []
    
    labels[labels == 0] = -1
    
    labels = labels.flatten()
    
    correctness = (labels == cls_list).astype(int)
    
    correctness[correctness == 1] = 1
    correctness[correctness == 0] = 0

    return correctness


def get_incorrect_images(data, correctness_list):
    """
    get incorrect images
    :param data: input of all images
    :param correctness_list: list of True/False results for class estimation input

    :return incorrect_imgs: list of incorrect images
    """
    ### YOUR CODE HERE ###
    incorrect_imgs = []
    
    for i, correctness in enumerate(correctness_list):
        if not correctness:
            incorrect_imgs.append(data[i])
            
    incorrect_imgs = np.array(incorrect_imgs)
    
    return incorrect_imgs

def load_image(path = "../example.jpg"):
    """
    Load image
    :return img: image
    """
    ### YOUR CODE HERE ###
    img = []
    
    if os.path.isfile(path):
        img = Image.open(path).convert('L')
        
        img = np.array(img, 'unit8')
        
    return (img / 255.0)


def easy_integral_images(images):
    integral_images = []
    for image in images:
        integral_image = cv2.integral(image)
        integral_images.append(integral_image)
    return integral_images


def detect_face(img, hwc_list, min_scale = 1.0, max_scale = 4.0, num_scales = 9):
    """
    face detection by multi-scale sliding window classification using strong classifier
    :param img: input image
    :param hwc_list: strong classifier compring list of haar-like feature based weak classifiers

    :return bboxs: list of bounding boxes of detected faces
    """
    ### YOUR CODE HERE ###
    bboxes = []
    h, w = img.shape
    # *** multi-scale input image, similar to code below
    scale_idx = np.arange(num_scales)
    scales = scale_idx * (max_scale - min_scale) / num_scales + min_scale

    for scale in scales:
        scaled_img = cv2.resize(img, (int(w / scale), int(h / scale)))
        
        sh, sw = scaled_img.shape
        
        window_sz = 18
        step = 6
        
        for y in tqdm(range(0, sh - window_sz, step)):
            for x in range(0, sw - window_sz, step):
                window = scaled_img[y:y + window_sz, x:x + window_sz]
                
                if window.shape[0] != window_sz or window.shape[1] != window_sz:
                    continue
                
                iimg = easy_integral_images([window])
                cls = apply_strong_classifier_training_iimgs(iimg, hwc_list)
                
                if cls[0] == 1:
                    x1 = int(x * scale)
                    y1 = int(y * scale)
                    bb = [int(x1*scale), int(y1*scale), x1 + int(window_sz * scale), y1 + int(window_sz * scale)]
                    bboxes.append(bb)
    
    return bboxes


def visualize_bboxes(img, bboxes):
    """
    Visualize bounding boxes
    :param img: input image to overlay bounding boxes
    :param bboxes: bounding boxes

    :return bbox_img: image with overlayed bounding boxes
    """
    ### YOUR CODE HERE ###
    bbox_img = img
    
    for bb in bboxes:
        x1 = bb[0]
        y1 = bb[1]
        x2 = bb[2]
        y2 = bb[3]
        
        cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (1, 1, 1), 1)
        
    return bbox_img 