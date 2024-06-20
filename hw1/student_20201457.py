# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech

# modified by Soochahn Lee @ Kookmin University
# For Introduction to Computer Vision course, Spring 2021


#20201457 반요한의 student.py 파일입니다.

import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

def my_imfilter(image, kernel):
    
    filtered_image = np.zeros_like(image)

    h = image.shape[0]
    w = image.shape[1]


    # 1D 가우시안 커널 적용을 위해 추가된 코드
    if len(kernel.shape) == 1:
        ker_size = kernel.shape[0] // 2

        #흑백
        if len(image.shape) == 2:  
            padded_image = np.pad(image, ((ker_size, ker_size), (0, 0)), mode='constant')

            for j in range(h):
                for i in range(w):
                    patch_sum = 0
                    for ker_idx in range(kernel.shape[0]):
                        img_idx = j + ker_idx - ker_size
                        if img_idx < 0 or img_idx >= h:
                            continue
                        patch_sum += padded_image[img_idx, i] * kernel[ker_idx]
                    filtered_image[j, i] = patch_sum

        #컬러
        elif len(image.shape) == 3:  
            padded_image = np.pad(image, ((ker_size, ker_size), (0, 0), (0, 0)), mode='constant')

            for c in range(image.shape[2]):
                for j in range(h):
                    for i in range(w):
                        patch_sum = 0
                        for ker_idx in range(kernel.shape[0]):
                            img_idx = j + ker_idx - ker_size
                            if img_idx < 0 or img_idx >= h:
                                continue
                            patch_sum += padded_image[img_idx, i, c] * kernel[ker_idx]
                        filtered_image[j, i, c] = patch_sum
                        
    else:  # 2D 필터링

        ker_h = kernel.shape[0]
        ker_w = kernel.shape[1]
        pad_h = ker_h // 2
        pad_w = ker_w // 2

        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        # 짝수일 경우 에러 메시지 뜨게함
        if ker_h % 2 == 0 or ker_w % 2 == 0:
            raise ValueError("kernel size must be odd")

        # 흑백
        if len(image.shape) == 2:
            for j in range(h):
                for i in range(w):
                    patch_sum = 0
                    for ker_j in range(kernel.shape[0]):
                        for ker_i in range(kernel.shape[1]):
                            img_j = j + ker_j
                            img_i = i + ker_i
                            patch_sum += padded_image[img_j, img_i] * kernel[ker_j, ker_i]
                    filtered_image[j, i] = patch_sum

        # 컬러
        elif len(image.shape) == 3:
            for c in range(image.shape[2]):
                for j in range(h):
                    for i in range(w):
                        patch_sum = 0
                        for ker_j in range(kernel.shape[0]):
                            for ker_i in range(kernel.shape[1]):
                                img_j = j + ker_j
                                img_i = i + ker_i
                                patch_sum += padded_image[img_j, img_i, c] * kernel[ker_j, ker_i]
                        filtered_image[j, i, c] = patch_sum

    #print('my_imfilter function in student.py needs to be implemented')

    return filtered_image




def my_imfilter_vect(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. 
    Use NumPy vectorization operations to avoid using for loops 
    as much as possible to achieve an optimal runtime.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    
    filtered_image = np.zeros_like(image)
        
    h = image.shape[0]
    w = image.shape[1]

    # 1D 가우시안 커널 적용을 위해 추가된 코드 << 이 부분을 제외하면 혹시나 시간 비교에서 문제가 생길까봐 똑같이 구현해줬습니다. 
    if len(kernel.shape) == 1:
            
        ker_size = kernel.shape[0] // 2

        if len(image.shape) == 2:
            padded_image = np.pad(image, ((ker_size, ker_size), (0, 0)), mode='constant')
        
            for j in range(h):
                for i in range(w):
                    patch = padded_image[j:j+kernel.shape[0], i]
                    filtered_image[j, i] = np.sum(patch * kernel)

        elif len(image.shape) == 3:
            padded_image = np.pad(image, ((ker_size, ker_size), (0, 0), (0, 0)), mode='constant')
        
            for c in range(image.shape[2]):
                for j in range(h):
                    for i in range(w):
                        patch = padded_image[j:j+kernel.shape[0], i, c]
                        filtered_image[j, i, c] = np.sum(patch * kernel)
                        
    
    else: # 2D 커널에 대한 컨벌루션
            
            ker_h = kernel.shape[0]
            ker_w = kernel.shape[1]
        
            # 짝수일 경우 에러 메시지 뜨게함
            if ker_h % 2 == 0 or ker_w % 2 == 0:
                raise ValueError("kernel size must be odd")    
        
            pad_h = ker_h // 2
            pad_w = ker_w // 2
        
            #흑백
            if len(image.shape) == 2:
                padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
                
                for j in range(h):
                    for i in range(w):
                        patch = padded_image[j:j+ker_h, i:i+ker_w]
                        filtered_image[j, i] = np.sum(patch * kernel)
        
            #컬러
            elif len(image.shape) == 3:
                padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        
                
                for c in range(image.shape[2]):
                    for j in range(h):
                        for i in range(w):
                            patch = padded_image[j:j+ker_h, i:i+ker_w, c]
                            filtered_image[j, i, c] = np.sum(patch * kernel)

    
    #print('my_imfilter function in student.py needs to be implemented')
    
    return filtered_image


#필터를 함수로 구현했습니다.

def my_mean_filter(n, m):

    mean_filter = np.ones((n, m)) / (n * m)

    return mean_filter

def my_gauss_2D(s, k):
    
    ker_size = k // 2
    
    y, x = np.mgrid[-ker_size:ker_size+1, -ker_size:ker_size+1]
    
    gaussian_filter = 1/(2*np.pi * (s**2)) * np.exp(-1 * (x**2 + y**2) / (2*(s**2)))
    
    total = gaussian_filter.sum()
    
    gaussian_filter /= total
    
    return gaussian_filter

def my_gauss_1D(s, k):
    
    ker_size = k // 2
    
    x = np.arange(-ker_size, ker_size + 1)
    
    gaussian_filter = 1 / (np.sqrt(2 * np.pi) * s) * np.exp(-1 * (x ** 2) / (2 * s ** 2))
    
    return gaussian_filter
    