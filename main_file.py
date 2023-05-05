import numpy as np
import cv2 

# It generates gaussian kernal which is to be convoluted with the grayscale image to reduce noise
def generate_gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g

# This checks if there are any pixels with more intensity around it which is in it's gradient direction 
# and supress the pixels having a maxima around it
def non_maximum_suppression(G, theta):
    # Finding dimensions of the image
    N, M = G.shape
    # Parsing through the all pixels
    for i_x in range(M):
        for i_y in range(N):
            
            grad_ang = theta[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
            
            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
            
            # top right (diagonal-1) direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
            
            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
            
            # top left (diagonal-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
            
            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
            
            # Non-maximum suppression step
            if M>neighb_1_x>= 0 and N>neighb_1_y>= 0:
                if  G[i_y, i_x]< G[neighb_1_y, neighb_1_x]:
                    G[i_y, i_x]= 0
                    continue

            if M>neighb_2_x>= 0 and N>neighb_2_y>= 0:
                if  G[i_y, i_x]< G[neighb_2_y, neighb_2_x]:
                    G[i_y, i_x]= 0

    return G

# Perform hysteresis thresholding to determine strong and weak edges.
def hysteresis_thresholding(img, t1, t2):
    
    weak = np.zeros_like(img)
    strong = np.zeros_like(img)
    strong_threshold = np.max(img) * t2
    weak_threshold = np.max(img) * t1

    strong[img >= strong_threshold] = 255
    weak[(img >= weak_threshold) & (img < strong_threshold)] = 128

    # perform connectivity analysis to determine strong edges
    M, N = img.shape
    edge_map = np.uint8(strong)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if weak[i,j] == 128:
                if (strong[i-1:i+2, j-1:j+2] == 255).any():
                    edge_map[i,j] = 255
                else:
                    edge_map[i,j] = 0

    return edge_map

# It applies convolution to the given image and the kernel matrix
def apply_convolution(img, kernel):
    M, N = img.shape
    m, n = kernel.shape
    padding = np.zeros((M+m-1, N+n-1))
    padding[m//2:M+m//2, n//2:N+n//2] = img

    output = np.zeros_like(img)
    for i in range(M):
        for j in range(N):
            output[i, j] =np.sum(padding[i:i+m, j:j+n] * kernel)

    return output

# Finds the gradient by using sobel operators
def sobel_op(img):
	dx_kernel = np.array([[-1, 0, 1],
                      	  [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

	dy_kernel = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float32)

	dx = np.zeros_like(img, dtype=np.float32)
	dy = np.zeros_like(img, dtype=np.float32)

	height, width = img.shape
        

	for i in range(1, height - 1):
		for j in range(1, width - 1):
			dx[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * dx_kernel)
			dy[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * dy_kernel)
	return dx,dy

# Final canny detector function
def Canny_detector(img, sigma, t1, t2):
    # Step 1: Convert given image to grayscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_gray
    
    # Step 2: Apply Gaussian filter to smooth the image
    kernel_size = int(2 * round(3 * sigma) + 1)
    gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma)
    img_smooth = apply_convolution(img, gaussian_kernel)
    smooth_img = img_smooth
    
    # Step 3: Compute gradient  Gnitude and direction using Sobel operators
    gx, gy = sobel_op(img)
    G_mag, G_dir = cv2.cartToPolar(gx, gy, angleInDegrees = True)

    # Step 4: Perform non-maximum suppression to thin the edges
    G_suppressed = non_maximum_suppression(G_mag, G_dir)

    # Step 5: Perform hysteresis thresholding to detect strong and weak edges
    edge_image= hysteresis_thresholding(G_suppressed, t1, t2)

	# returns grayscale image , smoothened image, edge image
    return img_gray, smooth_img, edge_image