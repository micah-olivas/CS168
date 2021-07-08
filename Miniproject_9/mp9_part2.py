from PIL import Image as im
import numpy as np
from matplotlib import pyplot as plt
import cvxpy as cvx

## part a
img = np.array(im.open("corrupted.png"), dtype=int)[:,:,0]
Known = (img > 0).astype(int)
plt.figure()
plt.imshow(Known)

## part b
avg_pix_img = np.copy(img)
for i in range(Known.shape[0]):
    for j in range(1,Known.shape[1]):
        if Known[i, j] == 0:
            num_valid_neighbors = 0
            pixel_sum = 0
            if i - 1 >= 0 and Known[i - 1,j] == 1:
                num_valid_neighbors += 1
                pixel_sum += avg_pix_img[i - 1, j]
            if i + 1 < Known.shape[0] and Known[i + 1,j] == 1:
                num_valid_neighbors += 1
                pixel_sum += avg_pix_img[i + 1, j]
            if j - 1 >= 0 and Known[i, j - 1] == 1:
                num_valid_neighbors += 1
                pixel_sum += avg_pix_img[i, j - 1]
            if j + 1 < Known.shape[1] and Known[i, j + 1] == 1:
                num_valid_neighbors += 1
                pixel_sum += avg_pix_img[i, j + 1]
            
            if num_valid_neighbors == 0: continue
            avg_pix_img[i,j] = pixel_sum / num_valid_neighbors
            
plt.figure()
plt.imshow(avg_pix_img)
plt.gray()
plt.savefig('mp9.part2.b.png')

## part c
U = cvx.Variable(img.shape)
obj = cvx.Minimize(tv(U))
constraints = [multiply(Known, U) == multiply(Known, img)]
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)
# recovered image is now in U.value

plt.figure()
plt.imshow(U.value)
plt.gray()
plt.savefig('mp9.part2.c.png')
