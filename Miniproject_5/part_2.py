import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# read image
img = mpimg.imread('p5_image.gif')[:,:,0]

# compute full svd
u,s,vt = np.linalg.svd(img, full_matrices=False)

# report dimensions
print(u.shape, s.shape, vt.shape)

# build grid of reconstructions
ks = [1, 3, 10, 20, 50, 100, 150, 200, 400, 800, 1170]
max_k = s.size

fig, axs = plt.subplots(3,4, figsize=(16,12))
ax_arr = axs.ravel()

for idx, k in enumerate(ks):
    s_trunc = np.zeros(max_k)
    s_trunc[:k] = s[:k]
    im_k = u @ np.diag(s_trunc) @ vt
    ax_arr[idx].imshow(im_k, cmap='Greys_r')
    ax_arr[idx].set_title(k)

plt.tight_layout()
plt.savefig('part2.b.full.png')
plt.clf()

# save 150
k = 150
s_trunc = np.zeros(max_k)
s_trunc[:k] = s[:k]
im_k = u @ np.diag(s_trunc) @ vt

fig, ax = plt.subplots()
ax.imshow(im_k, cmap='Greys_r')

plt.tight_layout()
plt.savefig('part2.b.150.png')
plt.clf()


