#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# In[2]:


img = Image.open('shr.jpg')

img_arr = np.asarray(img)

# In[83]:


plt.imshow(img_arr)
plt.show()

# In[4]:


img_arr.T.shape

# In[5]:


print(img_arr)

# In[6]:


img_bin = np.vectorize(np.binary_repr)(img_arr, width=8)
print(img_bin)

# In[7]:


rav = img_bin.copy().ravel()
for el in range(rav.shape[0]):
    i = np.random.randint(0, 8)
    rav[el] = f'{rav[el][:i]}{int(rav[el][i]) ^ 1}{rav[el][i + 1:]}'

img_err = rav.reshape(img_bin.shape)
img_err

# In[79]:


pixels_err = np.array(list(map(lambda x: int(x, 2), list(img_err.ravel()))))

# In[82]:


plt.imshow(pixels_err.reshape(img_bin.shape))
plt.show()

# In[84]:


with open('img_bin.txt', 'w+') as outfile:
    outfile.write('# Array shape: {0}\n'.format(img_err.shape))
    for slice_2d in img_bin:
        outfile.write('# New slice\n')
        np.savetxt(outfile, slice_2d, fmt='%s')


# In[10]:


def print_array(arr):
    """
    prints a 2-D numpy array in a nicer format
    """
    for a in arr:
        for elem in a:
            print("{}".format(elem).rjust(3), end="")
        print(end="\n")


left = np.identity(8).astype(int)
right = np.random.randint(0, 2, size=(8, 8))

G_sys = np.hstack([left, right])
print('G_sys')
print('--------------------------------------------------')
print_array(G_sys)


# In[11]:


def xor(a: list):
    res = a[0]
    for el in a[1:]:
        res = np.bitwise_xor(res, el)
    return res


# In[12]:


d = np.array(range(2 ** 8))
left_mat = (((d[:, None] & (1 << np.arange(8))[::-1])) > 0).astype(int)

rmat = np.zeros((2 ** 8, 8))
for i, r in enumerate(left_mat[1:]):
    a = right[np.nonzero(r)[0], :]
    rmat[i + 1] = xor(a)

rmat = rmat.astype(int)

rc_mat = np.hstack([left_mat, rmat])

res = np.hstack([left_mat, rc_mat, rc_mat.sum(axis=1).reshape(-1, 1)])

print('         i = 8          |                      c = 16                   | d')
print('---------------------------------------------------------------------------')
print_array(res)

# In[23]:


# d = np.array(range(2**8))
# left_mat = (((d[:,None] & (1 << np.arange(8))[::-1])) > 0).astype(int)


rmat = np.zeros((img_bin.ravel().shape[0], 16))
print(rmat.shape)
for i, r in enumerate(img_bin.ravel()):
    a = G_sys[np.nonzero(np.array(list(map(int, list(r)))))[0], :]
    if len(a) == 0:
        rmat[i] = np.zeros(16)
        continue
    rmat[i] = xor(a)

rmat = rmat.astype(int)

# rc_mat = np.hstack([left_mat, rmat])

# res = np.hstack([left_mat, rc_mat, rc_mat.sum(axis=1).reshape(-1, 1)])


# print('         i = 8          |                      c = 16                   | d')
# print('---------------------------------------------------------------------------')
# print_array(rmat)


# In[24]:


rmat

# In[25]:


res = []
for el in rmat:
    res.append(np.array2string(el, separator='')[1:-1])

# In[26]:


bin_16 = np.array(res).reshape(img_bin.shape)
bin_16

# In[33]:


rav = bin_16.copy().ravel()
for el in range(rav.shape[0]):
    i = np.random.randint(0, 16)
    rav[el] = f'{rav[el][:i]}{int(rav[el][i]) ^ 1}{rav[el][i + 1:]}'

bin_16_err = rav.reshape(img_bin.shape)
bin_16_err

# In[85]:


with open('bin16.txt', 'w+') as outfile:
    outfile.write('# Array shape: {0}\n'.format(bin_16.shape))
    for slice_2d in bin_16:
        outfile.write('# New slice\n')
        np.savetxt(outfile, slice_2d, fmt='%s')

# In[86]:


with open('bin16_err.txt', 'w+') as outfile:
    outfile.write('# Array shape: {0}\n'.format(bin_16.shape))
    for slice_2d in bin_16_err:
        outfile.write('# New slice\n')
        np.savetxt(outfile, slice_2d, fmt='%s')

# In[28]:


G_sys

# In[29]:


H_t_sys = np.vstack((G_sys[:, 8:], G_sys[:, :8]))
H_t_sys

# In[37]:


print('H_t_sys')
print('-------------------------')
print_array(H_t_sys)

# In[38]:


vori = np.zeros((bin_16_err.ravel().shape[0], 8))
for i, r in enumerate(bin_16_err.ravel()):
    a = H_t_sys[np.nonzero(np.array(list(map(int, list(r)))))[0], :]
    if len(a) == 0:
        vori[i] = np.zeros(8)
        continue
    #     print(a)
    #     print(xor(a))
    vori[i] = xor(a)

vori = vori.astype(int)
print(vori.shape)
vori

# In[42]:


vori[-1]

# In[39]:


v = rc_mat[11]

v[2] = np.bitwise_not(v[2].astype(bool))
print('v')
print('-------------------------')
print(v)

a = H_t_sys[np.nonzero(v)[0], :]

e = a[0]
for l in a[1:]:
    e = np.bitwise_xor(e, l)

print('e')
print('-------------------------')
print(e)

kpc = np.hstack([H_t_sys[::-1], np.rot90(np.identity(16).astype(int))])

print('    S = 4               |           e = 8                              ')
print('-----------------------------------------------------------------------')
print_array(kpc)

# In[ ]:


for v in vori:
    np.bitwise_xor(H_t_sys[::-1])

# In[55]:


# np.bitwise_xor(H_t_sys[::-1])
eer = []
for v in vori:
    m = np.vectorize(np.bitwise_xor)(v, H_t_sys[::-1]).sum(axis=1).argmin()
    e = np.rot90(np.identity(16).astype(int))[m]
    eer.append(e)
eer = np.array(eer)

# In[56]:


print_array(eer)

# In[45]:


H_t_sys[::-1]

# In[59]:


gumarner = []
for i, e in enumerate(bin_16_err.ravel()):
    p = np.array(list(map(int, list(e))))
    k = np.bitwise_xor(p, eer[i])
    gumarner.append(k)
gumarner = np.array(gumarner)

# In[60]:


gumarner

# In[61]:


res = []
for el in gumarner:
    res.append(np.array2string(el, separator='')[1:-1])
chisht = np.array(res).reshape(bin_16.shape)

# In[65]:


# In[119]:


from tqdm import tqdm

# In[128]:


# np.bitwise_xor(H_t_sys[::-1])
ier = []
for v in tqdm(gumarner):
    m = np.vectorize(np.bitwise_xor)(v, rc_mat).sum(axis=1).argmin()
    e = left_mat[m]
    ier.append(e)
ier = np.array(ier)

# In[129]:


left_mat.shape

# In[130]:


ier

# In[131]:


res = []
for el in ier:
    res.append(int(''.join(list(map(str, el))), 2))
verjnakan = np.array(res).reshape(bin_16.shape)

# In[132]:


ier.shape

# In[133]:


verjnakan

# In[134]:


plt.imshow(verjnakan)

# In[ ]:




