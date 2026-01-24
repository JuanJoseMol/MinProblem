import numpy as np
from matplotlib import pyplot as plt
"""
a = np.load('../data/data_div2k.npz')
print (a.files)
b = a['train_data']
print (b.shape)
c = a['test_data']
print (c.shape)
plt.imshow(c[2])
plt.title(f"testSample2-s{c[2].shape}.npz")
plt.show()

"""

from PIL import Image

name = "castle19.png"
name = "noiseRaro0649.png"
name = "mariposa0829.png"
name = "pinguino0344.png"
name = "Coral0026.png"

image = Image.open("../data/"+name).convert("RGB")  # Ensure RGB mode


a = np.array(image)
plt.imshow(a)
plt.title(f"{name}-s{a.shape}")
plt.show()
