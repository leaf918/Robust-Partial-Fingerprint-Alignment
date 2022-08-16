from skimage import transform as trans
import numpy as np

src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32)
dst = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32)
tform_simi = trans.SimilarityTransform()
res = tform_simi.estimate(dst, src)
M = tform_simi.params
print("simi trans report")
print(res)
print(M)
print('=' * 12)
res = tform_simi.estimate(dst[:2, :], src[:2, :])
M = tform_simi.params
print("simi trans report 2")
print(res)
print(M)
print('=' * 12)
tform_eucl = trans.EuclideanTransform()
res = tform_eucl.estimate(dst[:2, :], src[:2, :])
print("euclidean trans report")
print(res)
print(M)
print('=' * 12)
