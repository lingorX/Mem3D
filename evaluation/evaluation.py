import numpy as np

def dice(annotation, segmentation, void_pixels=None):

    assert(annotation.shape == segmentation.shape)

    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    return np.sum(annotation & segmentation) / \
           (np.sum(annotation, dtype=np.float32) + np.sum(segmentation, dtype=np.float32))*2


def jaccard(annotation, segmentation, void_pixels=None):

    assert(annotation.shape == segmentation.shape)


    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    return np.sum(annotation & segmentation) / \
           np.sum(annotation | segmentation, dtype=np.float32)