import cv2
import torch
import kornia.geometry as kornia
import numpy as np

def get_perspective_transform(src, dst, isNorm=False):
    r"""Calculates a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
            the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
        - Output: :math:`(B, 3, 3)`
    """
    if not torch.is_tensor(src):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(src)))
    if not torch.is_tensor(dst):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == (4, 2):
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got {}"
                         .format(src.shape, dst.shape))

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    if isNorm:
        src, transform1 = kornia.epipolar.normalize_points(src)
        dst, transform2 = kornia.epipolar.normalize_points(dst)

    src_x, src_y  = torch.chunk(src, dim=-1, chunks=2)
    dst_x, dst_y  = torch.chunk(dst, dim=-1, chunks=2)
    ones, zeros = torch.ones_like(src_x), torch.zeros_like(src_x)

    ax = torch.cat([src_x, src_y, ones, zeros, zeros, zeros, -src_x*dst_x, -src_y*dst_x], dim=-1)
    ay = torch.cat([zeros, zeros, zeros, src_x, src_y, ones, -src_x*dst_y, -src_y*dst_y], dim=-1)\

    # A is Bx8x8
    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    # b is a Bx8x1
    batch_size = dst.shape[0]
    b = dst.reshape(batch_size, -1, 1)
    # solve the system Ax = b
    X = torch.linalg.solve(A, b)

    # create variable to return
    batch_size = src.shape[0]
    M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
    M[..., :8] = torch.squeeze(X, dim=-1)
    M = M.view(-1, 3, 3)  # Bx3x3

    if isNorm:
        M = transform2.inverse() @ (M @ transform1)
    
    return M

def four_point_to_homography(corners, deltas, crop=False):
    """
    Args:
        corners ():
        deltas ():
        crop (bool): If set to true, homography will aready contain cropping part.
    """

    # in order to apply transform and center crop,
    # subtract points by top-left corner (corners[N, 0])
    if 'torch' in str(type(corners)):
        # corners = corners.repeat(1,4,1)
        if crop:
            corners = corners - corners[:, 0].view(-1, 1, 2)
        corners_hat = corners + deltas

        return get_perspective_transform(corners, corners_hat, isNorm=False)

    elif 'numpy' in str(type(corners)):
        if crop:
            corners = corners - corners[:, 0].reshape(-1, 1, 2)
        corners_hat = corners + deltas
        return cv2.getPerspectiveTransform(np.float32(corners), np.float32(corners_hat))

    else:
        assert False, 'Wrong type?'
        
def image_shape_to_corners(patch):
    assert len(patch.shape) == 4, 'patch should be of size B, C, H, W'
    batch_size = patch.shape[0]
    image_width = patch.shape[-1]
    image_height = patch.shape[-2]
    if 'torch' in str(type(patch)):
        corners = torch.tensor([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]],
                               device=patch.device, dtype=patch.dtype, requires_grad=False)
        corners = corners.repeat(batch_size, 1, 1)
    elif 'numpy' in str(type(patch)):
        corners = np.float32([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]])
        corners = np.tile(np.expand_dims(corners, axis=0), (batch_size, 1, 1))
    else:
        assert False, 'Wrong type?'

    return corners


def warp_image(image, homography, target_h, target_w, inverse=False):

    if 'torch' in str(type(homography)):
        if inverse:
            homography = torch.inverse(homography)
        
        return kornia.transform.warp_perspective(image, homography, tuple((target_h, target_w)), align_corners=False)

    elif 'numpy' in str(type(homography)):
        if inverse:
            homography = np.linalg.inv(homography)

        return cv2.warpPerspective(image, homography, dsize=tuple((target_w, target_h)))

    else:
        assert False, 'Wrong type?'


def perspectiveTransform(points, homography):
    """
    Transform point with given homography.

    Args:
        points (np.array of size Nx2) - 2D points to be transformed
        homography (np.array of size 3x3) - homography matrix

    Returns:
        (np.array of size Nx2) - transformed 2D points
    """

    # Asserts
    assert len(points.shape) == 2 and points.shape[1] == 2, 'points arg should be of size Nx2, but has size: {}'. \
        format(points.shape)
    assert homography.shape == (3, 3), 'homography arg should be of size 3x3, but has size: {}'.format(homography.shape)

    if 'torch' in str(type(homography)) and 'torch' in str(type(points)):

        points = torch.nn.functional.pad(points, (0, 1), "constant", 1.)
        points_transformed = homography @ (points.permute(1, 0))
        points_transformed = points_transformed.permute(1, 0)
        return points_transformed[:, :2] / points_transformed[:, 2:].repeat(1, 2)

    elif 'numpy' in str(type(homography)) and 'numpy' in str(type(points)):

        return cv2.perspectiveTransform([points], homography).squeeze()

    else:
        assert False, 'Wrong or inconsistent types?'


def perspectiveTransformBatched(points, homography):
    """
    Transform point with given homography.

    Args:
        points (np.array of size BxNx2) - 2D points to be transformed
        homography (np.array of size Bx3x3) - homography matrix

    Returns:
        (np.array of size BxNx2) - transformed 2D points
    """

    # Asserts
    assert len(points.shape) == 3 and points.shape[2] == 2, 'points arg should be of size Nx2, but has size: {}'. \
        format(points.shape)
    assert homography.shape[1:] == (3, 3), 'homography arg should be of size 3x3, but has size: {}'\
        .format(homography.shape)

    if 'torch' in str(type(homography)) and 'torch' in str(type(points)):

        points = torch.nn.functional.pad(points, (0, 1), "constant", 1.)
        points_transformed = homography @ (points.permute(0, 2, 1))
        points_transformed = points_transformed.permute(0, 2, 1)
        return points_transformed[:, :, :2] / points_transformed[:, :, 2:].repeat(1, 1, 2)

    elif 'numpy' in str(type(homography)) and 'numpy' in str(type(points)):
        assert False, 'Not implemented - I was too lazy, sorry!'
    else:
        assert False, 'Wrong or inconsistent types?'


def calc_reprojection_error(source_points, target_points, homography):
    """
    Calculate reprojection error for a given homography.

    Args:
        source_points (np.array of size Nx2) - 2D points to be transformed
        target_points (np.array of size Nx2) - target 2D points
        homography (np.array of size 3x3) - homography matrix

    Returns:
        (float) - reprojection error
    """

    # Asserts
    assert len(source_points.shape) == 2 and source_points.shape[1] == 2, 'source_points arg should be of size Nx2, ' \
                                                                          'but has size: {}'.format(source_points.shape)
    assert len(target_points.shape) == 2 and target_points.shape[1] == 2, 'target_points arg should be of size Nx2, ' \
                                                                          'but has size: {}'.format(target_points.shape)
    assert homography.shape == (3, 3), 'homography arg should be of size 3x3, but has size: {}'.format(homography.shape)

    if 'torch' in str(type(homography)) and 'torch' in str(type(source_points)) and 'torch' in str(type(target_points)):

        transformed_points = perspectiveTransform(source_points, homography)
        reprojection_error = torch.sum((transformed_points - target_points) ** 2)
        return reprojection_error

    if 'numpy' in str(type(homography)) and 'numpy' in str(type(source_points)) and 'numpy' in str(type(target_points)):

        transformed_points = cv2.perspectiveTransform(np.expand_dims(source_points, axis=0), homography).squeeze()
        reprojection_error = np.sum((transformed_points - target_points) ** 2)
        return reprojection_error

    else:
        assert False, 'Wrong or inconsistent types?'
