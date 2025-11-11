import os
from typing import List, Any
import cv2
import numba

import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import QPoint, Q_ARG, Q_RETURN_ARG
from cv2 import pointPolygonTest
from functorch.dim import pointwise
from numpy import ndarray
from sympy.integrals.intpoly import point_sort
from torch import Tensor
from tqdm import tqdm

from utils.yaml_reader import YamlUtil


def getDpi(imgUrl):
    img = Image.open(imgUrl)
    dpi = img.info.get('dpi', (96, 96))
    return dpi


def mm_to_pixels(mm, imgUrl):
    inches = mm / 25.4
    dpi = getDpi(imgUrl)
    pixel_length = dpi[0] * inches
    return pixel_length


def create_gaussian_kernel(size, sigma, direction):
    kernel = cv2.getGaussianKernel(size, sigma, cv2.CV_32F)
    if direction == 0:
        return kernel
    elif direction == 1:
        return kernel.T
    return None


def create_fdog_kernel(size, sigma_c, sigma_s, rho, direction):
    kernel1 = create_gaussian_kernel(size, sigma_c, direction)
    kernel2 = create_gaussian_kernel(size, sigma_s, direction)
    dog_kernel = kernel1 - rho * kernel2
    return dog_kernel


def compute_all_candidates(images_rgb_nhwc: np.ndarray):
    out = []
    for i_frame in tqdm(range(images_rgb_nhwc.shape[0]), desc="Computing candidate points", unit=" image(s)"):
        image_np_gray = cv2.cvtColor(images_rgb_nhwc[i_frame], cv2.COLOR_RGB2GRAY)

        # gradient magnitude
        gx = cv2.Sobel(image_np_gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
        gy = cv2.Sobel(image_np_gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
        mag = cv2.magnitude(gx, gy)

        # normalization
        mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # neighbor max in 3*3 window
        k = np.ones((3, 3), np.uint8)
        k[1, 1] = 0
        nbr_max = cv2.dilate(mag_norm, k)

        # local maximum
        local_max = (mag_norm > nbr_max) & (mag_norm >= float(EdgeSnappingConfig.theta))

        # salient_img = local_max.astype(np.uint8) * 255
        # cv2.imwrite(f"debug/salient_points_on_frame_{i}.jpg", salient_img)

        # all candidate points on this frame
        ys, xs = np.nonzero(local_max)
        out.append(np.stack([xs, ys], axis=1))
    return out


class EdgeSnappingConfig:
    theta = None
    alpha = None
    beta = None
    beta_shift = None
    sigma_c = None
    sigma_s = None
    sigma_m = None
    rho = None
    X_MAX = None
    Y_MAX = None
    r_s = None
    candidate_num = None
    sampling_num = None
    average_weight_threshold = None

    fdog_kernel = None
    gaussian_kernel = None

    isConfigInit: bool = False

    @staticmethod
    def load(config_yaml_path='config/snapping_init.yaml'):
        if EdgeSnappingConfig.isConfigInit:
            return

        settings = YamlUtil.read(config_yaml_path)
        s = settings['snapping']

        EdgeSnappingConfig.theta = s['theta']
        EdgeSnappingConfig.alpha = s['alpha']
        EdgeSnappingConfig.beta = s['beta']
        EdgeSnappingConfig.beta_shift = s['beta_shift']
        EdgeSnappingConfig.sigma_c = s['sigma_c']
        EdgeSnappingConfig.sigma_s = s['sigma_s']
        EdgeSnappingConfig.sigma_m = s['sigma_m']
        EdgeSnappingConfig.rho = s['rho']
        EdgeSnappingConfig.X_MAX = s['x']
        EdgeSnappingConfig.Y_MAX = s['y']
        EdgeSnappingConfig.r_s = s['r_s']
        EdgeSnappingConfig.candidate_num = s['candidate_num']
        EdgeSnappingConfig.sampling_num = s['sampling_num']
        EdgeSnappingConfig.average_weight_threshold = s['average_weight_threshold']

        EdgeSnappingConfig.isConfigInit = True

        EdgeSnappingConfig.fdog_kernel = create_fdog_kernel(
            2 * EdgeSnappingConfig.X_MAX + 1,
            EdgeSnappingConfig.sigma_c,
            EdgeSnappingConfig.sigma_s,
            EdgeSnappingConfig.rho,
            1
        )
        EdgeSnappingConfig.gaussian_kernel = create_gaussian_kernel(
            2 * EdgeSnappingConfig.Y_MAX + 1,
            EdgeSnappingConfig.sigma_m,
            0
        )


def local_snapping(stroke: np.ndarray,
                   image_rgb_hwc: np.ndarray,
                   points_stroke_candidate: List[np.ndarray]):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    # convert np gray image [H, W] to tensor [B=1, C=1, H, W]
    H = image_rgb_hwc.shape[0]
    W = image_rgb_hwc.shape[1]
    image_tensor_gray_chw = rgb_np_to_gray_tensor(device, image_rgb_hwc)

    # convert jagged array to flatten array with index pointer page
    points_stroke_candidate = candidate_point_sets_defense(points_stroke_candidate, stroke)

    points_stroke_candidate_flatten, flatten_index_ptr = pack_jagged_list_to_array(points_stroke_candidate)
    n_candidate_points = flatten_index_ptr[-1]

    # order to xy and get point number of stroke
    stroke_len = stroke.shape[0]
    stroke = stroke.astype(np.float32)

    # ready for dp
    # energy -> accumulated energy for each candidate point
    # prev -> the best previous candidate point idx
    energy = np.full(n_candidate_points, np.inf, dtype=np.float32)
    prev = np.full(n_candidate_points, -1, dtype=np.int32)

    # accumulated energy for first candidate group is zero
    energy[flatten_index_ptr[0]: flatten_index_ptr[1]] = 0.0

    # TODO: correct the xy-error, tensor format error, and 255 0-1 error

    for i_group in range(stroke_len - 1):
        # get a candidate group of i, i+1 from flatten slice
        Q_i, Q_j = slice_candidate_group_by_index(points_stroke_candidate_flatten, flatten_index_ptr, i_group)

        # print(f"Qi_xy: {Qi_xy.shape[0]} * Qj_xy: {Qj_xy.shape[0]} = {Qi_xy.shape[0] * Qj_xy.shape[0]}")

        # weights between each two points in two groups
        p_i, p_j = stroke[i_group], stroke[i_group + 1]
        weights = compute_weights(H, W,
                                  p_i, p_j,
                                  Q_i, Q_j,
                                  image_tensor_gray_chw,
                                  device)  # shape: [K_i, K_j]

        # print(f"weights.shape: {weights.shape} ")

        dp_energy_iteration(i_group, flatten_index_ptr, energy, prev, weights)

    return pick_best_path(last_start=flatten_index_ptr[-2],
                          last_end=flatten_index_ptr[-1],
                          energy=energy,
                          prev=prev,
                          candidates_flatten=points_stroke_candidate_flatten,
                          stroke=stroke,
                          average_weight_standard=EdgeSnappingConfig.average_weight_threshold,
                          average_distance_standard=EdgeSnappingConfig.r_s / 4.0)


def candidate_point_sets_defense(points_candidate: list[np.ndarray], stroke: np.ndarray) -> list[np.ndarray]:
    fixed_candidates = []
    for t, cand in enumerate(points_candidate):
        if cand is None:
            p = stroke[t].astype(np.float32)
            fixed_candidates.append(np.array([[p[0], p[1]]], dtype=np.float32))
        else:
            fixed_candidates.append(cand)
    return fixed_candidates


def pick_best_path(last_start,
                   last_end,
                   energy: np.ndarray,
                   prev: np.ndarray,
                   candidates_flatten: np.ndarray,
                   stroke: np.ndarray,
                   average_weight_standard,
                   average_distance_standard):
    # TODO: add checks of two conditions based on the paper
    stroke_len = stroke.shape[0]

    while True:
        # last point index of underlying candidate stroke
        best_idx = np.argmin(energy[last_start:last_end]) + last_start
        curr_idx = best_idx

        avg_en = energy[best_idx] / (stroke_len - 1)

        # use backtrack to find all points of candidate stroke
        best_path_indices = []
        while best_idx != -1:
            best_path_indices.insert(0, best_idx)
            best_idx = prev[best_idx]

        # use slice to get the candidate stroke
        candidate_stroke_xy = candidates_flatten[best_path_indices]

        avg_dist = np.linalg.norm(candidate_stroke_xy - stroke, axis=1).mean()

        # print(f"avg en = {avg_en} (std: <{average_weight_standard}), avg dist = {avg_dist} (std: >{average_distance_standard})")

        if avg_dist < average_distance_standard:
            energy[curr_idx] = np.inf
            if np.isinf(energy[last_start:last_end]).all():
                return candidate_stroke_xy
            continue

        if avg_en > average_weight_standard:
            return candidate_stroke_xy

        return candidate_stroke_xy


@numba.njit
def dp_energy_iteration(i: int, flatten_index_ptr: np.ndarray,
                        energy: np.ndarray,
                        prev: np.ndarray,
                        weights: np.ndarray):
    start_i, end_i = flatten_index_ptr[i], flatten_index_ptr[i + 1]
    start_j, end_j = flatten_index_ptr[i + 1], flatten_index_ptr[i + 2]

    for idx_j in range(end_j - start_j):
        best_prev = -1
        best_energy = np.inf

        for idx_i in range(end_i - start_i):
            bi_energy = energy[start_i + idx_i] + weights[idx_i, idx_j]

            if bi_energy < best_energy:
                best_prev = start_i + idx_i
                best_energy = bi_energy

        energy[start_j + idx_j] = best_energy
        prev[start_j + idx_j] = best_prev


def compute_weights(H: int, W: int,
                    p_i: np.ndarray, p_j: np.ndarray,
                    Q_i: np.ndarray, Q_j: np.ndarray,
                    image_gray_chw: Tensor,
                    device):
    theta_flatten_gpu = compute_affine_theta_vectorized(Q_i,
                                                        Q_j,
                                                        H, W).to(device)  # shape: [K_{i} * K_{i+1}, 2, 3]

    grid_gpu = torch.nn.functional.affine_grid(theta_flatten_gpu,
                                               size=[theta_flatten_gpu.shape[0],
                                                     1,
                                                     2 * EdgeSnappingConfig.Y_MAX + 1,
                                                     2 * EdgeSnappingConfig.X_MAX + 1],
                                               align_corners=False).to(device)  # shape: [K_{i} * K_{i+1}, 2*Y+1, 2*X+1, 2]

    image_affined = (torch.nn.functional.grid_sample(image_gray_chw.expand(grid_gpu.shape[0], -1, -1, -1),
                                                     grid_gpu,
                                                     mode='bilinear',
                                                     padding_mode='zeros',
                                                     align_corners=False)
                     .squeeze(1)
                     .reshape(Q_i.shape[0],
                              Q_j.shape[0],
                              2 * EdgeSnappingConfig.Y_MAX + 1,
                              2 * EdgeSnappingConfig.X_MAX + 1)
                     .cpu().numpy())  # change to nparray such that do tensor dot afterward

    # print(f"theta_flatten_gpu: {theta_flatten_gpu.shape}")
    # print(f"grid_gpu.shape = {grid_gpu.shape}")
    # print(f"image_affine_and_trimmed: {image_affine_and_trimmed.shape}")
    # print(f"fdog.shape: {EdgeSnappingConfig.fdog_kernel.shape}")
    # print(f"gaus.shape: {EdgeSnappingConfig.gaussian_kernel.shape}")

    res_dot_on_x = np.tensordot(image_affined,
                                EdgeSnappingConfig.fdog_kernel.squeeze(),
                                axes=([-1], [0]))

    res_dot_on_x_y = np.tensordot(res_dot_on_x,
                                  EdgeSnappingConfig.gaussian_kernel.squeeze(),
                                  axes=([-1], [0])).squeeze()

    tilde_H_response = np.where(res_dot_on_x_y < 0, 1.0 + np.tanh(res_dot_on_x_y), 1.0)

    # print(f"temp.shape = {res_dot_on_x.shape}, res_dot_on_x_y.shape = {res_dot_on_x_y.shape}, tilde_H.shape = {tilde_H_response.shape}")

    # deform term (based on paper)
    r_s_square = float(EdgeSnappingConfig.r_s) ** 2

    p_diff = (p_j - p_i).astype(np.float32)
    q_diff = Q_j.astype(np.float32)[None, :, :] - Q_i.astype(np.float32)[:, None, :]
    diff = p_diff.reshape(1, 1, 2) - q_diff
    deform_term = np.sum(diff * diff, axis=-1) / r_s_square

    # shift term (not on paper)
    shift_i = np.sum((Q_i.astype(np.float32) - p_i[None, :]) ** 2, axis=-1)  # [K_i]
    shift_j = np.sum((Q_j.astype(np.float32) - p_j[None, :]) ** 2, axis=-1)  # [K_{i+1}]
    shift_term = (shift_i[:, None] + shift_j[None, :]) / (2.0 * r_s_square)  # [K_i, K_{i+1}]

    weights = deform_term + EdgeSnappingConfig.alpha * tilde_H_response

    # print(f"p_diff.shape = {p_diff.shape}")
    # print(f"q_diff.shape = {q_diff.shape}")
    # print(f"diff.shape = {diff.shape}")
    # print(f"square_norm.shape = {square_norm.shape}")
    # print(f"weights.shape = {weights.shape}")

    return weights


def slice_candidate_group_by_index(candidates_flatten_xy: np.ndarray, flatten_index_ptr: np.ndarray, i: int) \
        -> tuple[np.ndarray, np.ndarray]:
    Ui = slice(flatten_index_ptr[i], flatten_index_ptr[i + 1])
    Uj = slice(flatten_index_ptr[i + 1], flatten_index_ptr[i + 2])
    Qi_xy = candidates_flatten_xy[Ui]
    Qj_xy = candidates_flatten_xy[Uj]
    return Qi_xy, Qj_xy


def rgb_np_to_gray_tensor(device, image_rgb_hwc: np.ndarray) -> Tensor:
    image_gray_hw = cv2.cvtColor(image_rgb_hwc.astype(np.float32), cv2.COLOR_RGB2GRAY) / 255.0
    image_tensor_gray_gpu = (
        torch.from_numpy(image_gray_hw)
        .unsqueeze(0).unsqueeze(0)
        .to(device, non_blocking=True)
        .contiguous()
    )  # shape: [1, 1, H, W]
    return image_tensor_gray_gpu


# make jagged array to a integrated long array with index range pointer page
def pack_jagged_list_to_array(points_stroke_candidate: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    # total number of stroke points
    n_stroke_points = len(points_stroke_candidate)

    # index range pointer, index_ptr[i] means the start idx such that index_ptr[i+1] - index_ptr[i] means total number of current index
    index_ptr = np.zeros(n_stroke_points + 1, dtype=np.int32)
    for i in range(n_stroke_points):
        index_ptr[i + 1] = index_ptr[i] + (0 if points_stroke_candidate[i] is None else len(points_stroke_candidate[i]))
    n_total_candidates = index_ptr[-1]

    # flatten candidate points array
    points_flatten = np.empty((n_total_candidates, 2), dtype=np.float32)
    i_current = 0
    for i in range(n_stroke_points):
        points = points_stroke_candidate[i]
        if points is None or len(points) == 0:
            continue
        # make sure it is in float format
        points_float = points.astype(np.float32)
        # build flatten array
        points_flatten[i_current: i_current + len(points), :] = points_float[:, :]
        i_current += len(points)

    return points_flatten, index_ptr


def compute_affine_theta_vectorized(Q_i: np.ndarray,
                                    Q_j: np.ndarray,
                                    H: int, W: int,
                                    eps: np.float32 = 1e-6) -> Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Q_i = torch.from_numpy(Q_i.copy().astype(np.float32)).to(device)
    Q_j = torch.from_numpy(Q_j.copy().astype(np.float32)).to(device)

    # print(f"new Qi: {Qi.shape}")
    # print(f"new Qj: {Qj.shape}")

    len_i = Q_i.shape[0]
    len_j = Q_j.shape[0]

    m = 0.5 * (Q_i[:, None, :] + Q_j[None, :, :])  # [len_i, len_j, 2]
    d = (Q_j[None, :, :] - Q_i[:, None, :])  # [len_i, len_j, 2]
    L = torch.linalg.norm(d, dim=-1, keepdim=True)  # [len_i, len_j, 1]
    v = d / L.clamp(min=eps)  # [len_i, len_j, 2]

    u = torch.empty_like(v)  # [len_i, len_j, 2]
    u[..., 0] = -v[..., 1]
    u[..., 1] = v[..., 0]

    # print(f"m.shape: {m.shape}")
    # print(f"d.shape: {d.shape}")
    # print(f"L.shape: {L.shape}")
    # print(f"u.shape: {u.shape}")
    # print(f"v.shape: {v.shape}")

    X = float(EdgeSnappingConfig.X_MAX)
    Y = float(EdgeSnappingConfig.Y_MAX)

    # in sample grid, (x_norm. y_norm) means the center of a pixel
    sx = 2.0 / W
    sy = 2.0 / H
    bx = 1.0 / W - 1.0
    by = 1.0 / H - 1.0

    # image coordinates to NDC
    #                       [a00, a01, t0]
    # target affine matrix: [a10, a11, t1] in NDC (for pytorch grid sampling)

    # print(type(sy))
    # print(sy)

    a00 = sx * (X * u[..., 0])
    a01 = sx * (Y * v[..., 0])
    a10 = sy * (X * u[..., 1])
    a11 = sy * (Y * v[..., 1])
    t0 = sx * m[..., 0] + bx
    t1 = sy * m[..., 1] + by

    # print(f"a00.shape: {a00.shape}")
    # print(f"a01.shape: {a01.shape}")
    # print(f"a10.shape: {a10.shape}")
    # print(f"a11.shape: {a11.shape}")
    # print(f"t0.shape: {t0.shape}")
    # print(f"t1.shape: {t1.shape}")

    # build affine matrix - theta
    theta = torch.stack([
        torch.stack([a00, a01, t0], dim=-1),
        torch.stack([a10, a11, t1], dim=-1)
    ], dim=-2)  # [len_i, len_j, 2, 3]

    theta_flat = theta.reshape(len_i * len_j, 2, 3).contiguous()  # [len_i * len_j, 2, 3]

    # print(f"theta.shape: {theta.shape}")
    # print(f"theta_flat.shape: {theta_flat.shape}")

    return theta_flat
