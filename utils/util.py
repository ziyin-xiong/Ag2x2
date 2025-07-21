import numpy as np
import math
import torch

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def world2screen(points_w, view_m, projection_m, camera_w):
    B, S, _ = points_w.shape  # points_w [B, S, 3]
    points_w_homo = torch.cat([points_w.to("cuda:0"), torch.ones(B, S, 1).to("cuda:0")], dim=-1)  # [B, S, 4]
    points_c = torch.matmul(points_w_homo, view_m)
    points_ndc = torch.matmul(points_c, projection_m)
    points_ndc = points_ndc[..., :3] / points_ndc[..., 3].unsqueeze(-1)
    points_s = torch.empty(B, S, 2).to("cuda:0")
    points_s[..., 0] = (points_ndc[..., 0] + 1.0) * 0.5 * camera_w
    points_s[..., 1] = (1 - points_ndc[..., 1]) * 0.5 * camera_w
    return points_s

def cam2world(pos, rot, camera_m):
    gm = torch.eye(4)
    gm[:3, :3] = rot
    gm[:3, 3] = pos
    grasp_pose_world = torch.matmul(camera_m, gm)
    return grasp_pose_world

def compute_camera_transform(camera, target, up_vector=torch.tensor([0.0, 0.0, 1.0])):
    forward = target - camera
    forward = forward / torch.norm(forward)
    right = torch.cross(up_vector, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)
    up = up / torch.norm(up)
    rot_m = torch.stack([right, up, forward], dim=1)
    trans_m = torch.eye(4)
    trans_m[:3, :3] = rot_m
    trans_m[:3, 3] = camera
    return trans_m