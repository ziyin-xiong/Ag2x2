import os
import copy
import pickle
import imageio
import numpy as np
from tqdm import tqdm
from isaacgym import gymapi
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import json
import torch
import cv2

class APPROACH:
    def __init__(self, vec_env, cfg, save_goal=False, save_video=False):
        self.vec_env = vec_env
        self.env_name = cfg['name'].rsplit('@', 1)[0]
        self.traj_path = cfg['traj_path']
        self.traj_basedir = os.path.dirname(self.traj_path)
        self.traj_name = os.path.basename(self.traj_path).split('.')[0][7:]
        self.dummy_traj = pickle.load(open(self.traj_path, 'rb'))['trajectory']
        self.save_goal = save_goal
        self.save_video = save_video
        self.FSM_STATE = 'FREE'  ## StateDict: {'FREE', 'APPROACH', 'ATTACH'}

    def transfer(self, env_id=0):
        if self.env_name in ['ag2x2@close_door_outward', 'ag2x2@close_door_inward', 'ag2x2@open_pen_cap',
                             'ag2x2@lift_pot', 'ag2x2@swing_cup', 'ag2x2@close_scissors', 'ag2x2@push_box',
                             'ag2x2@put_cube_into_drawer', 'ag2x2@put_cube_into_wave', 'ag2x2@lift_tray',
                             'ag2x2@press_buttons', 'ag2x2@sweep_dirt', 'ag2x2@straighten_rope']:
            quat_default = np.array([0., 1., 0., 0.])
        else:
            raise NotImplementedError
        self.FSM_STATE = 'FREE'
        related_tm = None
        self.execute_traj = []
        for i_step, act in enumerate(self.dummy_traj):
            if act['attached_info_indices'] == -1: 
                pos = act['panda_hand'][env_id, :3] - Rotation.from_quat(quat_default).as_matrix() @ np.array([0, 0, 0.1])
                attractor_pose = gymapi.Transform()
                attractor_pose.p = gymapi.Vec3(*pos)
                attractor_pose.r = gymapi.Quat(0.707, 0, 0.707, 0)
                gripper_effort = 100
            else:
                attached_info = act['attach_info'][act['attached_info_indices'][env_id]]
                actor_trans = act['attach_info'][act['attached_info_indices'][env_id]]['object_trans']
                actor_rot = act['attach_info'][act['attached_info_indices'][env_id]]['object_rot']
                attach_trans = attached_info['attach_info']['translation']
                attach_rot = attached_info['attach_info']['rotation_matrix']

                attach_trans = actor_rot @ attach_trans + actor_trans
                attach_rot = actor_rot @ attach_rot

                attach_tm = np.eye(4)
                attach_tm[:3, :3] = attach_rot
                attach_tm[:3, 3] = attach_trans 
                attach_rot = Rotation.from_matrix(attach_rot).as_quat()
                if self.FSM_STATE == 'FREE':
                    self.FSM_STATE = 'APPROACH'
                    body_trans = act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], :3]
                    body_rot = act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], 3:7]
                    body_rot = Rotation.from_quat(body_rot).as_matrix()
                    body_tm = np.eye(4)
                    body_tm[:3, :3] = body_rot
                    body_tm[:3, 3] = body_trans
                    related_tm = attach_tm @ np.linalg.inv(body_tm)

                    #! plan the approach motion
                    attach_trans_back = attach_trans - Rotation.from_quat(attach_rot).as_matrix() @ np.array([0, 0, 0.1])
                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans_back)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    gripper_effort = 100
                    self.execute_traj.append([attractor_pose, gripper_effort, copy.deepcopy(self.FSM_STATE)])
                    self.FSM_STATE = 'ATTACH'
                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    gripper_effort = -100
                else:
                    body_trans = act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], :3]
                    body_rot = act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], 3:7]
                    body_rot = Rotation.from_quat(body_rot).as_matrix()
                    body_tm_t = np.eye(4)
                    body_tm_t[:3, :3] = body_rot
                    body_tm_t[:3, 3] = body_trans
                    body_tm_t = body_tm_t @ np.linalg.inv(body_tm)
                    attach_tm = body_tm_t @ related_tm @ body_tm
                    attach_trans = attach_tm[:3, 3]
                    attach_rot = Rotation.from_matrix(attach_tm[:3, :3]).as_quat()

                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    gripper_effort = -100
                
            self.execute_traj.append([attractor_pose, gripper_effort, copy.deepcopy(self.FSM_STATE)])
        
        # smooth the excute_traj on attractor_pose
        SMOOTH_STEPS = 5
        transition_flag = False
        execute_traj_smooth = []
        for i_step in range(len(self.execute_traj) - 1):
            lower_pos = np.array([self.execute_traj[i_step][0].p.x, self.execute_traj[i_step][0].p.y, self.execute_traj[i_step][0].p.z])
            upper_pos = np.array([self.execute_traj[i_step + 1][0].p.x, self.execute_traj[i_step + 1][0].p.y, self.execute_traj[i_step + 1][0].p.z])
            lower_quat = np.array([self.execute_traj[i_step][0].r.x, self.execute_traj[i_step][0].r.y, self.execute_traj[i_step][0].r.z, self.execute_traj[i_step][0].r.w])
            upper_quat = np.array([self.execute_traj[i_step + 1][0].r.x, self.execute_traj[i_step + 1][0].r.y, self.execute_traj[i_step + 1][0].r.z, self.execute_traj[i_step + 1][0].r.w])
            interp_rot = Slerp([0, 1], Rotation.from_quat([lower_quat, upper_quat]))
            smooth_steps = SMOOTH_STEPS
            if self.execute_traj[i_step + 1][2] == 'APPROACH':
                smooth_steps = 100
            elif self.execute_traj[i_step + 1][2] == 'ATTACH' and transition_flag == False:
                smooth_steps = 100
            for i_smooth in range(smooth_steps):
                i_smooth_pos = lower_pos + (upper_pos - lower_pos) * i_smooth / smooth_steps
                i_smooth_quat = interp_rot(i_smooth / smooth_steps).as_quat()
                attractor_pose = gymapi.Transform()
                attractor_pose.p = gymapi.Vec3(*i_smooth_pos)
                attractor_pose.r = gymapi.Quat(*i_smooth_quat)
                gripper_effort = self.execute_traj[i_step + 1][1]
                execute_traj_smooth.append([attractor_pose, gripper_effort])
            if self.execute_traj[i_step + 1][2] == 'ATTACH' and transition_flag == False:
                for _ in range(20):
                    gripper_effort = self.execute_traj[i_step + 2][1]
                    execute_traj_smooth.append([attractor_pose, gripper_effort])
                transition_flag = True
        self.execute_traj = execute_traj_smooth

        self.FSM_STATE = 'FREE'
        related_tm = None
        self.execute_traj1 = []
        for i_step, act in enumerate(self.dummy_traj1):
            if act['another_attached_info_indices'] == -1: 
                pos = act['another_panda_hand'][env_id, :3] - Rotation.from_quat(quat_default).as_matrix() @ np.array([0, 0, 0.1])
                attractor_pose = gymapi.Transform()
                attractor_pose.p = gymapi.Vec3(*pos)
                attractor_pose.r = gymapi.Quat(0.707, 0.707, 0, 0)
                gripper_effort = 100
            else:
                attached_info = act['attach_info'][act['another_attached_info_indices'][env_id]]
                actor_trans = act['attach_info'][act['another_attached_info_indices'][env_id]]['object_trans']
                actor_rot = act['attach_info'][act['another_attached_info_indices'][env_id]]['object_rot']
                attach_trans = attached_info['attach_info']['translation']
                attach_rot = attached_info['attach_info']['rotation_matrix']

                attach_trans = actor_rot @ attach_trans + actor_trans
                attach_rot = actor_rot @ attach_rot
                attach_tm = np.eye(4)
                attach_tm[:3, :3] = attach_rot
                attach_tm[:3, 3] = attach_trans 
                attach_rot = Rotation.from_matrix(attach_rot).as_quat()
                if self.FSM_STATE == 'FREE':
                    self.FSM_STATE = 'APPROACH'
                    body_trans = act['rigid_bodies'][env_id, act['another_attached_body_handle'][env_id], :3]
                    body_rot = act['rigid_bodies'][env_id, act['another_attached_body_handle'][env_id], 3:7]
                    body_rot = Rotation.from_quat(body_rot).as_matrix()
                    body_tm = np.eye(4)
                    body_tm[:3, :3] = body_rot
                    body_tm[:3, 3] = body_trans
                    related_tm = attach_tm @ np.linalg.inv(body_tm)

                    #! plan the approach motion
                    attach_trans_back = attach_trans - Rotation.from_quat(attach_rot).as_matrix() @ np.array([0, 0, 0.1])
                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans_back)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    gripper_effort = 100
                    self.execute_traj1.append([attractor_pose, gripper_effort, copy.deepcopy(self.FSM_STATE)])
                    self.FSM_STATE = 'ATTACH'
                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    gripper_effort = -100
                else:
                    body_trans = act['rigid_bodies'][env_id, act['another_attached_body_handle'][env_id], :3]
                    body_rot = act['rigid_bodies'][env_id, act['another_attached_body_handle'][env_id], 3:7]
                    body_rot = Rotation.from_quat(body_rot).as_matrix()
                    body_tm_t = np.eye(4)
                    body_tm_t[:3, :3] = body_rot
                    body_tm_t[:3, 3] = body_trans
                    body_tm_t = body_tm_t @ np.linalg.inv(body_tm)
                    attach_tm = body_tm_t @ related_tm @ body_tm
                    attach_trans = attach_tm[:3, 3]
                    attach_rot = Rotation.from_matrix(attach_tm[:3, :3]).as_quat()

                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    gripper_effort = -100.
                
            self.execute_traj1.append([attractor_pose, gripper_effort, copy.deepcopy(self.FSM_STATE)])
        
        # smooth the excute_traj on attractor_pose
        SMOOTH_STEPS = 5
        transition_flag = False
        execute_traj_smooth1 = []
        for i_step in range(len(self.execute_traj1) - 1):
            lower_pos = np.array([self.execute_traj1[i_step][0].p.x, self.execute_traj1[i_step][0].p.y, self.execute_traj1[i_step][0].p.z])
            upper_pos = np.array([self.execute_traj1[i_step + 1][0].p.x, self.execute_traj1[i_step + 1][0].p.y, self.execute_traj1[i_step + 1][0].p.z])
            lower_quat = np.array([self.execute_traj1[i_step][0].r.x, self.execute_traj1[i_step][0].r.y, self.execute_traj1[i_step][0].r.z, self.execute_traj1[i_step][0].r.w])
            upper_quat = np.array([self.execute_traj1[i_step + 1][0].r.x, self.execute_traj1[i_step + 1][0].r.y, self.execute_traj1[i_step + 1][0].r.z, self.execute_traj1[i_step + 1][0].r.w])
            interp_rot = Slerp([0, 1], Rotation.from_quat([lower_quat, upper_quat]))
            smooth_steps = SMOOTH_STEPS
            if self.execute_traj1[i_step + 1][2] == 'APPROACH':
                smooth_steps = 100
            elif self.execute_traj1[i_step + 1][2] == 'ATTACH' and transition_flag == False:
                smooth_steps = 100
            for i_smooth in range(smooth_steps):
                i_smooth_pos = lower_pos + (upper_pos - lower_pos) * i_smooth / smooth_steps
                i_smooth_quat = interp_rot(i_smooth / smooth_steps).as_quat()
                attractor_pose = gymapi.Transform()
                attractor_pose.p = gymapi.Vec3(*i_smooth_pos)
                attractor_pose.r = gymapi.Quat(*i_smooth_quat)
                gripper_effort = self.execute_traj1[i_step + 1][1]
                execute_traj_smooth1.append([attractor_pose, gripper_effort])
            if self.execute_traj1[i_step + 1][2] == 'ATTACH' and transition_flag == False:
                for _ in range(20):
                    gripper_effort = self.execute_traj1[i_step + 2][1]
                    execute_traj_smooth1.append([attractor_pose, gripper_effort])
                transition_flag = True
        self.execute_traj1 = execute_traj_smooth1
        
    def run(self):
        # transfer to get the excute_traj
        self.transfer()
        length = min(len(self.execute_traj), len(self.execute_traj1))
        pbar = tqdm(total=length, desc='processing')
        print(f'TPATH: {self.traj_path}')
        
        for i_step in range(length):
            infos = self.vec_env.task.step_plan(self.execute_traj[i_step], self.execute_traj1[i_step])
            pbar.update(1)
