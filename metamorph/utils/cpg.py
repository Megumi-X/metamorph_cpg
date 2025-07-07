import mujoco_py
import numpy as np
from typing import Dict, Set

class CPGNetwork:
    def __init__(self, num_oscillators: int, W: np.ndarray, a: np.ndarray, x0: np.ndarray, dt: float):
        self.num_oscillators = num_oscillators
        self.W = W
        self.dt = dt

        self.coupling_list = self.create_coupling_list(W)

        self.theta = np.zeros((num_oscillators))
        self.r = np.zeros((num_oscillators))
        self.R = np.zeros((num_oscillators))

        self.phi = np.zeros((num_oscillators, num_oscillators))
        self.nu = np.zeros((num_oscillators))
        self.x0 = x0
        self.a = a
    
    def create_coupling_list(self, W):
        rows, cols = np.where(np.triu(W) > 0)
        coupling_list = []
        for i, j in zip(rows, cols):
            coupling_list.append([i, j])
        return np.array(coupling_list, dtype=np.intp)

    def set_parameters(self, params: np.ndarray):
        N = self.num_oscillators
        
        self.R = params[0: N]
        self.nu = params[N: 2 * N]
        phases = params[2 * N: 3 * N]

        self.phi.fill(0.0)
        rows = self.coupling_list[:, 0]
        cols = self.coupling_list[:, 1]
        phase_diffs = phases[rows] - phases[cols]
        self.phi[rows, cols] = phase_diffs
        self.phi[cols, rows] = -phase_diffs

    def step(self):
        def compute_theta_dot(t, y):
            phase_differences = y[np.newaxis, :] - y[:, np.newaxis]
            sin_argument = phase_differences - self.phi
            interaction_matrix = self.W * np.sin(sin_argument) * self.r
            theta_dot_interactions = np.sum(interaction_matrix, axis=1)
            theta_dot = theta_dot_interactions + 2 * np.pi * self.nu
            return theta_dot

        def compute_r_dot(t, r):
            r_dot = self.a * (self.R - r)
            return r_dot
        
        self.theta = self.integrate_rk4(compute_theta_dot, self.theta, 0, self.dt)
        self.r = self.integrate_rk4(compute_r_dot, self.r, 0, self.dt)

    def get_output(self) -> np.ndarray:
        return (self.x0 + self.r * np.sin(self.theta))

    def reset(self):
        self.theta = np.zeros((self.num_oscillators))
        self.r = np.zeros((self.num_oscillators))
        self.R = np.zeros((self.num_oscillators))
        self.phi = np.zeros((self.num_oscillators, self.num_oscillators))

    def integrate_rk4(self, f, y0, t0, dt):
        k1 = f(t0, y0)
        k2 = f(t0 + dt / 2, y0 + dt / 2 * k1)
        k3 = f(t0 + dt / 2, y0 + dt / 2 * k2)
        k4 = f(t0 + dt, y0 + dt * k3)
        return y0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def get_adjacency_matrix(model) -> np.ndarray:
    num_actuators = model.nu
    adjacency_matrix = np.zeros((num_actuators, num_actuators), dtype=np.int32)
    actuator_to_joint_map: Dict[int, int] = {}
    for i in range(num_actuators):
        if model.actuator_trntype[i] == 0:
            joint_id = model.actuator_trnid[i, 0]
            actuator_to_joint_map[i] = joint_id

    joint_to_bodies_map: Dict[int, Set[int]] = {}
    for i in range(model.njnt):
        child_body_id = model.jnt_bodyid[i]
        parent_body_id = model.body_parentid[child_body_id]
        
        if parent_body_id != -1:
            joint_to_bodies_map[i] = {parent_body_id, child_body_id}
        else:
            joint_to_bodies_map[i] = {0, child_body_id}

    actuator_ids = list(actuator_to_joint_map.keys())
    for i in range(len(actuator_ids)):
        for j in range(i + 1, len(actuator_ids)):
            actuator_i_id = actuator_ids[i]
            actuator_j_id = actuator_ids[j]

            joint_i_id = actuator_to_joint_map[actuator_i_id]
            joint_j_id = actuator_to_joint_map[actuator_j_id]

            bodies_i = joint_to_bodies_map.get(joint_i_id)
            bodies_j = joint_to_bodies_map.get(joint_j_id)

            if bodies_i and bodies_j:
                if not bodies_i.isdisjoint(bodies_j):
                    adjacency_matrix[actuator_i_id, actuator_j_id] = 1
                    adjacency_matrix[actuator_j_id, actuator_i_id] = 1

    return adjacency_matrix

def get_initial_angles(model):
    initial_angles = []
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        qpos_address = model.jnt_qposadr[joint_id]
        initial_angle_rad = model.qpos0[qpos_address]
        initial_angles.append(initial_angle_rad)
    return np.array(initial_angles, dtype=np.float32)