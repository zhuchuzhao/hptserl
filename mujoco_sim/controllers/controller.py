from typing import Optional, Tuple, Union
import mujoco
import numpy as np
# from mujoco_sim.utils.mujoco_utils import MujocoModelNames


class Controller:
    def __init__(
        self,
        model,
        data,
        site_id,
        dof_ids: np.ndarray,
        config: dict,
    ):

        self.model = model
        self.data = data
        self.site_id = site_id
        self.integration_dt = config.get("integration_dt", model.opt.timestep)
        self.dof_ids = dof_ids
        # self.model_names = MujocoModelNames(self.model)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        # Set parameters from the config dictionary
        self.trans_damping_ratio = config.get("trans_damping_ratio", 0.996)
        self.rot_damping_ratio = config.get("rot_damping_ratio", 0.286)
        self.error_tolerance_pos = config.get("error_tolerance_pos", 0.001)
        self.error_tolerance_ori = config.get("error_tolerance_ori", 0.001)
        self.max_pos_error = config.get("max_pos_error", 0.01)
        self.max_ori_error = config.get("max_ori_error", 0.03)
        self.method = config.get("method", "dynamics")
        self.inertia_compensation = config.get("inertia_compensation", False)
        self.pos_gains = config.get("pos_gains", (100, 100, 100))
        self.ori_gains = config.get("ori_gains", tuple(gain * 1/8 for gain in self.pos_gains))
        self.pos_kd = config.get("pos_kd", None)
        self.ori_kd = config.get("ori_kd", None)
        self.max_angvel = config.get("max_angvel", 4)

        # Preallocate memory for commonly used variables
        self.quat = np.zeros(4)
        self.quat_des = np.zeros(4)
        self.quat_conj = np.zeros(4)
        self.quat_err = np.zeros(4)
        self.ori_err = np.zeros(3)
        self.x_err_norm = 0.0
        self.ori_err_norm = 0.0
        self.J_v = np.zeros((3, model.nv), dtype=np.float64)
        self.J_w = np.zeros((3, model.nv), dtype=np.float64)
        self.M = np.zeros((model.nv, model.nv), dtype=np.float64)
        self.error = np.zeros(6)

    def set_parameters(
        self,
        trans_damping_ratio: Optional[float] = None,
        rot_damping_ratio: Optional[float] = None,        
        error_tolerance_pos: Optional[float] = None,
        error_tolerance_ori: Optional[float] = None,
        max_pos_error: Optional[float] = None,
        max_ori_error: Optional[float] = None,
        max_angvel: Optional[float] = None,
        pos_gains: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        pos_kd: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        ori_kd: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        method: Optional[str] = None,
        inertia_compensation: Optional[bool] = None,
    ):
        if trans_damping_ratio is not None:
            self.trans_damping_ratio = trans_damping_ratio
        if rot_damping_ratio is not None:
            self.rot_damping_ratio = rot_damping_ratio
        if error_tolerance_pos is not None:
            self.error_tolerance_pos = error_tolerance_pos
        if error_tolerance_ori is not None:
            self.error_tolerance_ori = error_tolerance_ori
        if max_pos_error is not None:
            self.max_pos_error = max_pos_error
        if max_ori_error is not None:
            self.max_ori_error = max_ori_error
        if max_angvel is not None:
            self.max_angvel = max_angvel
        if pos_gains is not None:
            self.pos_gains = pos_gains
        if ori_gains is not None:
            self.ori_gains = ori_gains
        if pos_kd is not None:
            self.pos_kd = pos_kd
        if ori_kd is not None:
            self.ori_kd = ori_kd
        if method is not None:
            if method in ["dynamics", "pinv", "svd", "trans", "dls"]:
                self.method = method
            else:
                raise ValueError("Method must be one of 'dynamics', 'pinv', 'svd', 'trans', 'dls'")
        if inertia_compensation is not None:
            self.inertia_compensation = inertia_compensation

    def compute_gains(self, gains, kd_values, method: str, damping_ratio: float) -> np.ndarray:

        if method == "dynamics":
            kp = np.asarray(gains) 
            if kd_values is None:
                kd = damping_ratio * 2 * np.sqrt(kp)
            else:
                kd = np.asarray(kd_values)
            
        else:
            kp = np.asarray(gains) / self.integration_dt
            if kd_values is None:
                kd = 0 * kp * self.integration_dt
            else:
                kd = np.asarray(kd_values)

        return np.stack([kp, kd], axis=-1)

    def control(self, pos: Optional[np.ndarray] = None, ori: Optional[np.ndarray] = None, ) -> np.ndarray:
        # Desired position and orientation
        x_des = self.data.site_xpos[self.site_id] if pos is None else np.asarray(pos)
        if ori is None:
            mujoco.mju_mat2Quat(self.quat_des, self.data.site_xmat[self.site_id])
        else:
            self.quat_des[:] = np.asarray(ori)

        kp_kv_pos = self.compute_gains(self.pos_gains, self.pos_kd, self.method, self.trans_damping_ratio)
        kp_kv_ori = self.compute_gains(self.ori_gains, self.ori_kd, self.method, self.rot_damping_ratio)

        ddx_max = self.max_pos_error if self.max_pos_error is not None else 0.0
        dw_max = self.max_ori_error if self.max_ori_error is not None else 0.0

        q = self.data.qpos[self.dof_ids]
        dq = self.data.qvel[self.dof_ids]

        mujoco.mj_jacSite(self.model, self.data, self.J_v, self.J_w, self.site_id)
        J_v = self.J_v[:, self.dof_ids]
        J_w = self.J_w[:, self.dof_ids]
        J = np.concatenate([J_v, J_w], axis=0)

        # Position Control
        x_err = self.data.site_xpos[self.site_id] - x_des
        dx_err = J_v @ dq

        self.x_err_norm = np.linalg.norm(x_err)

        if ddx_max > 0.0 and self.x_err_norm > ddx_max:
            x_err = np.clip(x_err, -ddx_max, ddx_max)
        
        # Check positional error tolerance
        if self.x_err_norm < self.error_tolerance_pos:
            # If the error is within tolerance, set it to zero
            x_err = np.zeros_like(x_err)
            # dx_err = np.zeros_like(dx_err)

        x_err *= -kp_kv_pos[:, 0]
        dx_err *= -kp_kv_pos[:, 1]

        ddx = x_err + dx_err

        # Orientation Control
        mujoco.mju_mat2Quat(self.quat, self.data.site_xmat[self.site_id])
        mujoco.mju_negQuat(self.quat_conj, self.quat_des)
        mujoco.mju_mulQuat(self.quat_err, self.quat, self.quat_conj)
        mujoco.mju_quat2Vel(self.ori_err, self.quat_err, 1.0)
        w_err = J_w @ dq

        self.ori_err_norm = np.linalg.norm(self.ori_err)

        if dw_max > 0.0 and self.ori_err_norm > dw_max:
            self.ori_err = np.clip(self.ori_err, -dw_max, dw_max) 

        # Check orientation error tolerance
        if self.ori_err_norm < self.error_tolerance_ori:
            # If the error is within tolerance, set it to zero
            self.ori_err = np.zeros_like(self.ori_err)
            # w_err = np.zeros_like(w_err)

        self.ori_err *= -kp_kv_ori[:, 0]
        w_err *= -kp_kv_ori[:, 1]

        dw = self.ori_err + w_err

        self.error = np.concatenate([ddx, dw], axis=0)

        # # wrist_force = self._data.sensor("ur5e/wrist_force").data
        # bodyid = self.model.site_bodyid[self.model.site("attachment_site").id]
        # bodyid2 = self.model.body("connector_body").id

        # rootid = self.model.body_rootid[bodyid]
        # cfrc_int = self.data.cfrc_int[bodyid].copy()
        # total_mass = self.model.body_subtreemass[bodyid]
        # gravity_force = -self.model.opt.gravity * total_mass
        # # print("Gravity Force: ", gravity_force)
        # wrist_force = cfrc_int[3:] - gravity_force
        # print("Wrist Force before: ", wrist_force)

        # thresh = -2
        # wrist_force = [a_ - thresh if a_ < thresh else 0 for a_ in wrist_force]

        # dif = self.data.site_xpos[self.model.site("attachment_site").id] - self.data.subtree_com[rootid]
        # wrist_torque = cfrc_int[:3] - np.cross(dif, cfrc_int[3:])

        # print("Wrist Force: ", wrist_force)
        # print("Wrist Torque: ", wrist_torque)
        # direction_vector = np.array([0, 0, 1, 0, 0, 0])
        # print("Error: ", self.error)
        # compensation = 0.05 * np.array(wrist_force, dtype=np.float64)* direction_vector[:3]
        # print("Compensation: ", compensation)
        # self.error[:3] -= compensation
        # print("Error_after: ", self.error)
        # # print("cfrc_ext: ", self.data.cfrc_ext[bodyid2][3:])

        # # self.error[3:] -= 0.01 *np.array(wrist_torque, dtype=np.float64)* direction_vector[3:]

        if self.method == "dynamics":
            # Mx_inv_v = J_v @ M_inv @ J_v.T
            # Mx_inv_w = J_w @ M_inv @ J_w.T

            # # Compute Mx_v
            # if abs(np.linalg.det(Mx_inv_v)) >= 1e-2:
            #     Mx_v = np.linalg.inv(Mx_inv_v)
            # else:
            #     Mx_v = np.linalg.pinv(Mx_inv_v, rcond=1e-2)

            # # Compute Mx_w
            # if abs(np.linalg.det(Mx_inv_w)) >= 1e-2:
            #     Mx_w = np.linalg.inv(Mx_inv_w)
            # else:
            #     Mx_w = np.linalg.pinv(Mx_inv_w, rcond=1e-2)


            if self.inertia_compensation:
                mujoco.mj_fullM(self.model, self.M, self.data.qM)
                M = self.M[self.dof_ids, :][:, self.dof_ids]
                M_inv = np.linalg.inv(M)
                ddq = M_inv @ J.T @ self.error
            else:
                ddq = J.T @ self.error  # Skip M_inv if inertia compensation is off

            dq += ddq * self.integration_dt

        elif self.method == "pinv":
            J_pinv = np.linalg.pinv(J)
            dq = J_pinv @ self.error
            q += dq
        elif self.method == "svd":
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            S_inv = np.zeros_like(S)
            S_inv[S > 1e-5] = 1.0 / S[S > 1e-5]
            J_pinv = Vt.T @ np.diag(S_inv) @ U.T
            dq = J_pinv @ self.error
            q += dq
        elif self.method == "trans":
            dq = J.T @ self.error
            q += dq
        else:
            damping = 1e-4
            lambda_I = damping * np.eye(J.shape[0])
            dq = J.T @ np.linalg.inv(J @ J.T + lambda_I) @ self.error

        # Scale down joint velocities if they exceed maximum.
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max
        q += dq * self.integration_dt

        q_min = self.model.actuator_ctrlrange[:6, 0]
        q_max = self.model.actuator_ctrlrange[:6, 1]
        np.clip(q, q_min, q_max, out=q)

        return q, dq
