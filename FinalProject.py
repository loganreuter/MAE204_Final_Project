# Library imports
import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt

import csv
import math

R = 0.0475 # meters
W = 0.15 # meters
L = 0.47/2 # meters
PINV_THRESHOLD = 1e-2

def NextState(curr_state: np.ndarray, velocities: np.ndarray, dt: float, max_vel: float) -> np.ndarray:
    next_state = np.zeros(13)

    velocities[np.where(velocities > max_vel)] = max_vel
    velocities[np.where(velocities < -max_vel)] = -max_vel

    # determine the new joint angles
    next_state[3:8] = curr_state[3:8] + dt * velocities[4:]

    # determine the new wheel angles
    next_state[8:12] = curr_state[8:12] + dt * velocities[:4]

    # calculate the change in wheel angles
    d_wheel_angles = next_state[8:12] - curr_state[8:12]

    # eqn 13.10 (pg. 541)
    H_0 =  (1/R) * np.array([[-L-W, 1, -1], [L+W, 1, 1], [L+W, 1, -1], [-L-W, 1, 1]])

    # body twist is a 3x1 vector
    # eqn 13.33 (pg. 569)
    #   w_bz
    #   v_bx
    #   v_by
    body_twist = np.matmul(np.linalg.pinv(H_0, rcond=PINV_THRESHOLD), d_wheel_angles)

    # eqn 13.35 (pg. 570)
    dq_b = np.zeros(3)
    if body_twist[0] == 0.0:
        dq_b = body_twist
    else:
        dq_b = np.array([
            body_twist[0],
            ( body_twist[1] * np.sin(body_twist[0]) + body_twist[2] * (np.cos(body_twist[0]) - 1) ) / body_twist[0],
            ( body_twist[2] * np.sin(body_twist[0]) + body_twist[1] * (1 - np.cos(body_twist[0])) ) / body_twist[0]
        ])
    
    # eqn 13.36 (pg 570)
    dq = np.matmul(np.array([
        [1, 0, 0],
        [0, np.cos(curr_state[0]), -np.sin(curr_state[0])],
        [0, np.sin(curr_state[0]), np.cos(curr_state[0])],
    ]), dq_b)

    next_state[:3] = curr_state[:3] + dq

    return next_state    

def TrajectoryGenerator(T_se_i, T_sc_i, T_sc_f, T_ce_grasp, T_ce_standoff, k: int, Tf: int, write_debug: bool = False):
    # number of configurations
    N = int( (Tf * k) / 0.001 )
    output = [() for _ in range(6 * N + 2)]
    idx = 0

    # Trajectory to standoff position
    T_se_standoff_pick = np.matmul(T_sc_i, T_ce_standoff)
    traj = mr.ScrewTrajectory(T_se_i, T_se_standoff_pick, Tf, N, 5)
    for t in traj:
        output[idx] = (t, 0)
        idx += 1

    # Trajectory to grasp position
    T_se_grasp_pick = np.matmul(T_sc_i, T_ce_grasp)
    traj = mr.ScrewTrajectory(T_se_standoff_pick, T_se_grasp_pick, Tf, N, 5)
    for t in traj:
        output[idx] = (t, 0)
        idx += 1

    # open the gripper
    output[idx] = (T_se_grasp_pick, 1)
    idx += 1

    # move back to the standoff point for the block pick up
    traj = mr.ScrewTrajectory(T_se_grasp_pick, T_se_standoff_pick, Tf, N, 5)
    for t in traj:
        output[idx] = (t, 1)
        idx += 1

    # move to the standoff point for the block drop off
    T_se_standoff_drop = np.matmul(T_sc_f, T_ce_standoff)
    traj = mr.ScrewTrajectory(T_se_standoff_pick, T_se_standoff_drop, Tf, N, 5)
    for t in traj:
        output[idx] = (t, 1)
        idx += 1

    # place down the block
    T_se_grasp_drop = np.matmul(T_sc_f, T_ce_grasp)
    traj = mr.ScrewTrajectory(T_se_standoff_drop, T_se_grasp_drop, Tf, N, 5)
    for t in traj:
        output[idx] = (t, 1)
        idx += 1
    
    # open the gripper
    output[idx] = (T_se_grasp_drop, 0)
    idx += 1

    # move the the standoff point above the drop location
    traj = mr.ScrewTrajectory(T_se_grasp_drop, T_se_standoff_drop, Tf, N, 5)
    for t in traj:
        output[idx] = (t, 0)
        idx += 1

    if write_debug:
        with open("trajectory.csv", "w+") as f:
            writer = csv.writer(f)
            for traj, gripper_state in output:
                writer.writerow(np.concatenate((traj[:3, :3].flatten(), traj[:3, 3], [gripper_state])))
        

    return output

def FeedbackControl(joint_angles, X_d, X_d_next, K_p, K_i, delt, joint_constraints: list[bool]):
    l = L
    w = W
    r = R
    error = np.zeros((6,1))
    
    phi = joint_angles[0]
    x = joint_angles[1]
    y = joint_angles[2]
    
    F = r/4 * np.asarray([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)], [1, 1, 1, 1], [-1, 1, -1, 1]])
    
    F6 = np.zeros((6,4))
    F6[2:5, :] = F
    
    M = np.asarray([[1, 0, 0, 0.033], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])

    T_b0 = np.asarray([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])
    
    T_sb = np.asarray([[np.cos(phi), -np.sin(phi), 0, x], [np.sin(phi), np.cos(phi), 0, y], [0, 0, 1, 0.0963], [0, 0, 0, 1]])
    
    Vd = mr.se3ToVec(1/delt * mr.MatrixLog6(mr.TransInv(X_d) @ X_d_next)).reshape((6,1))
    
    Blist = np.asarray([[0, 0, 1, 0, 0.033, 0], [0, -1, 0, -0.5076, 0, 0], 
                        [0, -1, 0, -0.3526, 0, 0], [0, -1, 0, -0.2176, 0, 0], 
                       [0, 0, 1, 0, 0, 0]]).T
    
    Jarm = mr.JacobianBody(Blist, joint_angles[3:])
    
    T_0e = mr.FKinBody(M, Blist, joint_angles[3:])
    
    # Calculate the new end effector position (pg. 570)
    X = T_sb @ T_b0 @ T_0e

    ad2 = mr.Adjoint(mr.TransInv(T_0e)@mr.TransInv(T_b0))
    
    Jbase = ad2 @ F6
    
    J = np.concatenate((Jbase,Jarm), axis=1)
    # J[:, joint_constraints] = 0

    Xerr = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ X_d)).reshape((6,1))
    ad = mr.Adjoint(mr.TransInv(X) @ X_d)
    error += Xerr*delt

    V = ad @ Vd + K_p @ Xerr + K_i @ error
    
    output = np.linalg.pinv(J, rcond=PINV_THRESHOLD) @ V

    Aw = J[:3, :] @ J[:3, :].T
    Av = J[3:, :] @ J[3:, :].T

    eig_w = np.linalg.eigvals(Aw)
    eig_v = np.linalg.eigvals(Av)

    manip_w = -1
    manip_v = -1
    if abs(min(eig_w)) > PINV_THRESHOLD:
        manip_w = math.sqrt( max(eig_w) / min(eig_w) )
    
    if abs(min(eig_v)) > PINV_THRESHOLD:
        manip_v = math.sqrt( max(eig_v) / min(eig_v) )

    return (output, Xerr, manip_w, manip_v)

def test_state(curr_state: np.ndarray) -> list[bool]:
    joint_constraints = [False for _ in range(9)]

    arm_joints = curr_state[3:8]
    arm_constr_offst = 4

    # if arm_joints[0] < -1e-2 or arm_joints[0] > 1e-2:
    #     joint_constraints[4] = True

    if arm_joints[1] > np.pi/2 or arm_joints[1] < -1.1:
        joint_constraints[5] = True

    if arm_joints[2] > -0.1:
        joint_constraints[4] = True

    if arm_joints[3] > -0.1:
        joint_constraints[7] = True

    return joint_constraints

def FullProgram() -> None:
    """
        T_ci: SE(6) Initial Resting Configuration of the cube frame
        T_cd: SE(6) Desired Configuration of the cube frame
        T_ri: SE(6) Initial configuration of the robot
        T_ref: SE(6) Reference Initial Configuration
        K_p: float Proportional gain
        K_i: float Integral Gain
    """

    steps = []
    err_v_time = []
    linear_manip = []
    angular_manip = []

    # T_se_i = np.array([[0.170, 0, 0.985, 0.387], [0, 1, 0, 0], [-0.985, 0, 0.170, 0.570], [0, 0, 0, 1]])
    
    # Original Task Config
    T_se_i = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]])
    T_sc_i = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0.25], [0, 0, 0, 1]])
    T_sc_f = np.array([[0, 1, 0, 0], [-1, 0, 0, -1], [0, 0, 1, .25], [0, 0, 0, 1]])
    T_ce_grasp = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -0.25], [0, 0, 0, 1]])
    T_ce_standoff = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -0.15], [0, 0, 0, 1]])

    # New Task Config
    # T_se_i = np.array([[-1, 0, 0.002, 0.335], [0, 1, 0, 0], [-0.002, 0, -1, 0.183], [0, 0, 0, 1]])
    # T_sc_i = np.array([[1, 0, 0, 1.5], [0, 1, 0, 0.5], [0, 0, 1, 0.25], [0, 0, 0, 1]])
    # T_sc_f = np.array([[1, 0, 0, 1.5], [0, 1, 0, -0.5], [0, 0, 1, 0.25], [0, 0, 0, 1]])
    # T_ce_grasp = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -0.25], [0, 0, 0, 1]])
    # T_ce_standoff = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -0.15], [0, 0, 0, 1]])

    # Best:
    K_p = np.eye(6) * [1, 0.15, 1, 0.7, 1, 0.25]
    K_i = np.zeros((6,6))

    # Overshoot:
    # K_p = np.eye(6) * [1, 0.15, 1, 5, 1, 0.25]
    # K_i = np.eye(6) * [20, 20, 20, 20, 20, 20]

    # New Task:
    # K_p = np.eye(6) * [1, 0.5, 1, 0.7, 1, 0.75]
    # K_i = np.zeros((6,6))

    k = 1 # number of steps per 0.001 seconds
    Tf = 2 # trajectory runtime (s)
    dt = 0.01
    max_speed = 5 # m/s

    trajectory = TrajectoryGenerator(T_se_i, T_sc_i, T_sc_f, T_ce_grasp, T_ce_standoff, k, Tf, write_debug=True)
    curr_state = np.array([0, 0, 0, 0, 0, -np.pi/4, -np.pi/4, 0, 0, 0, 0, 0, 0])

    for idx, traj in enumerate(trajectory):
        step, gripper_state = traj
        if idx == len(trajectory) - 1:
            break
        
        T_se_d = step
        T_se_d_next, _ = trajectory[idx + 1]

        joint_constraints = [False for _ in range(9)]

        velocities, err, manip_w, manip_v = FeedbackControl(np.array(curr_state[:8]), T_se_d, T_se_d_next, K_p, K_i, dt, joint_constraints)

        next_state = NextState(curr_state, velocities.reshape(1, 9)[0], dt, max_speed)

        joint_constraints = test_state(next_state)

        if True in joint_constraints:
            velocities, err, manip_w, manip_v = FeedbackControl(np.array(curr_state[:8]), T_se_d, T_se_d_next, K_p, K_i, dt, joint_constraints)

            next_state = NextState(curr_state, velocities.reshape(1, 9)[0], dt, max_speed)            

        curr_state = next_state

        curr_state[12] = gripper_state

        steps.append(curr_state)
        err_v_time.append(err)
        angular_manip.append(manip_w)
        linear_manip.append(manip_v)
    
    with open("final.csv", "w+") as file:
        writer = csv.writer(file)

        writer.writerows(steps)
    
    err_v_time = np.array(err_v_time)
    angular_manip = np.array(angular_manip)
    angular_manip[np.where(angular_manip == -1)] = max(angular_manip)
    linear_manip = np.array(linear_manip)
    linear_manip[np.where(linear_manip == -1)] = max(linear_manip)
    t = np.linspace(0, err_v_time.shape[0], err_v_time.shape[0]) * dt
    # for idx in range(len(err_v_time[0])):
    #     plt.plot(t, err_v_time[:, idx], label=f"{idx + 1}")
    plt.plot(t, angular_manip, label="u_1(Aw)")
    plt.plot(t, linear_manip, label="u_1(Av)")
    plt.legend()
    plt.show()
    

FullProgram()

# X = np.array([[0.170, 0, 0.985, 0.387], [0, 1, 0, 0], [-0.985, 0, 0.170, 0.570], [0, 0, 0, 1]])
# Xd = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]])
# Xd_next = np.array([[0, 0, 1, 0.6], [0, 1, 0, 0], [-1, 0, 0, 0.3], [0, 0, 0, 1]])
# curr_state = np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0, 0])
# dt = 0.01
# K_p = np.zeros((6,6))
# K_i = np.zeros((6,6))
# vel, err, T_se = FeedbackControl(X, Xd, Xd_next, K_p, K_i, dt, curr_state[:8])

# next_st = NextState(curr_state, vel.reshape(1, 9)[0], dt, 10)
# with open("test.csv", "w+") as f:
#     writer = csv.writer(f)
#     writer.writerow(curr_state)
#     writer.writerow(next_st)
