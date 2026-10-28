
from ikpy.chain import Chain
import numpy as np

arm = Chain.from_urdf_file("arm.urdf")

# Find the joint indices from the URDF
joint_indices = {
    link.name: i for i, link in enumerate(arm.links)
}

required_joints = [
    "base_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_pitch_joint",
    "wrist_pitch_joint",
]

for name in required_joints:
    if name not in joint_indices:
        raise ValueError(f"Joint not found in URDF: {name}")

base_i = joint_indices["base_yaw_joint"]
shoulder_i = joint_indices["shoulder_pitch_joint"]
elbow_i = joint_indices["elbow_pitch_joint"]
wrist_i = joint_indices["wrist_pitch_joint"]

# Only base, shoulder and elbow affect the wrist position.
arm.active_links_mask = np.array([
    link.name in {
        "base_yaw_joint",
        "shoulder_pitch_joint",
        "elbow_pitch_joint",
    }
    for link in arm.links
], dtype=bool)


def inverse_kinematics(x_mm, y_mm, z_mm):
    """Return driver joint angles in degrees for a wrist target in mm."""

    target = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0

    # Starting guess for IKPy
    base_guess = np.arctan2(y_mm, x_mm)
    radius = np.hypot(x_mm, y_mm)
    height = z_mm - 71.532
    shoulder_guess = -np.arctan2(height, max(radius, 1e-9))

    candidates = []

    # Try both elbow configurations
    for elbow_guess in [1.2, -1.2]:
        initial = np.zeros(len(arm.links))
        initial[base_i] = base_guess
        initial[shoulder_i] = shoulder_guess
        initial[elbow_i] = elbow_guess
        initial[wrist_i] = 0.0

        # IKPy performs the inverse kinematics calculation
        solution = arm.inverse_kinematics(
            target_position=target,
            initial_position=initial,
        )

        transforms = arm.forward_kinematics(
            solution, full_kinematics=True
        )
        reached = transforms[-1][:3, 3]
        error = np.linalg.norm(reached - target)

        # World-space height of the elbow pivot
        elbow_height = transforms[elbow_i][2, 3]

        candidates.append(
            (solution, error, elbow_height)
        )

    # Select the solution with the highest elbow
    solution, error, _ = max(
        candidates, key=lambda candidate: candidate[2]
    )

    if error > 0.005:
        raise ValueError(
            f"Target unreachable or IK failed: "
            f"position error = {error * 1000:.2f} mm"
        )

    # Convert IKPy radians to degrees
    base_deg = np.degrees(solution[base_i])
    shoulder_deg = np.degrees(solution[shoulder_i])
    elbow_deg = np.degrees(solution[elbow_i])
    wrist_deg = np.degrees(solution[wrist_i])

    # Apply your motor driver's offsets
    base_driver = base_deg + 180.0
    shoulder_driver = shoulder_deg + 360.0

    # Convert the signed IK angle to a 0–360 range,
    # then reverse the motor direction.
    elbow_normalized = elbow_deg % 360.0

    if np.isclose(elbow_normalized, 0.0):
        elbow_driver = (
            360.0 if np.isclose(elbow_deg, 0.0) else 0.0
        )
    else:
        elbow_driver = 360.0 - elbow_normalized

    # Convert IKPy angles from radians to degrees
    base_deg = np.degrees(solution[base_i])
    shoulder_deg = np.degrees(solution[shoulder_i])
    elbow_deg = np.degrees(solution[elbow_i])

    # Apply your motor driver's transformations
    return {
        "base": float(base_deg + 180.0),
        "shoulder": float(-shoulder_deg),
        "elbow": float(180.0 - elbow_deg),
        "wrist": 0.0,
    }
