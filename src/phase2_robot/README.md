# Phase 2: Real Robot Integration

**Goal**: Deploy policies on physical robot arm, collect real demonstrations.

---

## Structure

```
phase2_robot/
├── drivers/          # Robot SDK wrappers
│   └── robot_arm.py  # Control interface
├── teleop/           # Teleoperation
│   └── collector.py  # Demo collection
└── deploy/           # Deployment
    └── runner.py     # Real-time policy execution
```

---

## Milestones

- [ ] **M2.1** Robot arm SDK integration (vendor-specific)
- [ ] **M2.2** Camera calibration and coordinate transforms
- [ ] **M2.3** Teleoperation demo collection (keyboard/joystick)
- [ ] **M2.4** Safety limits (joint bounds, velocity, e-stop)
- [ ] **M2.5** Deploy sim-trained policy on real robot
- [ ] **M2.6** Measure real-world success rate

---

## Hardware Considerations

| Component | Notes |
|-----------|-------|
| **Robot Arm** | USB/Ethernet SDK, joint position control |
| **Camera** | RGB-D for visual policies, calibration required |
| **Gripper** | Binary or continuous control |
| **Safety** | Joint limits, max velocity, emergency stop |

---

## Key Files (to implement)

```python
# drivers/robot_arm.py
class RobotArm:
    def get_state(self) -> np.ndarray: ...
    def set_action(self, action: np.ndarray): ...
    def reset(self): ...

# teleop/collector.py  
class DemoCollector:
    def collect_episode(self) -> dict: ...
    def save_demos(self, path: str): ...

# deploy/runner.py
class PolicyRunner:
    def run_episode(self, policy, max_steps=1000): ...
```
