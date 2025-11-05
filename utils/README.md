# Utility Scripts

## calibrate_sim_offset.py

Automatically calculates the `sim_offset` by comparing `/objects_poses_sim` and `/objects_poses_real` topics.

### Usage:

1. Make sure both simulation and real world are running and publishing poses to their respective topics
2. Make sure the same objects are visible in both sim and real world
3. Run the script:

**Compare all common objects:**
```bash
python3 utils/calibrate_sim_offset.py
```

**Compare only specific objects:**
```bash
python3 utils/calibrate_sim_offset.py --object-name line_brown
```

**Compare multiple specific objects:**
```bash
python3 utils/calibrate_sim_offset.py --object-name line_brown --object-name fork_orange
```

### Output:

The script will:
- Listen to both `/objects_poses_sim` and `/objects_poses_real`
- Compare positions of common objects
- Calculate the average offset
- Display the recommended `sim_offset` value to update in `localizer_bridge.py`

### Example Output:

```
================================================================================
RECOMMENDED SIM_OFFSET
================================================================================

Average offset (meters): [-0.011800, -0.007000, 0.000000]
Average offset (mm):     [-11.80, -7.00, 0.00]

================================================================================
UPDATE localizer_bridge.py WITH:
================================================================================
self.sim_offset = np.array([-0.011800, -0.007000, 0.0])
```

Press `Ctrl+C` to stop the script.

