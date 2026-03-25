"""
generate_sample_data.py
=======================
Creates a realistic synthetic copper casting dataset in /data
for testing the analysis agent locally or in CI.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 480  # 8 hours at 1-min intervals

timestamps = pd.date_range("2024-06-01 06:00", periods=n, freq="1min")

# Normal process with occasional anomalies
furnace_temp    = np.random.normal(1105, 4, n)
furnace_temp[50:55] = 1150   # spike — critical

casting_speed   = np.random.normal(8.5, 0.3, n)
casting_speed[200:210] = 4.2  # below LSL

rod_exit_temp   = np.random.normal(540, 15, n)

cooling_water_flow = np.random.normal(45, 3, n)
cooling_water_temp = np.random.normal(28, 2, n)
cooling_water_temp[300:310] = np.nan  # missing data block

rod_diameter    = np.random.normal(8.0, 0.05, n)
rod_diameter[400] = 8.35   # outlier

oxygen_content  = np.random.normal(2.5, 0.8, n).clip(0)
conductivity    = np.random.normal(100.8, 0.15, n)

tensile_strength = np.random.normal(220, 8, n)
# Introduce downward drift in last 100 rows
tensile_strength[380:] -= np.linspace(0, 30, 100)

elongation      = np.random.normal(32, 2, n)

# Frozen sensor: cooling_water_flow frozen for 8 rows
cooling_water_flow[150:158] = cooling_water_flow[150]

df = pd.DataFrame({
    "timestamp":            timestamps,
    "furnace_temp":         furnace_temp.round(2),
    "casting_speed":        casting_speed.round(3),
    "rod_exit_temp":        rod_exit_temp.round(2),
    "cooling_water_flow":   cooling_water_flow.round(2),
    "cooling_water_temp":   cooling_water_temp.round(2),
    "rod_diameter":         rod_diameter.round(4),
    "oxygen_content":       oxygen_content.round(3),
    "conductivity":         conductivity.round(4),
    "tensile_strength":     tensile_strength.round(2),
    "elongation":           elongation.round(2),
})

out = Path("data/sample_cast_data.csv")
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Sample data written to {out}  ({len(df)} rows)")
