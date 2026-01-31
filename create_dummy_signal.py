import numpy as np
import os

try:
    signal = np.random.randn(12, 1000).astype(np.float32)
    np.save("sample.npy", signal)
    print(f"Created sample.npy at {os.path.abspath('sample.npy')}")
except Exception as e:
    print(f"Error: {e}")
