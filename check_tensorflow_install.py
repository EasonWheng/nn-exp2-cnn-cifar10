"""
Quick TensorFlow CPU installation check.

Run:
    python check_tensorflow_install.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import tensorflow as tf
    except Exception as exc:
        print("TensorFlow import failed.")
        print(f"Error: {exc}")
        return 1

    print("TensorFlow import succeeded.")
    print(f"TensorFlow version: {tf.__version__}")

    devices = tf.config.list_physical_devices()
    cpu_devices = tf.config.list_physical_devices("CPU")
    gpu_devices = tf.config.list_physical_devices("GPU")

    print(f"All visible devices: {devices}")
    print(f"CPU devices: {cpu_devices}")
    print(f"GPU devices: {gpu_devices}")

    # Minimal compute checks
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    add_out = tf.add(a, b).numpy()
    matmul_out = tf.matmul(a, b).numpy()

    print(f"Add result:\n{add_out}")
    print(f"MatMul result:\n{matmul_out}")
    print("TensorFlow runtime check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
