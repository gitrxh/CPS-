"""Relaunch the unchanged phase-one protocol after a missing-config preflight failure.

The failed v1 queue performed no scoring, training, or evaluation. Its receipt is
preserved. This wrapper changes only the queue log/claim directory.
"""
import run_phase1_queue as protocol

protocol.QUEUE = "outputs/phase1_queue_v2"

if __name__ == "__main__":
    protocol.main()
