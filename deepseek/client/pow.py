"""
Proof of Work (PoW) implementation for DeepSeek API

Python port of the Go PoW solver that handles the challenge-response
mechanism used by DeepSeek to prevent abuse.
"""

import hashlib
import random
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config.logger import get_logger

logger = get_logger("pow")


# Target difficulty (bits)
DEFAULT_DIFFICULTY = 16


@dataclass
class PowChallenge:
    """PoW challenge from DeepSeek API."""
    challenge_id: str
    challenge: str
    prefix: str
    difficulty: int = DEFAULT_DIFFICULTY
    timestamp: int = 0


@dataclass
class PowResult:
    """PoW solution result."""
    challenge_id: str
    proof_of_work: str
    timestamp: int


class PowSolver:
    """
    Pure Python PoW solver for DeepSeek challenges.

    Implements a simple hashcash-style proof of work where the client
    must find a nonce such that SHA256(prefix + nonce) starts with
    `difficulty` zero bits.
    """

    def __init__(self, difficulty: int = DEFAULT_DIFFICULTY):
        self.difficulty = difficulty
        self._lock = threading.Lock()
        self._solving = False
        self._solution: Optional[PowResult] = None

    def solve(self, challenge: PowChallenge) -> Optional[PowResult]:
        """
        Solve a PoW challenge.

        Args:
            challenge: The challenge to solve

        Returns:
            PowResult with the solution, or None if failed
        """
        prefix = challenge.prefix or ""
        target = challenge.challenge or ""
        difficulty = challenge.difficulty or self.difficulty

        # Target: number of leading zero bits
        # We need to find nonce such that SHA256(prefix + nonce) has at least
        # `difficulty` leading zero bits
        nonce_bytes_needed = (difficulty + 7) // 8

        start_time = time.time()
        attempts = 0
        nonce = random.randint(0, 2**63 - 1)

        logger.debug(f"Starting PoW solve: difficulty={difficulty}, prefix={prefix[:20]}...")

        while attempts < 100_000:  # Reduced limit for DeepSeekHashV1 (unsolvable with current algo)
            nonce_bytes = struct.pack(">Q", nonce)

            # Combine prefix and nonce
            data = prefix.encode() + nonce_bytes
            hash_result = hashlib.sha256(data).digest()

            # Check if hash starts with enough zero bytes
            if self._check_hash(hash_result, nonce_bytes_needed):
                elapsed = time.time() - start_time
                solution = PowResult(
                    challenge_id=challenge.challenge_id,
                    proof_of_work=f"{nonce}",
                    timestamp=int(time.time()),
                )
                logger.debug(f"PoW solved in {elapsed:.2f}s, {attempts} attempts")
                return solution

            nonce = (nonce + 1) % (2**64)
            attempts += 1

            # Yield occasionally to not block
            if attempts % 100_000 == 0:
                time.sleep(0)

        logger.warning(f"PoW solve failed after {attempts} attempts")
        return None

    def _check_hash(self, hash_bytes: bytes, zero_bytes: int) -> bool:
        """Check if hash has enough leading zero bytes."""
        if len(hash_bytes) < zero_bytes:
            return False

        # All leading bytes must be zero
        for i in range(zero_bytes):
            if hash_bytes[i] != 0:
                return False

        return True

    def solve_async(self, challenge: PowChallenge, callback=None):
        """Solve PoW in background thread."""
        def solve_thread():
            result = self.solve(challenge)
            with self._lock:
                self._solution = result
                self._solving = False
            if callback:
                callback(result)

        with self._lock:
            if self._solving:
                return  # Already solving
            self._solving = True
            self._solution = None

        thread = threading.Thread(target=solve_thread, daemon=True)
        thread.start()


def parse_pow_response(response: dict) -> Optional[PowChallenge]:
    """
    Parse PoW challenge from API response.

    Handles both flat and nested response formats:
    - Flat: {"challenge_id": ..., "challenge": ..., "prefix": ..., "difficulty": ...}
    - Nested: {"data": {"biz_data": {"challenge": {"challenge": ..., "salt": ..., "signature": ..., "difficulty": ...}}}}
    """
    try:
        # Try nested format first
        challenge_data = response
        outer_data = response.get("data", {})
        if isinstance(outer_data, dict):
            biz_data = outer_data.get("biz_data", {})
            if isinstance(biz_data, dict):
                challenge_data = biz_data.get("challenge", response)

        challenge_id = str(challenge_data.get("challenge_id", challenge_data.get("challenge", "")))
        challenge_value = str(challenge_data.get("challenge", ""))
        prefix = str(challenge_data.get("prefix", challenge_data.get("salt", "")))
        difficulty = int(challenge_data.get("difficulty", DEFAULT_DIFFICULTY))
        timestamp = int(challenge_data.get("timestamp", challenge_data.get("expire_at", 0)))

        if not challenge_value:
            return None

        return PowChallenge(
            challenge_id=challenge_id,
            challenge=challenge_value,
            prefix=prefix,
            difficulty=difficulty,
            timestamp=timestamp,
        )
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to parse PoW response: {e}")
        return None


def format_pow_solution(solution: PowResult) -> dict:
    """Format PoW solution for API request."""
    return {
        "challenge_id": solution.challenge_id,
        "proof_of_work": solution.proof_of_work,
        "timestamp": solution.timestamp,
    }


# Global solver instance
_solver: Optional[PowSolver] = None
_solver_lock = threading.Lock()


def get_solver() -> PowSolver:
    """Get or create global PoW solver."""
    global _solver
    with _solver_lock:
        if _solver is None:
            _solver = PowSolver()
        return _solver


def solve_pow(challenge: PowChallenge) -> Optional[PowResult]:
    """Quick helper to solve a PoW challenge."""
    # Try DeepSeekPowSolver package first (handles DeepSeekHashV1)
    try:
        from deepseekpowsolver import DeepSeekPowSolver
        solver = DeepSeekPowSolver()
        # Build full challenge dict for the solver
        ch_dict = {
            "challenge": challenge.challenge,
            "salt": challenge.prefix,
            "difficulty": challenge.difficulty,
            "algorithm": "DeepSeekHashV1",
        }
        result = solver.SleviS(ch_dict)
        if result:
            return PowResult(
                challenge_id=challenge.challenge_id,
                proof_of_work=result if isinstance(result, str) else str(result),
                timestamp=int(time.time()),
            )
    except Exception:
        pass
    
    # Fall back to simple SHA256 solver
    solver = get_solver()
    return solver.solve(challenge)
