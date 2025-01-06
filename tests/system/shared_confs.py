"""
Contains shared fixtures and other configurations across all tests
"""

import sys
import subprocess as sbps

import pytest


@pytest.fixture
def run_subprocess():
    """
    Pytest fixture to execute a subprocess with real-time output.

    This fixture provides a callable function to run a subprocess,
    streaming the output directly to the console during execution.

    Returns
    -------
    callable
        A function that takes a configuration file path as input and
        runs the subprocess. The function returns the subprocess exit code.

    Examples
    --------
    def test_example(run_python_subprocess):
        exit_code = run_python_subprocess("path/to/config.xml")
        assert exit_code == 0
    """

    def _run(config_path):
        try:
            command = ["python", "-m", "cimr_rgb", str(config_path)]
            result = sbps.run(
                command,
                stdout=sys.stdout,  # Stream subprocess stdout live
                stderr=sys.stderr,  # Stream subprocess stderr live
                text=True,
            )
            return result.returncode
        except Exception as e:
            print(f"Error occurred: {e}", file=sys.stderr)
            return -1

    return _run
