import pytest 
import subprocess
import pathlib as pb

# all the functional requirements of RGB are expected to be verified by tests (one test can be used to verify more than one functional requirement)

#     capability to take as input the following L1 (i.e. CIMR and three in orbit radiometers)

#     capability to execute three different algo for remapping

#     capability to execute at least one algo for geoprojection

#     capability to generate L1c from L1

#     capability to generate L1r from L1

#     capability to generate L1c from L1r

#     capability to propagate the uncertainty

# The inputs and configuration files shall be listed and detailed: one test is expected to be fully reproducible and thus associated to

#     one set of inputs

#     one configuration file

# Pass/Fail criteria are expected to be specific and detailed to allow the verification of the functional requirements as above (see comment in Sheet1 G13 of attached file)

# we need to agree on a common definition for the test methodology

#     Please verify that the definitions that are in the sheet "Definitions" in the attached spreadsheet are fine with your approach for verification

#     add a definition of for "demonstration"

@pytest.fixture
def system_config_path():
    """
    Fixture to provide the path to the system test configuration directory.
    """
    return pb.Path(__file__).parent / "configs"



@pytest.mark.parametrize("config_file", ["smap_ids_g.xml"])
def test_system_execution(system_config_path, config_file):
    """
    Test the system with various XML configuration files.

    Parameters:
    ----------
    system_config_path : Path
        Path to the configuration files directory.

    config_file : str
        The name of the configuration file to use for this test.
    """

    # Path to the configuration file
    config_path = system_config_path / config_file

    # Ensure the configuration file exists
    assert config_path.exists(), f"Configuration file {config_file} does not exist!"

    # Simulate running the main application with the configuration file
    result = subprocess.run(
        ["python", "-m", "cimr_rgb", str(config_path)],  # Adjust command if needed
        capture_output=True,
        text=True
    )

    # Check the return code
    assert result.returncode == 0, f"Execution failed with error: {result.stderr}"

    # Validate output (if your system produces a specific output file or log)
    # Example: Check that output logs contain specific messages
    #assert "Processing completed successfully" in result.stdout


