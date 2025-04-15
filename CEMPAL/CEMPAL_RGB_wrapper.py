#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import argparse
import subprocess as sbps
from pathlib import Path
import sys
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cimr-rgb-wrapper')

def parse_ipf_job_order(xml_file):
    """
    Parses an IPF Job Order XML file and extracts the input and output paths and processor version.
    
    Parameters
    ----------
    xml_file : Path
        Path to the IPF Job Order XML file.
    
    Returns
    -------
    dict
        Dictionary containing paths to the config file, L1B file, antenna patterns directory,
        output directory, and processor version.
    """
    logger.info(f"Parsing job order file: {xml_file}")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Initialize paths dictionary
        paths = {
            'cfg_file': None,
            'l1b_file': None,
            'antenna_patterns_dir': None,
            'output_dir': None,
            'processor_version': None
        }
        
        # Extract processor version
        version_elem = root.find('./Ipf_Conf/Version')
        if version_elem is not None and version_elem.text:
            paths['processor_version'] = version_elem.text.strip()
            logger.info(f"Found processor version: {paths['processor_version']}")
        
        # Extract all inputs
        input_elements = root.findall('./List_of_Ipf_Procs/Ipf_Proc/List_of_Inputs/Input')
        logger.info(f"Found {len(input_elements)} input elements in job order file.")
        
        for input_elem in input_elements:
            file_type_elem = input_elem.find('File_Type')
            if file_type_elem is None:
                continue
                
            file_type = file_type_elem.text.strip()
            
            # Find the file name within this input
            file_name_elem = input_elem.find('./List_of_File_Names/File_Name')
            if file_name_elem is None or not file_name_elem.text:
                continue
                
            file_path = file_name_elem.text.strip()
            
            # Assign path based on file type
            if file_type == 'CFG_RGB_':
                paths['cfg_file'] = file_path
                logger.info(f"Found config file: {file_path}")
            elif file_type == 'L1B_':
                paths['l1b_file'] = file_path
                logger.info(f"Found L1B file: {file_path}")
            elif file_type == 'AntennaPattern':
                paths['antenna_patterns_dir'] = file_path
                logger.info(f"Found antenna patterns directory: {file_path}")
        
        # Extract output directory
        output_elements = root.findall('./List_of_Ipf_Procs/Ipf_Proc/List_of_Outputs/Output')
        for output_elem in output_elements:
            file_type_elem = output_elem.find('File_Type')
            if file_type_elem is not None:
                output_type = file_type_elem.text.strip()
                # Check for either L1C or L1R output type
                if output_type in ['OUT_L1C', 'OUT_L1R']:
                    file_name_elem = output_elem.find('File_Name')
                    if file_name_elem is not None and file_name_elem.text:
                        paths['output_dir'] = file_name_elem.text.strip()
                        logger.info(f"Found output directory for {output_type}: {paths['output_dir']}")
                        # Once we've found a valid output directory, we can break
                        break
        
        # Validate that we found all required paths
        missing_paths = [k for k, v in paths.items() if v is None and k != 'processor_version']
        if missing_paths:
            logger.error(f"Missing required paths in JobOrder: {', '.join(missing_paths)}")
            return None
            
        return paths
        
    except Exception as e:
        logger.error(f"Error parsing job order file: {e}")
        return None


def create_config_file(paths):
    """
    Creates a configuration XML file for the CIMR-RGB processor by merging
    the content of the CFG_RGB_ file with paths extracted from the JobOrder file.
    
    Parameters
    ----------
    paths : dict
        Dictionary containing paths to the config file, L1B file, antenna patterns directory,
        output directory, and processor version.
    
    Returns
    -------
    str
        Path to the generated config file.
    """
    logger.info("Creating config file for CIMR-RGB processor")
    
    # First, parse the CFG_RGB_ file to get the base configuration
    if not os.path.exists(paths['cfg_file']):
        logger.error(f"Error: CFG_RGB_ file does not exist at path: {paths['cfg_file']}")
        return None
    
    try:
        # Parse the CFG_RGB_ file
        tree = ET.parse(paths['cfg_file'])
        root = tree.getroot()
        
        # Find or create the InputData element
        input_data = root.find("InputData")
        if input_data is None:
            input_data = ET.SubElement(root, "InputData")
            
            # Add default input data elements if they don't exist
            if input_data.find("type") is None:
                ET.SubElement(input_data, "type").text = "SMAP"
            if input_data.find("split_fore_aft") is None:
                ET.SubElement(input_data, "split_fore_aft").text = "True"
            if input_data.find("source_band") is None:
                ET.SubElement(input_data, "source_band").text = "L"
            if input_data.find("target_band") is None:
                ET.SubElement(input_data, "target_band").text = "L"
            if input_data.find("quality_control") is None:
                ET.SubElement(input_data, "quality_control").text = "True"
        
        # Update input paths
        path_elem = input_data.find("path")
        if path_elem is not None:
            path_elem.text = paths['l1b_file']
        else:
            ET.SubElement(input_data, "path").text = paths['l1b_file']
            
        ant_path_elem = input_data.find("antenna_patterns_path")
        if ant_path_elem is not None:
            ant_path_elem.text = paths['antenna_patterns_dir']
        else:
            ET.SubElement(input_data, "antenna_patterns_path").text = paths['antenna_patterns_dir']
        
        # Find or create the OutputData element
        output_data = root.find("OutputData")
        if output_data is None:
            output_data = ET.SubElement(root, "OutputData")
            
            # Add default output data elements if they don't exist
            if output_data.find("save_to_disk") is None:
                ET.SubElement(output_data, "save_to_disk").text = "True"
            if output_data.find("version") is None:
                ET.SubElement(output_data, "version").text = paths.get('processor_version', "1.0.0")
            if output_data.find("creator_name") is None:
                ET.SubElement(output_data, "creator_name").text = "Insert your name"
            if output_data.find("creator_email") is None:
                ET.SubElement(output_data, "creator_email").text = "example@example.com"
            if output_data.find("creator_url") is None:
                ET.SubElement(output_data, "creator_url").text = "https://example.com"
            if output_data.find("creator_institution") is None:
                ET.SubElement(output_data, "creator_institution").text = "Insert your institution"
            if output_data.find("timestamp_fmt") is None:
                ET.SubElement(output_data, "timestamp_fmt").text = "%Y-%m-%d_%H-%M-%S"
        
        # Update output path
        output_path_elem = output_data.find("output_path")
        if output_path_elem is not None:
            output_path_elem.text = paths['output_dir']
        else:
            ET.SubElement(output_data, "output_path").text = paths['output_dir']
        
        # Update version if it exists
        if 'processor_version' in paths:
            version_elem = output_data.find("version")
            if version_elem is not None:
                version_elem.text = paths['processor_version']
            else:
                ET.SubElement(output_data, "version").text = paths['processor_version']
        
        # Write the config to a file in the output directory
        output_filename = os.path.join(paths['output_dir'], "generated_config.xml")
        tree.write(output_filename, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Generated config file at: {output_filename}")
        return output_filename
        
    except Exception as e:
        logger.error(f"Error creating config file: {e}")
        return None


def execute_cimr_rgb(config_file, processor_command="cimr-rgb"):
    """
    Executes the processor with the config file, streaming output to the terminal live.
    
    Parameters
    ----------
    config_file : str
        Path to the config XML file.
    processor_command : str
        The command to run the processor (default: "cimr-rgb")
    """
    try:
        # Check if config file exists
        if not os.path.exists(config_file):
            logger.error(f"Error: Config file does not exist at path: {config_file}")
            sys.exit(1)
            
        # Construct the command to run the processor with the config file
        cmd = [processor_command, config_file]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # Run the command and stream output to the terminal
        process = sbps.run(
            cmd,
            stdout=sys.stdout,  # Stream subprocess stdout live
            stderr=sys.stderr,  # Stream subprocess stderr live
            text=True,
            check=True  # Raise exception if the command fails
        )
        
        logger.info(f"Processor execution completed successfully.")
        
    except sbps.CalledProcessError as e:
        logger.error(f"Error executing processor: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    """
    Main function to parse command-line arguments, extract paths from JobOrder,
    create a config file, and execute the processor.
    """
    logger.info(f"Python wrapper started. Args: {sys.argv}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    parser = argparse.ArgumentParser(
        description="Parse an IPF Job Order XML file, create a config file, and execute the CIMR-RGB processor."
    )
    parser.add_argument(
        "job_order_file", type=str, nargs='?',
        help="Path to the IPF Job Order XML file"
    )
    parser.add_argument(
        "--processor", default="cimr-rgb", help="Command to run the processor (default: cimr-rgb)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (print config file path without running processor)"
    )
    
    # For debugging, print all command-line arguments
    logger.info(f"All command-line arguments: {sys.argv}")
    
    # Parse known arguments, ignore unknown ones
    args, unknown = parser.parse_known_args()
    
    # If we have unknown arguments, check if any of them might be the job order file
    job_order_file = args.job_order_file
    if job_order_file is None and unknown:
        for arg in unknown:
            if arg.endswith('.xml') and os.path.exists(arg):
                job_order_file = arg
                logger.info(f"Found potential job order file from unknown args: {job_order_file}")
                break
    
    # If still no job order file, check if any argument contains "JobOrder" and ends with .xml
    if job_order_file is None:
        for arg in sys.argv[1:]:
            if 'JobOrder' in arg and arg.endswith('.xml') and os.path.exists(arg):
                job_order_file = arg
                logger.info(f"Found potential job order file based on name pattern: {job_order_file}")
                break
    
    if job_order_file is None or not os.path.exists(job_order_file):
        logger.error(f"Error: No valid job order file found or specified.")
        return 1
    
    # Convert to Path object
    job_order_path = Path(job_order_file)
    
    # Parse the job order file to get input and output paths
    paths = parse_ipf_job_order(job_order_path)
    
    if paths is None:
        logger.error("Failed to extract required paths from JobOrder file.")
        return 1
    
    # Create a config file using the extracted paths and CFG_RGB_ file
    config_file = create_config_file(paths)
    
    if config_file is None:
        logger.error("Failed to create config file.")
        return 1
    
    if args.debug:
        logger.info(f"Debug mode: Would execute processor with config file: {config_file}")
        return 0
    
    # Execute the processor with the generated config file
    execute_cimr_rgb(config_file, args.processor)
    return 0


if __name__ == "__main__":
    sys.exit(main())