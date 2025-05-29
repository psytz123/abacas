
#!/usr/bin/env python3
"""
Example script for fetching data from a Vnish firmware miner.

This script demonstrates how to use the VnishClient to fetch telemetry data
from an ASIC miner running Vnish firmware.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api_clients import VnishClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch data from a Vnish firmware miner")
    parser.add_argument("--ip", required=True, help="IP address of the miner")
    parser.add_argument("--username", default="admin", help="Username for authentication")
    parser.add_argument("--password", required=True, help="Password for authentication")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize the Vnish client
        client = VnishClient(
            miner_ip=args.ip,
            username=args.username,
            password=args.password
        )
        
        # Fetch telemetry data
        logger.info(f"Fetching telemetry data from miner at {args.ip}")
        telemetry = client.get_telemetry()
        
        # Output the data
        if args.output:
            with open(args.output, "w") as f:
                json.dump(telemetry, f, indent=2)
            logger.info(f"Telemetry data saved to {args.output}")
        else:
            print(json.dumps(telemetry, indent=2))
        
        logger.info("Done")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
