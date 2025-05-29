
#!/usr/bin/env python3
"""
Example script for fetching data from Prohashing.com.

This script demonstrates how to use the ProhashingClient to fetch profitability
data and worker performance from Prohashing.com.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from api_clients import ProhashingClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch data from Prohashing.com")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--worker-id", help="Worker ID for fetching worker performance")
    parser.add_argument("--algorithm", default="Scrypt", help="Mining algorithm (e.g., Scrypt, SHA-256)")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--type", choices=["profitability", "worker", "all"], default="all",
                        help="Type of data to fetch (default: all)")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize the Prohashing client
        client = ProhashingClient(api_key=args.api_key)
        
        # Fetch data based on the specified type
        data = {}
        
        if args.type == "profitability" or args.type == "all":
            logger.info("Fetching algorithm profitability data")
            data["profitability"] = client.get_all_algorithm_profitability()
        
        if (args.type == "worker" or args.type == "all") and args.worker_id:
            logger.info(f"Fetching worker performance data for worker {args.worker_id}")
            data["worker_performance"] = client.get_worker_pool_performance(
                worker_id=args.worker_id,
                algorithm=args.algorithm
            )
        
        # Output the data
        if args.output:
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Data saved to {args.output}")
        else:
            print(json.dumps(data, indent=2))
        
        logger.info("Done")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
