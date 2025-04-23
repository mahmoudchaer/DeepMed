#!/usr/bin/env python3

import json
import argparse
import os
import sys

# Define environment-specific host mappings
ENV_HOSTS = {
    "dev": "localhost",
    "staging": "staging-api.deepmed.ai",
    "prod": "api.deepmed.ai"
}

def generate_config(env, output_file):
    """Generate a service configuration file for a specific environment"""
    if env not in ENV_HOSTS:
        print(f"Error: Unknown environment '{env}'. Must be one of: {', '.join(ENV_HOSTS.keys())}")
        return False
    
    host = ENV_HOSTS[env]
    
    # Determine template path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "../config/services-template.json")
    
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        return False
    
    try:
        # Read template
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Replace HOST placeholder with actual host
        config_json = template.replace("HOST", host)
        
        # Parse to ensure it's valid JSON
        config = json.loads(config_json)
        
        # Environment-specific customizations
        if env == "staging":
            # Example: disable certain services in staging
            if "Experimental Services" in config:
                del config["Experimental Services"]
        
        elif env == "prod":
            # Example: production might have different port mappings 
            # or additional security requirements
            pass
        
        # Write to output file
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Generated configuration for {env} environment at {output_file}")
        return True
    
    except Exception as e:
        print(f"Error generating config: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate service configuration for different environments")
    parser.add_argument("--env", required=True, choices=ENV_HOSTS.keys(), help="Environment to generate config for")
    parser.add_argument("--output", required=True, help="Output file path")
    args = parser.parse_args()
    
    success = generate_config(args.env, args.output)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 