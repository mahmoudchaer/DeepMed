#!/usr/bin/env python3

import requests
import json
import time
import argparse
import concurrent.futures
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import socket
from urllib.parse import urlparse

# Default services with health endpoints
DEFAULT_SERVICES = {
    "Tabular Data Services": {
        "Data Cleaner": {"url": "http://localhost:5001", "endpoint": "/health"},
        "Feature Selector": {"url": "http://localhost:5002", "endpoint": "/health"},
        "Anomaly Detector": {"url": "http://localhost:5003", "endpoint": "/health"},
        "Model Trainer": {"url": "http://localhost:5004", "endpoint": "/health"},
        "Medical Assistant": {"url": "http://localhost:5005", "endpoint": "/health"},
        "Model Coordinator": {"url": "http://localhost:5020", "endpoint": "/health"},
        "Logistic Regression": {"url": "http://localhost:5010", "endpoint": "/health"},
        "Decision Tree": {"url": "http://localhost:5011", "endpoint": "/health"},
        "Random Forest": {"url": "http://localhost:5012", "endpoint": "/health"},
        "SVM": {"url": "http://localhost:5013", "endpoint": "/health"},
        "KNN": {"url": "http://localhost:5014", "endpoint": "/health"},
        "Naive Bayes": {"url": "http://localhost:5015", "endpoint": "/health"},
        "Tabular Predictor": {"url": "http://localhost:5101", "endpoint": "/health"}
    },
    "Image Processing Services": {
        "Pipeline Service": {"url": "http://localhost:6001", "endpoint": "/health"},
        "Augmentation Service": {"url": "http://localhost:6002", "endpoint": "/health"},
        "Image Classification": {"url": "http://localhost:6003", "endpoint": "/health"},
        "Anomaly Detection Service": {"url": "http://localhost:6004", "endpoint": "/health"},
        "Image Predictor": {"url": "http://localhost:6005", "endpoint": "/health"}
    },
    "Chatbot Services": {
        "Embedding Service": {"url": "http://localhost:5201", "endpoint": "/health"},
        "LLM Generator": {"url": "http://localhost:5202", "endpoint": "/health"},
        "Vector Search": {"url": "http://localhost:5203", "endpoint": "/health"},
        "Chatbot Gateway": {"url": "http://localhost:5200", "endpoint": "/health"}
    },
    "Monitoring": {
        "Monitoring Service": {"url": "http://localhost:3000", "endpoint": "/health"}
    }
}

def load_services_config():
    """Load services configuration from environment or file"""
    # First check if custom config file is provided
    config_file = os.environ.get('HEALTH_CHECK_CONFIG')
    if config_file and os.path.isfile(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            return DEFAULT_SERVICES
    
    # Check if host override is provided
    host_override = os.environ.get('HEALTH_CHECK_HOST')
    if host_override:
        services = DEFAULT_SERVICES.copy()
        # Update all URLs with the new host
        for category in services:
            for service_name in services[category]:
                url = services[category][service_name]["url"]
                # Replace localhost with the override host
                new_url = url.replace("localhost", host_override)
                services[category][service_name]["url"] = new_url
        return services
    
    return DEFAULT_SERVICES

def resolve_service_hostname(hostname):
    """Try to resolve hostname to check if it exists in DNS"""
    try:
        socket.gethostbyname(hostname)
        return True
    except socket.gaierror:
        return False

def check_service_health(category, service_name, service_info, timeout=2):
    """Check health of a specific service, trying multiple hostnames if needed"""
    url = service_info["url"]
    endpoint = service_info.get("endpoint", "/health")
    full_url = f"{url}{endpoint}"
    result = {
        "service": service_name,
        "category": category,
        "url": full_url,
        "status": "unknown",
        "response_time": None,
        "details": {},
        "error": None
    }
    start_time = time.time()
    response = None
    response_time = None
    # Try the primary URL first
    try:
        response = requests.get(full_url, timeout=timeout)
        response_time = time.time() - start_time
        if response.status_code == 200:
            try:
                health_data = response.json()
                result["details"] = health_data
                result["status"] = health_data.get("status", "healthy")
            except json.JSONDecodeError:
                result["status"] = "invalid"
                result["error"] = "Invalid JSON response"
        else:
            result["status"] = "error"
            result["error"] = f"HTTP Error: {response.status_code}"
        result["response_time"] = response_time
        return result
    except Exception as e:
        last_error = str(e)
    # If the primary URL fails, try alternatives
    # Build alternative hostnames
    parsed = urlparse(url)
    port = parsed.port
    service_base_name = service_name.lower().replace(' ', '_').replace('-', '_')
    alt_hosts = [
        service_base_name,
        service_name.lower().replace(' ', '-').replace('_', '-'),
        'localhost',
        '127.0.0.1',
        'host.docker.internal',
        'docker.for.win.localhost',
        'docker.for.mac.localhost',
        f"deepmed_{service_base_name}",
        f"deepmed-{service_name.lower().replace(' ', '-').replace('_', '-')}"
    ]
    # Add external IP from env or config
    external_ip = os.getenv('EXTERNAL_IP', '20.119.81.37')
    alt_hosts.append(external_ip)
    # Try DNS-discovered hosts first
    discovered_hosts = []
    for host_candidate in [
        service_base_name,
        f"deepmed_{service_base_name}",
        f"deepmed-{service_base_name}",
        f"{service_base_name}_service",
        f"deepmed_{service_base_name}_service"
    ]:
        if resolve_service_hostname(host_candidate):
            discovered_hosts.append(host_candidate)
    alt_hosts = discovered_hosts + alt_hosts
    # Try each alternative
    for alt_host in alt_hosts:
        alt_url = f"http://{alt_host}:{port}{endpoint}"
        try:
            alt_start = time.time()
            response = requests.get(alt_url, timeout=timeout)
            response_time = time.time() - alt_start
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    result["details"] = health_data
                    result["status"] = health_data.get("status", "healthy")
                except json.JSONDecodeError:
                    result["status"] = "invalid"
                    result["error"] = "Invalid JSON response"
                result["url"] = alt_url
                result["response_time"] = response_time
                # Update the service_info URL for future checks
                service_info["url"] = f"http://{alt_host}:{port}"
                return result
            else:
                last_error = f"HTTP Error: {response.status_code}"
        except Exception as e:
            last_error = str(e)
            continue
    # If all attempts fail
    result["status"] = "timeout"
    result["error"] = f"Connection timeout ({last_error})"
    result["response_time"] = None
    return result

def check_all_services(services, max_workers=10, timeout=2, verbose=False, github_actions=False):
    """Check health of all registered services in parallel"""
    console = Console()
    results = []
    
    with Progress(disable=github_actions) as progress:
        task = progress.add_task("[cyan]Checking services...", total=sum(len(svcs) for svcs in services.values()))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_service = {}
            
            for category, svcs in services.items():
                for service_name, service_info in svcs.items():
                    future = executor.submit(check_service_health, category, service_name, service_info, timeout)
                    future_to_service[future] = (category, service_name)
            
            for future in concurrent.futures.as_completed(future_to_service):
                category, service_name = future_to_service[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if verbose or github_actions:
                        status = result["status"]
                        status_color = {
                            "healthy": "green",
                            "degraded": "yellow",
                            "unhealthy": "red",
                            "error": "red",
                            "timeout": "red",
                            "unreachable": "red",
                            "unknown": "yellow",
                            "invalid": "red"
                        }.get(status, "white")
                        
                        if github_actions:
                            # GitHub Actions output format
                            if status != "healthy":
                                print(f"::error::{service_name} - {result['url']} - Status: {status}" + 
                                      (f" - Error: {result['error']}" if result["error"] else ""))
                            else:
                                print(f"{service_name} - {result['url']} - Status: {status}")
                        else:
                            # Regular rich console output
                            console.print(f"[{status_color}]{status}[/{status_color}] - {category} - {service_name} - {result['url']}")
                            
                            if status != "healthy" and result["error"]:
                                console.print(f"  Error: {result['error']}", style="red")
                except Exception as e:
                    error_msg = f"Error checking {service_name}: {str(e)}"
                    if github_actions:
                        print(f"::error::{error_msg}")
                    else:
                        console.print(error_msg, style="red")
                
                progress.update(task, advance=1)
    
    return results

def display_results(results, github_actions=False):
    """Display results in a nicely formatted table or GitHub Actions friendly format"""
    console = Console()
    
    # Group results by category
    results_by_category = {}
    for result in results:
        category = result["category"]
        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append(result)
    
    # Display summary
    healthy_count = sum(1 for r in results if r["status"] == "healthy")
    total_count = len(results)
    
    if github_actions:
        print(f"::group::Health Check Summary")
        print(f"Health status: {healthy_count}/{total_count} services healthy")
        unhealthy_services = [f"{r['service']} ({r['status']})" for r in results if r["status"] != "healthy"]
        if unhealthy_services:
            print(f"Unhealthy services: {', '.join(unhealthy_services)}")
        print(f"::endgroup::")
        
        # Output summary for each category
        for category, category_results in results_by_category.items():
            print(f"::group::{category}")
            for result in category_results:
                status = result["status"]
                response_time = f"{result['response_time']*1000:.2f}ms" if result["response_time"] else "N/A"
                details = ""
                if result["details"]:
                    details = ", ".join(f"{k}: {v}" for k, v in result["details"].items() if k != "status")
                elif result["error"]:
                    details = f"Error: {result['error']}"
                
                print(f"{result['service']}: {status} - {response_time} - {details}")
            print(f"::endgroup::")
    else:
        # Regular rich console output
        console.print(f"\n[bold]Health Check Summary:[/bold] {healthy_count}/{total_count} services healthy")
        
        # Display tables by category
        for category, category_results in results_by_category.items():
            console.print(f"\n[bold]{category}[/bold]")
            
            table = Table(show_header=True)
            table.add_column("Service", style="cyan")
            table.add_column("Status", style="cyan")
            table.add_column("Response Time", style="cyan")
            table.add_column("Details", style="cyan")
            
            for result in category_results:
                service_name = result["service"]
                status = result["status"]
                response_time = f"{result['response_time']*1000:.2f}ms" if result["response_time"] else "N/A"
                
                details = ""
                if result["details"]:
                    details = ", ".join(f"{k}: {v}" for k, v in result["details"].items() if k != "status")
                elif result["error"]:
                    details = result["error"]
                
                status_style = {
                    "healthy": "green",
                    "degraded": "yellow",
                    "unhealthy": "red",
                    "error": "red",
                    "timeout": "red",
                    "unreachable": "red",
                    "unknown": "yellow",
                    "invalid": "red"
                }.get(status, "white")
                
                table.add_row(
                    service_name,
                    f"[{status_style}]{status}[/{status_style}]",
                    response_time,
                    details
                )
            
            console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Health check for Docker services")
    parser.add_argument("--timeout", type=int, default=2, help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=10, help="Maximum number of worker threads")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--config", help="Path to services configuration JSON file")
    parser.add_argument("--host", help="Host override (replaces localhost in service URLs)")
    parser.add_argument("--github-actions", action="store_true", help="Format output for GitHub Actions")
    args = parser.parse_args()
    
    # Set environment variables from arguments if provided
    if args.config:
        os.environ['HEALTH_CHECK_CONFIG'] = args.config
    if args.host:
        os.environ['HEALTH_CHECK_HOST'] = args.host
    
    # Load services configuration
    services = load_services_config()
    
    # Skip intro output in GitHub Actions mode
    if not args.github_actions:
        console = Console()
        console.print("[bold]DeepMed Docker Services Health Check[/bold]")
        console.print("Checking all services, please wait...\n")
    
    results = check_all_services(
        services=services,
        max_workers=args.workers,
        timeout=args.timeout,
        verbose=args.verbose,
        github_actions=args.github_actions
    )
    
    display_results(results, github_actions=args.github_actions)
    
    # Save results to JSON file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        if not args.github_actions:
            console = Console()
            console.print(f"\nResults saved to {args.output}")
    
    # Return non-zero exit code if any service is unhealthy
    unhealthy_services = sum(1 for r in results if r["status"] != "healthy")
    return 1 if unhealthy_services > 0 else 0

if __name__ == "__main__":
    exit(main()) 