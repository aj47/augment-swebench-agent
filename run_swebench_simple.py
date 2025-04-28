#!/usr/bin/env python3
"""
Simplified script to run SWEBench problems on macOS.

This script uses subprocess to run Docker commands directly instead of using the Python Docker library.
"""

import os
import sys
import logging
import subprocess
import json
import argparse
from pathlib import Path
import time
import uuid
import platform

from rich.console import Console
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('swebench_simple.log')
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, check=True):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"

def get_issue_image_name(problem_id: str) -> str:
    """Fetch a docker image for the issue."""
    issue_key = problem_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{issue_key}:latest"

def run_swebench_problem(problem_id, problem_statement, workspace_path):
    """Run a SWEBench problem using Docker CLI commands."""
    console = Console()
    logs_prefix = f"[bold blue]{problem_id}[/bold blue]"

    # Create workspace directory
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Get the Docker image name
    image_name = get_issue_image_name(problem_id)
    console.print(f"{logs_prefix} Using Docker image: {image_name}")

    # Pull the Docker image
    console.print(f"{logs_prefix} Pulling Docker image...")
    success, output = run_command(f"docker pull {image_name}")
    if not success:
        console.print(f"{logs_prefix} [bold red]Failed to pull Docker image: {output}[/bold red]")
        return False

    # Generate a unique container name
    container_name = f"sweb.augment.{problem_id}_{uuid.uuid4().hex[:8]}"

    try:
        # Run the Docker container
        console.print(f"{logs_prefix} Starting Docker container...")
        success, output = run_command(
            f'docker run --name {container_name} -d '
            f'-v /testbed '
            f'{image_name} '
            f'bash -c "git config --global user.email a && git config --global user.name a && '
            f'git config --global --add safe.directory /testbed && git commit --allow-empty -am augment && sleep 7200"'
        )
        if not success:
            console.print(f"{logs_prefix} [bold red]Failed to start Docker container: {output}[/bold red]")
            return False

        container_id = output.strip()
        console.print(f"{logs_prefix} Docker container started with ID: {container_id}")

        # Wait for the container to start
        time.sleep(5)

        # Get the volume path
        success, output = run_command(f"docker inspect --format='{{{{.Mounts}}}}' {container_id}")
        if not success:
            console.print(f"{logs_prefix} [bold red]Failed to get volume info: {output}[/bold red]")
            return False

        # Parse the volume path from the output
        volume_info = output.strip()
        volume_parts = volume_info.split()
        if len(volume_parts) < 3:
            console.print(f"{logs_prefix} [bold red]Failed to parse volume path from: {volume_info}[/bold red]")
            return False

        volume_path = Path(volume_parts[2])
        console.print(f"{logs_prefix} Docker volume path: {volume_path}")

        # Create a symlink to the volume
        problem_link = workspace_path / problem_id
        problem_link.unlink(missing_ok=True)
        problem_link.symlink_to(volume_path)
        console.print(f"{logs_prefix} Created symlink from {problem_link} to {volume_path}")

        # Set permissions on the volume using Docker exec
        console.print(f"{logs_prefix} Setting permissions on volume...")
        # Wait a bit to make sure the container is fully started
        time.sleep(5)
        # Check if the container is running
        success, output = run_command(f'docker inspect --format="{{{{.State.Running}}}}" {container_id}')
        if success and output.strip().lower() == "true":
            run_command(f"docker exec {container_id} bash -c 'chmod -R a+rwx /testbed'", check=False)
        else:
            console.print(f"{logs_prefix} [bold yellow]Warning: Container is not running, skipping permission setting[/bold yellow]")

        # Write the problem statement to a file
        with open(workspace_path / "problem_statement.txt", "w") as f:
            f.write(problem_statement)

        console.print(f"{logs_prefix} Problem setup complete. You can now work with the problem at {workspace_path}")
        console.print(f"{logs_prefix} To access the Docker container, run: docker exec -it {container_id} bash")

        return True
    except Exception as e:
        console.print(f"{logs_prefix} [bold red]Error: {str(e)}[/bold red]")
        logger.exception(f"Error processing problem {problem_id}")
        return False
    finally:
        # Note: We don't stop the container here so the user can work with it
        pass

def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run SWEBench problems on macOS")
    parser.add_argument(
        "--problem-ids",
        nargs="+",
        required=True,
        help="Specific problem IDs to run on",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Custom workspace directory (default: /tmp/swebench_workspace/UUID)",
    )

    args = parser.parse_args()

    # Initialize console
    console = Console()

    # Check if we're running on macOS
    if platform.system() != "Darwin":
        console.print("[bold red]This script is designed for macOS. Please use run_agent_on_swebench_problem.py on other platforms.[/bold red]")
        return

    # Load the SWE-bench dataset
    console.print("Loading SWE-bench dataset...")
    swebench_dataset = load_dataset("princeton-nlp/SWE-bench_Verified")[  # pyright: ignore[reportIndexIssue]
        "test"
    ].to_pandas()  # pyright: ignore

    # Filter by problem IDs
    examples = swebench_dataset[swebench_dataset["instance_id"].isin(args.problem_ids)]
    num_examples = len(examples)
    console.print(f"Running on {num_examples} specific problems selected by ID.")
    if num_examples == 0:
        console.print("[bold red]No problems found with the specified IDs.[/bold red]")
        return

    # Create workspace directory
    if args.workspace:
        workspace_base_path = Path(args.workspace).resolve()
    else:
        workspace_base_path = Path(f"/tmp/swebench_workspace/{uuid.uuid4().hex[:8]}").resolve()
    console.print(f"Workspace base path: {workspace_base_path}")
    workspace_base_path.mkdir(parents=True, exist_ok=True)

    # Iterate over the examples
    for i, problem in enumerate(examples.itertuples()):
        problem_id = problem.instance_id
        problem_statement = problem.problem_statement

        console.print(f"\nProcessing example {i + 1}/{len(examples)}")

        # Create a directory for this problem
        problem_workspace = workspace_base_path / problem_id

        # Run the problem
        success = run_swebench_problem(problem_id, problem_statement, problem_workspace)

        if success:
            console.print(f"[bold green]Successfully set up problem {problem_id}[/bold green]")
        else:
            console.print(f"[bold red]Failed to set up problem {problem_id}[/bold red]")

    console.print("\nAll problems processed.")
    console.print(f"Workspace directory: {workspace_base_path}")
    console.print("To work with a problem, navigate to its directory and use the Docker container.")

if __name__ == "__main__":
    main()
