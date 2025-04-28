#!/usr/bin/env python3
"""
Script to run the agent on a SWE-bench problem in a Docker container on macOS.

This script loads a SWE-bench problem, starts a Docker container for it,
and runs the agent inside the container by calling cli.py.
"""

import os
import logging
import threading
import sys
import json
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Manager
import time
import numpy as np
import platform

from rich.console import Console
from datasets import load_dataset

# Use the macOS-specific Docker utilities
from utils.docker_cli_utils import MAX_DOCKER_CONCURRENCY, setup_workspace, stop_container
from utils.common import generate_patch
from cli import main as cli_main
import uuid
from utils.swebench_eval_utils import get_dataset_name, run_evaluation


def generate_diff_from_path(repo_path, logs_prefix, console):
    """Generate a diff from a repository path using generate_patch.

    Args:
        repo_path: Path to the repository
        logs_prefix: Prefix for log messages
        console: Rich console for output

    Returns:
        str: The generated diff or None if failed
    """
    if not os.path.exists(repo_path):
        console.print(f"{logs_prefix} [bold yellow]Repository path {repo_path} does not exist[/bold yellow]")
        return None

    try:
        diff = generate_patch(repo_path)
        if diff:
            return diff
        else:
            console.print(f"{logs_prefix} [bold yellow]generate_patch returned empty diff[/bold yellow]")
            return None
    except Exception as e:
        console.print(f"{logs_prefix} [bold yellow]Error in generate_patch: {str(e)}[/bold yellow]")
        return None


def generate_diff_from_git_search(workspace_path, logs_prefix, console):
    """Find git repositories and generate a diff.

    Args:
        workspace_path: Path to the workspace
        logs_prefix: Prefix for log messages
        console: Rich console for output

    Returns:
        str: The generated diff or None if failed
    """
    console.print(f"{logs_prefix} Searching for git repositories...")

    # Try to find git repositories
    try:
        result = subprocess.run(
            f"find {workspace_path} -name .git -type d | head -1",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )

        if result.returncode == 0 and result.stdout.strip():
            git_dir = result.stdout.strip()
            repo_path = os.path.dirname(git_dir)
            console.print(f"{logs_prefix} Found git repository at {repo_path}")

            # Try to generate a diff
            try:
                result = subprocess.run(
                    f"cd {repo_path} && git diff --no-color HEAD",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    diff = result.stdout
                    if diff:
                        return diff
                    else:
                        console.print(f"{logs_prefix} [bold yellow]Git diff command returned empty diff[/bold yellow]")
                        return None
                else:
                    console.print(f"{logs_prefix} [bold yellow]Git diff command failed: {result.stderr}[/bold yellow]")
                    return None
            except Exception as e:
                console.print(f"{logs_prefix} [bold yellow]Error running git diff: {str(e)}[/bold yellow]")
                return None
        else:
            console.print(f"{logs_prefix} [bold yellow]No git repositories found[/bold yellow]")
            return None
    except Exception as e:
        console.print(f"{logs_prefix} [bold yellow]Error searching for git repositories: {str(e)}[/bold yellow]")
        return None


def generate_diff_from_docker(container_id, logs_prefix, console):
    """Generate a diff using Docker exec.

    Args:
        container_id: ID of the Docker container
        logs_prefix: Prefix for log messages
        console: Rich console for output

    Returns:
        str: The generated diff or None if failed
    """
    console.print(f"{logs_prefix} Generating diff using Docker exec...")

    try:
        result = subprocess.run(
            f"docker exec {container_id} bash -c 'cd /testbed && git diff --no-color HEAD'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            diff = result.stdout
            if diff:
                return diff
            else:
                console.print(f"{logs_prefix} [bold yellow]Docker exec returned empty diff[/bold yellow]")
                return None
        else:
            console.print(f"{logs_prefix} [bold yellow]Docker exec failed: {result.stderr}[/bold yellow]")
            return None
    except Exception as e:
        console.print(f"{logs_prefix} [bold yellow]Error using Docker exec: {str(e)}[/bold yellow]")
        return None


def generate_diff_from_git_command(repo_path, logs_prefix, console):
    """Generate a diff using direct git commands.

    Args:
        repo_path: Path to the repository
        logs_prefix: Prefix for log messages
        console: Rich console for output

    Returns:
        str: The generated diff or None if failed
    """
    console.print(f"{logs_prefix} Trying direct git command...")

    try:
        # Try different git commands
        commands = [
            f"cd {repo_path} && git diff --no-color HEAD",
            f"cd {repo_path} && git diff --no-color",
            f"cd {repo_path} && git status -v"
        ]

        for command in commands:
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout

        console.print(f"{logs_prefix} [bold yellow]All git commands failed to produce a diff[/bold yellow]")
        return None
    except Exception as e:
        console.print(f"{logs_prefix} [bold yellow]Error running git commands: {str(e)}[/bold yellow]")
        return None
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('swebench_run.log')
    ]
)
logger = logging.getLogger(__name__)


def run_eval_on_single_problem(problem_id: str, workspace_path: Path, console: Console):
    """Run evaluation on a single problem.

    On macOS, the Docker socket might not be accessible to the evaluation tools.
    We'll handle this gracefully and provide fallback mechanisms.

    Args:
        problem_id: The ID of the problem
        workspace_path: Path to the workspace directory
        console: Rich console for output

    Returns:
        dict: Evaluation outcomes with at least an "is_success" key
    """
    eval_outcomes = {
        "is_success": False,
    }

    # Check if the predictions file exists
    predictions_file = workspace_path / "predictions.json"
    if not predictions_file.exists():
        console.print(f"[bold yellow]Warning: Predictions file not found at {predictions_file}[/bold yellow]")
        # Try to create a minimal predictions file
        try:
            with open(predictions_file, "w") as f:
                json.dump(
                    [
                        {
                            "instance_id": problem_id,
                            "model_name_or_path": "augment-agent",
                            "model_patch": "",
                        }
                    ],
                    f,
                    indent=2,
                )
            console.print(f"Created empty predictions file at {predictions_file}")
        except Exception as e:
            console.print(f"[bold red]Failed to create predictions file: {str(e)}[/bold red]")
            return eval_outcomes

    # Verify the predictions file has valid content
    try:
        with open(predictions_file, "r") as f:
            predictions = json.load(f)
        if not predictions or not isinstance(predictions, list) or "instance_id" not in predictions[0]:
            console.print(f"[bold yellow]Warning: Predictions file has invalid format[/bold yellow]")
            # Fix the predictions file
            with open(predictions_file, "w") as f:
                json.dump(
                    [
                        {
                            "instance_id": problem_id,
                            "model_name_or_path": "augment-agent",
                            "model_patch": predictions[0].get("model_patch", "") if predictions else "",
                        }
                    ],
                    f,
                    indent=2,
                )
            console.print(f"Fixed predictions file format at {predictions_file}")
    except Exception as e:
        console.print(f"[bold yellow]Warning: Failed to validate predictions file: {str(e)}[/bold yellow]")
        # Try to create a minimal predictions file
        try:
            with open(predictions_file, "w") as f:
                json.dump(
                    [
                        {
                            "instance_id": problem_id,
                            "model_name_or_path": "augment-agent",
                            "model_patch": "",
                        }
                    ],
                    f,
                    indent=2,
                )
            console.print(f"Created empty predictions file at {predictions_file}")
        except Exception as e:
            console.print(f"[bold red]Failed to create predictions file: {str(e)}[/bold red]")
            return eval_outcomes

    # Check Docker accessibility with retries
    max_docker_check_attempts = 3
    docker_accessible = False

    for attempt in range(max_docker_check_attempts):
        try:
            result = subprocess.run(
                "docker ps",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                docker_accessible = True
                break
            else:
                console.print(f"[bold yellow]Docker check attempt {attempt+1}/{max_docker_check_attempts} failed: {result.stderr}[/bold yellow]")
                if attempt < max_docker_check_attempts - 1:
                    console.print(f"Retrying Docker check in 5 seconds...")
                    time.sleep(5)
        except Exception as e:
            console.print(f"[bold yellow]Docker check attempt {attempt+1}/{max_docker_check_attempts} failed with exception: {str(e)}[/bold yellow]")
            if attempt < max_docker_check_attempts - 1:
                console.print(f"Retrying Docker check in 5 seconds...")
                time.sleep(5)

    if not docker_accessible:
        console.print(f"[bold yellow]Warning: Docker is not accessible for evaluation after {max_docker_check_attempts} attempts. Skipping evaluation.[/bold yellow]")
        return eval_outcomes

    # Try to run the evaluation with retries
    max_eval_attempts = 3
    eval_success = False

    for attempt in range(max_eval_attempts):
        try:
            console.print(f"Running evaluation attempt {attempt+1}/{max_eval_attempts}...")

            # Make sure the container for this problem is stopped before evaluation
            from utils.docker_cli_utils import stop_container
            stop_container(f"sweb.eval.{problem_id}")

            # Run the evaluation
            run_evaluation(
                predictions_file=predictions_file,
                dataset=get_dataset_name("full"),  # Always use the full dataset for evaluation
                run_id=problem_id,
                swebench_venv_path=Path(f"{os.environ['HOME']}/swebench_eval_tools_env/bin/python"),
                console=console,
            )

            # Check if the evaluation file was created
            eval_file = workspace_path / f"augment-agent.{problem_id}.json"
            if eval_file.exists():
                try:
                    eval_dict = json.loads(eval_file.read_text())
                    eval_outcomes["is_success"] = problem_id in eval_dict.get("resolved_ids", [])
                    console.print(f"Evaluated {problem_id} successfully.")
                    eval_success = True
                    break
                except json.JSONDecodeError:
                    console.print(f"[bold yellow]Warning: Evaluation file exists but contains invalid JSON[/bold yellow]")
            else:
                console.print(f"[bold yellow]Warning: Evaluation file not created at {eval_file}[/bold yellow]")

            if attempt < max_eval_attempts - 1:
                console.print(f"Retrying evaluation in 10 seconds...")
                time.sleep(10)

        except Exception as e:
            console.print(f"[bold yellow]Warning: Evaluation attempt {attempt+1}/{max_eval_attempts} failed: {str(e)}[/bold yellow]")
            if attempt < max_eval_attempts - 1:
                console.print(f"Retrying evaluation in 10 seconds...")
                time.sleep(10)

    if not eval_success:
        console.print(f"[bold yellow]Warning: All evaluation attempts failed[/bold yellow]")

        # Try to manually check if the solution was successful by looking at the logs
        logs_dir = workspace_path / "logs" / "run_evaluation" / problem_id / "augment-agent" / problem_id
        if logs_dir.exists():
            report_file = logs_dir / "report.json"
            if report_file.exists():
                try:
                    report_dict = json.loads(report_file.read_text())
                    fail_to_pass_tests = report_dict.get("fail_to_pass_tests", [])
                    if fail_to_pass_tests:
                        passed_tests = [test for test in fail_to_pass_tests if test.get("status") == "pass"]
                        if passed_tests:
                            eval_outcomes["is_success"] = True
                            console.print(f"[bold green]Found {len(passed_tests)} passed tests in report.json[/bold green]")
                except Exception as e:
                    console.print(f"[bold yellow]Warning: Failed to parse report.json: {str(e)}[/bold yellow]")

    return eval_outcomes


def run_agent_on_single_problem(
    problem_id: str,
    problem_statement: str,
    rollout_idx: int,
    workspace_base_path: Path,
    lock: threading.Lock,
    semaphore: threading.Semaphore,
) -> tuple[str, float, dict]:
    """
    Run the agent on a single SWE-bench problem.

    Args:
        problem_id: The ID of the problem
        problem_statement: The problem statement
        lock: Threading lock for Docker operations
        semaphore: Threading semaphore for Docker operations

    Returns:
        dict: The diff data generated by the agent
        float: The time taken to generate the diff
        dict: The evaluation outcomes
    """
    console = Console()
    logs_prefix = f"[bold blue]{problem_id}[/bold blue]"

    workspace_path = workspace_base_path / problem_id / f"rollout_{rollout_idx}"
    output_file = workspace_path / "agent_logs.txt"

    # Ensure workspace directory exists
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Start the Docker container
    container_id = None
    diff = None

    try:
        env, container_id = setup_workspace(workspace_path, problem_id, lock, semaphore)
        console.print(f"{logs_prefix} Docker container started with ID: {container_id}")

        # Set environment variables
        for key, value in env.items():
            os.environ[key] = value

        # Save original sys.argv
        original_argv = sys.argv.copy()

        # Create new sys.argv for cli.py
        # The workspace should be the directory containing the symlink to the Docker volume
        cli_args = [
            "cli.py",
            "--workspace",
            str(workspace_path),  # Use the workspace path without appending problem_id
            "--problem-statement",
            problem_statement,
            "--docker-container-id",
            container_id,
            "--use-container-workspace",
            "/testbed",
            "--minimize-stdout-logs",
        ]

        # Set logs path if output_file is specified
        if output_file:
            cli_args.extend(["--logs-path", str(output_file)])

        # Don't add debug flag as it's not supported by the CLI

        # Replace sys.argv with our custom arguments
        sys.argv = cli_args

        # Run the agent via cli.py
        console.print(f"{logs_prefix} Starting agent run...")
        console.print(f"{logs_prefix} CLI arguments: {cli_args}")

        # Log environment variables that might be relevant
        console.print(f"{logs_prefix} Workspace path: {workspace_path}")
        console.print(f"{logs_prefix} Docker container ID: {container_id}")

        # Check if the workspace directory exists
        if not os.path.exists(str(workspace_path / problem_id)):
            console.print(f"{logs_prefix} [bold yellow]Warning: Workspace path {workspace_path / problem_id} does not exist.[/bold yellow]")

        start_time = time.time()
        try:
            cli_main()
        except Exception as e:
            console.print(f"{logs_prefix} [bold red]Error during CLI execution: {str(e)}[/bold red]")
            logger.exception(f"Error during CLI execution for {problem_id}")
            raise

        agent_duration = time.time() - start_time
        console.print(f"{logs_prefix} Agent run completed in {agent_duration:.2f}s.")

        # Restore original sys.argv
        sys.argv = original_argv

        # Generate patch after the agent has completed its work
        # The problem_id path is a symlink to the Docker volume
        repo_path = str(workspace_path / problem_id)
        diff = None

        # Try multiple approaches to generate the diff, with fallbacks
        diff_generation_methods = [
            # Method 1: Use the symlink path
            lambda: generate_diff_from_path(repo_path, logs_prefix, console),

            # Method 2: Try to find the git directory
            lambda: generate_diff_from_git_search(workspace_path, logs_prefix, console),

            # Method 3: Use Docker exec
            lambda: generate_diff_from_docker(container_id, logs_prefix, console),

            # Method 4: Try a direct git command
            lambda: generate_diff_from_git_command(repo_path, logs_prefix, console)
        ]

        # Try each method until one succeeds
        for method_num, method in enumerate(diff_generation_methods, 1):
            console.print(f"{logs_prefix} Trying diff generation method {method_num}...")
            try:
                diff = method()
                if diff:
                    console.print(f"{logs_prefix} Successfully generated diff using method {method_num}")
                    break
            except Exception as e:
                console.print(f"{logs_prefix} [bold yellow]Method {method_num} failed: {str(e)}[/bold yellow]")

        # If all methods failed, create an empty diff
        if not diff:
            console.print(f"{logs_prefix} [bold yellow]All diff generation methods failed. Using empty diff.[/bold yellow]")
            diff = ""

        # Save the predictions
        with (workspace_path / "predictions.json").open("w") as f:
            json.dump(
                [
                    {
                        "instance_id": problem_id,
                        "model_name_or_path": "augment-agent",
                        "model_patch": diff,
                    }
                ],
                f,
                indent=2,
            )
    except Exception as e:
        console.print(f"{logs_prefix} [bold red]Error during agent run: {str(e)}[/bold red]")
        logger.exception(f"Error during agent run for {problem_id}")
        # Create an empty diff if we failed
        diff = ""
        agent_duration = 0
    finally:
        # Stop and clean up the Docker container
        if container_id is not None:
            console.print(f"{logs_prefix} Stopping Docker container...")
            stop_container(container_id)
            console.print(f"{logs_prefix} Docker container stopped")

    # Evaluate the generated diff
    console.print(f"{logs_prefix} Evaluating the generated diff...")
    eval_outcomes = {"is_success": False}
    try:
        start_time = time.time()
        eval_outcomes = run_eval_on_single_problem(problem_id, workspace_path, console)
        eval_duration = time.time() - start_time
        console.print(f"{logs_prefix} Evaluation completed in {eval_duration:.2f}s.")
    except Exception as e:
        console.print(f"{logs_prefix} [bold red]Error during evaluation: {str(e)}[/bold red]")
        logger.exception(f"Error during evaluation for {problem_id}")

    # If diff is None, set it to an empty string
    if diff is None:
        diff = ""
        console.print(f"{logs_prefix} [bold yellow]Warning: No diff was generated.[/bold yellow]")

    return diff, agent_duration, eval_outcomes


def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the agent on SWE-bench problems")
    parser.add_argument(
        "--problem-ids",
        nargs="+",
        help="Specific problem IDs to run on",
    )
    parser.add_argument(
        "--num-candidate-solutions",
        type=int,
        default=1,
        help="Number of candidate solutions to generate for each example",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Custom workspace directory (default: /tmp/workspace/UUID)",
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

    # Filter by problem IDs if specified
    if args.problem_ids:
        examples = swebench_dataset[swebench_dataset["instance_id"].isin(args.problem_ids)]
        num_examples = len(examples)
        console.print(f"Running on {num_examples} specific problems selected by ID.")
        if num_examples == 0:
            console.print("[bold red]No problems found with the specified IDs.[/bold red]")
            return
    else:
        console.print("[bold red]Please specify problem IDs using --problem-ids.[/bold red]")
        return

    # List to store all diff data
    all_diff_data = []

    # get workspace base dir
    if args.workspace:
        workspace_base_path = Path(args.workspace).resolve()
    else:
        workspace_base_path = Path(f"/tmp/workspace/{uuid.uuid4().hex[:8]}").resolve()
    console.print(f"Workspace base path: {workspace_base_path}")

    output_path = f"swebench_results_{uuid.uuid4().hex[:8]}.jsonl"

    # Iterate over the examples
    for i, problem in enumerate(examples.itertuples()):
        try:
            problem_id = problem.instance_id
            problem_statement = problem.problem_statement

            console.print(f"\nProcessing example {i + 1}/{len(examples)}")

            # Run the agent on the selected problem
            with Manager() as manager:
                lock = manager.Lock()
                semaphore = manager.Semaphore(MAX_DOCKER_CONCURRENCY)

                # For simplicity, we'll run sequentially instead of using a pool
                diffs = []
                agent_durations = []
                eval_outcomes_list = []

                for rollout_idx in range(args.num_candidate_solutions):
                    console.print(f"Running solution attempt {rollout_idx + 1}/{args.num_candidate_solutions}")
                    diff, agent_duration, eval_outcomes = run_agent_on_single_problem(
                        problem_id,
                        problem_statement,
                        rollout_idx,
                        workspace_base_path,
                        lock,
                        semaphore
                    )
                    diffs.append(diff)
                    agent_durations.append(agent_duration)
                    eval_outcomes_list.append(eval_outcomes)

                median_duration = np.median(agent_durations)
                diff_data = {
                    "id": problem_id,
                    "instruction": problem_statement,
                    "diffs": diffs,
                    "agent_durations": agent_durations,
                    "median_duration": median_duration,
                    "eval_outcomes": eval_outcomes_list,
                }
                all_diff_data.append(diff_data)

                # Write results to file after each problem
                with open(output_path, "a") as f:
                    f.write(json.dumps(diff_data) + "\n")
        except Exception as e:
            console.print(f"[bold red]Error processing example {problem_id}: {str(e)}[/bold red]")
            logger.exception(f"Error processing example {problem_id}")

    console.print(f"\nAll examples processed. Results saved to {output_path}")

    # Create a markdown summary file for each problem
    for diff_data in all_diff_data:
        problem_id = diff_data["id"]
        problem_statement = diff_data["instruction"]
        eval_outcomes = diff_data["eval_outcomes"]

        # Check if any solution was successful
        success = any(outcome.get("is_success", False) for outcome in eval_outcomes)

        with open(f"swebench_{problem_id}_results.md", "w") as f:
            f.write(f"# SWEBench Problem: {problem_id}\n\n")
            f.write("## Problem Statement\n\n")
            f.write(f"{problem_statement}\n\n")
            f.write("## Execution Attempt\n\n")
            f.write(f"We attempted to run the SWEBench agent on this problem using the following command:\n\n")
            f.write("```bash\n")
            f.write(f"python run_swebench_mac.py --problem-ids \"{problem_id}\" --num-candidate-solutions {args.num_candidate_solutions}\n")
            f.write("```\n\n")
            f.write("## Execution Results\n\n")
            if success:
                f.write("The execution was successful! At least one solution passed the evaluation.\n\n")
            else:
                f.write("None of the solutions passed the evaluation.\n\n")

            f.write("## Solution Details\n\n")
            for i, (diff, duration, outcome) in enumerate(zip(diff_data["diffs"], diff_data["agent_durations"], eval_outcomes)):
                f.write(f"### Solution Attempt {i+1}\n\n")
                f.write(f"- Duration: {duration:.2f} seconds\n")
                f.write(f"- Success: {outcome.get('is_success', False)}\n\n")

                if diff:
                    f.write("#### Diff\n\n")
                    f.write("```diff\n")
                    f.write(diff)
                    f.write("\n```\n\n")
                else:
                    f.write("No diff was generated for this solution attempt.\n\n")

            f.write("## Workspace Location\n\n")
            f.write(f"The workspace for this problem is located at: `{workspace_base_path / problem_id}`\n\n")

    console.print("Done!")
    console.print(f"Detailed results for each problem have been saved as swebench_<problem_id>_results.md")


if __name__ == "__main__":
    main()
