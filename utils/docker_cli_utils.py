import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

AUGMENT_ROOT = Path(__file__).parent.parent
MAX_DOCKER_CONCURRENCY = 4

def run_docker_command(command, check=True, retry_count=3, retry_delay=5):
    """Run a Docker command using subprocess with retries.

    Args:
        command: The Docker command to run
        check: Whether to check the return code
        retry_count: Number of times to retry on failure
        retry_delay: Delay between retries in seconds

    Returns:
        Tuple of (success, output)
    """
    for attempt in range(retry_count):
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=check,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120  # Add a timeout to prevent hanging
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                logging.warning(f"Docker command failed (attempt {attempt+1}/{retry_count}): {result.stderr}")
                if attempt < retry_count - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next retry
                    retry_delay *= 1.5
                else:
                    return False, result.stderr
        except subprocess.CalledProcessError as e:
            logging.error(f"Docker command failed with exception (attempt {attempt+1}/{retry_count}): {e.stderr}")
            if attempt < retry_count - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry
                retry_delay *= 1.5
            else:
                return False, e.stderr
        except subprocess.TimeoutExpired as e:
            logging.error(f"Docker command timed out (attempt {attempt+1}/{retry_count})")
            if attempt < retry_count - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry
                retry_delay *= 1.5
            else:
                return False, f"Command timed out after 120 seconds: {command}"

    return False, "Maximum retry attempts reached"

def get_issue_image_name(problem_id: str) -> str:
    """Fetch a docker image for the issue."""
    issue_key = problem_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{issue_key}:latest"

def set_volume_permissions(container_id: str, volume_path: Path = None):
    """Set permissions on the Docker volume using Docker exec commands.

    On macOS, we can't directly access the Docker volume paths from the host.
    Instead, we use Docker exec to run commands inside the container.

    Args:
        container_id: The ID of the Docker container
        volume_path: Optional path to the volume (not used on macOS)

    Returns:
        bool: True if permissions were set successfully, False otherwise
    """
    logging.info(f"Setting permissions for volume in container {container_id}")

    # First check if the container is running
    success, output = run_docker_command(
        f'docker inspect --format="{{{{.State.Running}}}}" {container_id}'
    )

    if not success:
        logging.warning(f"Failed to check container status: {output}")
        # Try to restart the container if it's not running
        restart_success, restart_output = run_docker_command(
            f'docker start {container_id}'
        )
        if not restart_success:
            logging.warning(f"Failed to restart container: {restart_output}")
            return False
        logging.info(f"Container {container_id} restarted successfully")
        # Give it a moment to start up
        time.sleep(5)

    # Check again if the container is running
    success, output = run_docker_command(
        f'docker inspect --format="{{{{.State.Running}}}}" {container_id}'
    )

    if not success or output.strip().lower() != "true":
        logging.warning(f"Container {container_id} is not running, cannot set permissions")
        return False

    try:
        # Set permissions inside the container
        success, output = run_docker_command(
            f'docker exec {container_id} bash -c "chmod -R a+rwx /testbed"'
        )
        if not success:
            logging.warning(f"Failed to set permissions in container: {output}")
            # Try a different approach
            alt_success, alt_output = run_docker_command(
                f'docker exec {container_id} bash -c "find /testbed -type d -exec chmod 755 {{}} \\; && find /testbed -type f -exec chmod 644 {{}} \\;"'
            )
            if not alt_success:
                logging.warning(f"Alternative permission setting also failed: {alt_output}")
                return False
            logging.info("Used alternative method to set permissions")

        logging.info(f"Volume permissions set in container {container_id}")
        return True
    except Exception as e:
        logging.warning(f"Failed to set permissions: {e}")
        # Don't raise an exception, as this might not be fatal
        return False

def start_container(workspace: Path, problem_id: str, semaphore: Any) -> str:
    """Start a docker container for the issue using CLI commands.

    Args:
        workspace: Path to the workspace directory
        problem_id: ID of the problem
        semaphore: Semaphore for Docker operations

    Returns:
        str: ID of the started container

    Raises:
        Exception: If the container could not be started after multiple attempts
    """
    # Stop any existing container with the same name pattern
    stop_container(f"sweb.augment.{problem_id}")

    # Generate a unique container name
    container_name = f"sweb.augment.{problem_id}_{uuid.uuid4().hex[:8]}"

    # Get the image name
    image_name = get_issue_image_name(problem_id)
    logging.info(f"Starting container for {problem_id}")

    # Pull the image with retries
    max_pull_attempts = 3
    for pull_attempt in range(max_pull_attempts):
        with semaphore:
            logging.info(f"Pulling image {image_name} (attempt {pull_attempt+1}/{max_pull_attempts})")
            success, output = run_docker_command(f"docker pull {image_name}")
            if success:
                logging.info(f"Finished pulling image {image_name}")
                break
            else:
                logging.error(f"Failed to pull image {image_name}: {output}")
                if pull_attempt < max_pull_attempts - 1:
                    logging.info(f"Retrying image pull in 10 seconds...")
                    time.sleep(10)
                else:
                    raise Exception(f"Failed to pull image {image_name} after {max_pull_attempts} attempts: {output}")

    # Run the container with retries
    max_run_attempts = 3
    container_id = None

    for run_attempt in range(max_run_attempts):
        with semaphore:
            logging.info(f"Running docker container for {image_name} in {workspace} (attempt {run_attempt+1}/{max_run_attempts})")
            command = (
                f'docker run --name {container_name} -d '
                f'-v /testbed '
                f'{image_name} '
                f'bash -c "git config --global user.email a && git config --global user.name a && '
                f'git config --global --add safe.directory /testbed && git commit --allow-empty -am augment && sleep 7200"'
            )
            success, output = run_docker_command(command)
            if success:
                # Get the container ID
                container_id = output.strip()
                logging.info(f"Started container {container_id} for {problem_id}")
                break
            else:
                logging.error(f"Failed to start container (attempt {run_attempt+1}/{max_run_attempts}): {output}")
                if run_attempt < max_run_attempts - 1:
                    logging.info(f"Retrying container start in 10 seconds...")
                    time.sleep(10)
                    # Try to clean up any failed container with this name
                    run_docker_command(f"docker rm -f {container_name}", check=False)
                    # Generate a new container name for the next attempt
                    container_name = f"sweb.augment.{problem_id}_{uuid.uuid4().hex[:8]}"
                else:
                    raise Exception(f"Failed to start container after {max_run_attempts} attempts: {output}")

    if not container_id:
        raise Exception(f"Failed to get container ID for {problem_id}")

    # Give it time to start and stabilize
    time.sleep(15)

    # Verify the container is running
    success, output = run_docker_command(f'docker inspect --format="{{{{.State.Running}}}}" {container_id}')
    if not success or output.strip().lower() != "true":
        logging.warning(f"Container {container_id} is not running after creation, attempting to start it")
        success, output = run_docker_command(f"docker start {container_id}")
        if not success:
            logging.error(f"Failed to start container {container_id}: {output}")
            raise Exception(f"Failed to start container {container_id}: {output}")
        # Give it more time to start
        time.sleep(10)

    # Get the volume path with retries
    max_inspect_attempts = 3
    volume_path = None

    for inspect_attempt in range(max_inspect_attempts):
        success, output = run_docker_command(f"docker inspect --format='{{{{.Mounts}}}}' {container_id}")
        if success:
            # Parse the volume path from the output
            # Example output: [{volume 1234567890abcdef /var/lib/docker/volumes/1234567890abcdef/_data /testbed local  true }]
            volume_info = output.strip()
            volume_parts = volume_info.split()
            if len(volume_parts) >= 3:
                volume_path = Path(volume_parts[2])
                break
            else:
                logging.warning(f"Failed to parse volume path from: {volume_info} (attempt {inspect_attempt+1}/{max_inspect_attempts})")
        else:
            logging.warning(f"Failed to get volume info (attempt {inspect_attempt+1}/{max_inspect_attempts}): {output}")

        if inspect_attempt < max_inspect_attempts - 1:
            logging.info(f"Retrying volume path retrieval in 5 seconds...")
            time.sleep(5)
        else:
            # If we can't get the volume path, try a fallback approach
            logging.warning("Using fallback approach to find volume path")
            success, output = run_docker_command(f"docker volume ls --format '{{{{.Name}}}}' | grep {container_id[:12]}")
            if success and output.strip():
                volume_name = output.strip().split('\n')[0]
                volume_path = Path(f"/var/lib/docker/volumes/{volume_name}/_data")
                logging.info(f"Found volume path using fallback: {volume_path}")
            else:
                # Last resort: use a fixed pattern based on container ID
                volume_path = Path(f"/var/lib/docker/volumes/{container_id[:64]}/_data")
                logging.warning(f"Using last resort volume path: {volume_path}")

    # Create symlink to the volume
    (workspace / problem_id).unlink(missing_ok=True)
    (workspace / problem_id).symlink_to(volume_path)
    logging.info(f"Created symlink from {workspace / problem_id} to {volume_path}")

    # Set permissions on the volume
    # Wait a bit to make sure the container is fully started
    time.sleep(5)
    perm_success = set_volume_permissions(container_id)

    if not perm_success:
        logging.warning(f"Failed to set permissions for container {container_id}, but continuing anyway")

    return container_id

def stop_container(container_id_or_name: str) -> None:
    """Stop a docker container using CLI commands.

    This function will attempt to stop and remove all containers that match the given ID or name pattern.
    It will not raise exceptions if the container doesn't exist or can't be stopped.

    Args:
        container_id_or_name: The ID or name pattern of the container(s) to stop
    """
    # Check if the container exists by ID first (exact match)
    if len(container_id_or_name) >= 12:  # Looks like a container ID
        success, output = run_docker_command(f"docker ps -a --filter id={container_id_or_name} --format '{{{{.ID}}}}'", check=False)
        if success and output.strip():
            container_id = output.strip()
            logging.info(f"Found container by ID: {container_id}")

            # Stop the container
            logging.info(f"Stopping container {container_id}")
            stop_success, stop_output = run_docker_command(f"docker stop {container_id}", check=False)
            if not stop_success:
                logging.warning(f"Failed to stop container {container_id}: {stop_output}")
                # Try force removal
                logging.info(f"Attempting force removal of container {container_id}")
                run_docker_command(f"docker rm -f {container_id}", check=False)
            else:
                # Remove the container
                logging.info(f"Removing container {container_id}")
                run_docker_command(f"docker rm {container_id}", check=False)

            # Verify removal
            verify_success, verify_output = run_docker_command(f"docker ps -a --filter id={container_id} --format '{{{{.ID}}}}'", check=False)
            if not verify_success or not verify_output.strip():
                logging.info(f"Container {container_id} removed")
            else:
                logging.warning(f"Container {container_id} may still exist after removal attempt")

            return

    # Check for containers by name pattern
    success, output = run_docker_command(f"docker ps -a --filter name={container_id_or_name} --format '{{{{.ID}}}}'", check=False)
    if not success or not output.strip():
        logging.info(f"No containers found matching pattern: {container_id_or_name}")
        return

    # Stop all matching containers
    container_ids = output.strip().split('\n')
    for container_id in container_ids:
        if not container_id:
            continue

        logging.info(f"Stopping container {container_id}")
        stop_success, stop_output = run_docker_command(f"docker stop {container_id}", check=False)
        if not stop_success:
            logging.warning(f"Failed to stop container {container_id}: {stop_output}")
            # Try force removal
            logging.info(f"Attempting force removal of container {container_id}")
            run_docker_command(f"docker rm -f {container_id}", check=False)
        else:
            # Remove the container
            logging.info(f"Removing container {container_id}")
            rm_success, rm_output = run_docker_command(f"docker rm {container_id}", check=False)
            if not rm_success:
                logging.warning(f"Failed to remove container {container_id}: {rm_output}")
                # Try force removal
                logging.info(f"Attempting force removal of container {container_id}")
                run_docker_command(f"docker rm -f {container_id}", check=False)

        # Verify removal
        verify_success, verify_output = run_docker_command(f"docker ps -a --filter id={container_id} --format '{{{{.ID}}}}'", check=False)
        if not verify_success or not verify_output.strip():
            logging.info(f"Container {container_id} removed")
        else:
            logging.warning(f"Container {container_id} may still exist after removal attempt")

    # Wait a bit to ensure Docker has fully processed the removals
    time.sleep(3)

def setup_workspace(
    workspace: Path, problem_id: str, lock: Any, semaphore: Any
) -> Tuple[Dict[str, str], str]:
    """Setup the workspace for the agent."""
    env: Dict[str, str] = os.environ.copy()

    # Create a conda environment; we don't use it, but it protects the
    # agent's environment from changes.
    logging.debug(f"Creating conda enviroment in {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    # Multiple simultaneous conda installs are no good.
    with lock:
        subprocess.check_output(
            [
                "conda",
                "create",
                "-y",
                "-q",
                "-p",
                str(workspace / "conda_3.9"),
                "python==3.9",
            ]
        )

    env["ISSUE_ID"] = problem_id
    env["SWEBENCH_WORKSPACE"] = str(workspace)
    env["PATH"] = f"{workspace}/python_wrappers/bin:{workspace}/conda_3.9/bin" + (
        f":{env['PATH']}" if "PATH" in env else ""
    )
    env["PYTHONPATH"] = f"{AUGMENT_ROOT}" + (
        f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else ""
    )
    for k, v in env.items():
        logging.debug(f"ENV {k}=={v}")

    # Start the container
    container_id = start_container(workspace, problem_id, semaphore)

    return env, container_id
