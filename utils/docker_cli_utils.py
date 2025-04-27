import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

AUGMENT_ROOT = Path(__file__).parent.parent
MAX_DOCKER_CONCURRENCY = 4

def run_docker_command(command, check=True):
    """Run a Docker command using subprocess."""
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
        logging.error(f"Docker command failed: {e.stderr}")
        return False, e.stderr

def get_issue_image_name(problem_id: str) -> str:
    """Fetch a docker image for the issue."""
    issue_key = problem_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{issue_key}:latest"

def set_volume_permissions(volume_path: Path):
    """Set permissions on the Docker volume."""
    # Make sure we can read the volume
    my_uid = os.getuid()
    my_gid = os.getgid()
    logging.info(f"Fixing permissions for {volume_path} to {my_uid}:{my_gid}")
    env = os.environ.copy()
    
    try:
        # Make parent directories readable
        success, output = run_docker_command(
            f"sudo chmod a+rx {' '.join([p.as_posix() for p in volume_path.parents])}"
        )
        if not success:
            logging.warning(f"Failed to chmod parent directories: {output}")
            raise Exception(f"Failed to chmod parent directories: {output}")
        
        # Change ownership of the volume
        success, output = run_docker_command(
            f"sudo chown -R {my_uid}:{my_gid} {volume_path.as_posix()}"
        )
        if not success:
            logging.warning(f"Failed to chown volume: {output}")
            raise Exception(f"Failed to chown volume: {output}")
        
        logging.debug(f"Volume permissions set: {output}")
    except Exception as e:
        logging.warning(f"Failed to set permissions: {e}")
        raise

def start_container(workspace: Path, problem_id: str, semaphore: Any) -> str:
    """Start a docker container for the issue using CLI commands."""
    # Stop any existing container with the same name
    container_name = f"sweb.augment.{problem_id}_{uuid.uuid4().hex[:8]}"
    stop_container(f"sweb.augment.{problem_id}")
    
    # Get the image name
    image_name = get_issue_image_name(problem_id)
    logging.info(f"Starting container for {problem_id}")
    
    # Pull the image
    with semaphore:
        logging.info(f"Pulling image {image_name}")
        success, output = run_docker_command(f"docker pull {image_name}")
        if not success:
            logging.error(f"Failed to pull image {image_name}: {output}")
            raise Exception(f"Failed to pull image {image_name}: {output}")
        logging.info(f"Finished pulling image {image_name}")
    
    # Run the container
    with semaphore:
        logging.info(f"Running docker container for {image_name} in {workspace}")
        command = (
            f'docker run --name {container_name} -d '
            f'-v /testbed '
            f'{image_name} '
            f'bash -c "git config --global user.email a && git config --global user.name a && '
            f'git config --global --add safe.directory /testbed && git commit --allow-empty -am augment && sleep 7200"'
        )
        success, output = run_docker_command(command)
        if not success:
            logging.error(f"Failed to start container: {output}")
            raise Exception(f"Failed to start container: {output}")
        
        # Get the container ID
        container_id = output.strip()
        logging.info(f"Started container {container_id} for {problem_id}")
    
    # Give it a second to start
    time.sleep(10)
    
    # Get the volume path
    success, output = run_docker_command(f"docker inspect --format='{{{{.Mounts}}}}' {container_id}")
    if not success:
        logging.error(f"Failed to get volume info: {output}")
        raise Exception(f"Failed to get volume info: {output}")
    
    # Parse the volume path from the output
    # Example output: [{volume 1234567890abcdef /var/lib/docker/volumes/1234567890abcdef/_data /testbed local  true }]
    volume_info = output.strip()
    volume_parts = volume_info.split()
    if len(volume_parts) < 3:
        logging.error(f"Failed to parse volume path from: {volume_info}")
        raise Exception(f"Failed to parse volume path from: {volume_info}")
    
    volume_path = Path(volume_parts[2])
    (workspace / problem_id).unlink(missing_ok=True)
    (workspace / problem_id).symlink_to(volume_path)
    
    # Set permissions on the volume
    retry = True
    while True:
        try:
            set_volume_permissions(volume_path)
            break
        except Exception as e:
            logging.warning(f"Failed to set permissions: {e}")
            if not retry:
                # Don't fail for this, it may not be fatal.
                break
            retry = False
            time.sleep(10)
    
    return container_id

def stop_container(container_id_or_name: str) -> None:
    """Stop a docker container using CLI commands."""
    # Check if the container exists
    success, output = run_docker_command(f"docker ps -a --filter name={container_id_or_name} --format '{{{{.ID}}}}'", check=False)
    if not success or not output.strip():
        logging.info(f"Container {container_id_or_name} not found")
        return
    
    # Stop the container
    container_ids = output.strip().split('\n')
    for container_id in container_ids:
        if not container_id:
            continue
        
        logging.info(f"Stopping container {container_id}")
        run_docker_command(f"docker stop {container_id}", check=False)
        
        logging.info(f"Removing container {container_id}")
        run_docker_command(f"docker rm -f {container_id}", check=False)
        time.sleep(5)
        logging.info(f"Container {container_id} removed")

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
