# Docker Robustness Improvements for SWEBench

This document describes the improvements made to the Docker integration in the SWEBench execution to make it more robust, particularly on macOS systems.

## Overview

The SWEBench execution relies heavily on Docker to create isolated environments for each problem. However, Docker integration can be challenging, especially on macOS where Docker Desktop runs in a VM with different socket locations and volume mounting behavior compared to Linux.

We've made several significant improvements to the Docker integration to address these challenges and make the SWEBench execution more robust.

## Key Improvements

### 1. Enhanced Docker Command Execution

The `run_docker_command` function in `utils/docker_cli_utils.py` has been enhanced with:

- **Retry Mechanism**: Commands now automatically retry on failure with exponential backoff
- **Timeout Protection**: Commands now have a timeout to prevent hanging indefinitely
- **Better Error Reporting**: More detailed error messages are now provided
- **Graceful Failure Handling**: Commands can continue execution even after non-critical failures

Example:
```python
def run_docker_command(command, check=True, retry_count=3, retry_delay=5):
    """Run a Docker command using subprocess with retries."""
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
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    return False, result.stderr
        except Exception as e:
            # Error handling with retries
            # ...
```

### 2. Improved Container Lifecycle Management

The container lifecycle management has been improved with:

- **Container State Verification**: The system now verifies container state after creation
- **Automatic Container Restart**: Containers are automatically restarted if they stop unexpectedly
- **Enhanced Permission Setting**: Multiple approaches to setting permissions with fallbacks
- **Better Container Cleanup**: More thorough container cleanup to prevent resource leaks

Example:
```python
def start_container(workspace: Path, problem_id: str, semaphore: Any) -> str:
    """Start a docker container for the issue using CLI commands."""
    # ...
    
    # Verify the container is running
    success, output = run_docker_command(f'docker inspect --format="{{{{.State.Running}}}}" {container_id}')
    if not success or output.strip().lower() != "true":
        logging.warning(f"Container {container_id} is not running after creation, attempting to start it")
        success, output = run_docker_command(f"docker start {container_id}")
        # ...
```

### 3. More Robust Volume Path Handling

Volume path handling has been improved with:

- **Multiple Path Finding Strategies**: Several methods to find volume paths
- **Fallback Mechanisms**: Automatic fallbacks when primary methods fail
- **Better Symlink Management**: Improved symlink creation and verification
- **Volume Access Verification**: Checks to ensure volume access before proceeding

Example:
```python
# Get the volume path with retries
max_inspect_attempts = 3
volume_path = None

for inspect_attempt in range(max_inspect_attempts):
    # Try to get volume path
    # ...
    
    if inspect_attempt < max_inspect_attempts - 1:
        logging.info(f"Retrying volume path retrieval in 5 seconds...")
        time.sleep(5)
    else:
        # If we can't get the volume path, try a fallback approach
        logging.warning("Using fallback approach to find volume path")
        # ...
```

### 4. Enhanced Diff Generation

Diff generation has been improved with:

- **Multiple Generation Strategies**: Several methods to generate diffs
- **Strategy Prioritization**: Methods are tried in order of reliability
- **Detailed Error Reporting**: Each method reports detailed errors
- **Graceful Fallbacks**: System continues even if some methods fail

Example:
```python
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
```

### 5. More Robust Evaluation Process

The evaluation process has been improved with:

- **Docker Accessibility Check**: Checks if Docker is accessible before evaluation
- **Prediction File Validation**: Validates and fixes prediction files
- **Multiple Evaluation Attempts**: Retries evaluation with proper delays
- **Detailed Error Reporting**: More detailed error reporting for evaluation
- **Fallback Evaluation**: Manual evaluation when automated process fails

Example:
```python
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
        # ...
        
        if attempt < max_eval_attempts - 1:
            console.print(f"Retrying evaluation in 10 seconds...")
            time.sleep(10)
            
    except Exception as e:
        console.print(f"[bold yellow]Warning: Evaluation attempt {attempt+1}/{max_eval_attempts} failed: {str(e)}[/bold yellow]")
        # ...
```

## Testing and Validation

These improvements have been tested with various SWEBench problems, including:

- Basic Docker operations (pulling images, running containers)
- Volume mounting and permissions
- Container naming and cleanup
- SWEBench problem setup and execution
- Diff generation and evaluation

## Remaining Challenges

While the Docker integration is now more robust, there are still some challenges:

1. **Docker Socket Connection**: The evaluation still sometimes fails with "Error while fetching server API version" which suggests Docker socket connectivity issues
2. **Container Running State**: Containers sometimes stop running after creation
3. **Volume Access**: There are still occasional issues with accessing Docker volumes

## Conclusion

The improvements made to the Docker integration in SWEBench have significantly enhanced its robustness. The system now handles failures more gracefully, provides better error reporting, and implements multiple fallback strategies. While some challenges remain, the overall reliability of the Docker integration has been improved, making the SWEBench execution more reliable across different environments.
