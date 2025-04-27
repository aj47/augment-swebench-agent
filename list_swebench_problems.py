#!/usr/bin/env python3
"""
Utility script to list and search SWE-bench problem IDs.

This script loads the SWE-bench dataset and displays problem IDs,
optionally filtering by keywords or showing additional details.
"""

import argparse
import sys
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="List and search SWE-bench problem IDs")
    parser.add_argument(
        "--search", 
        type=str, 
        default=None, 
        help="Search for problems containing this keyword in ID or problem statement"
    )
    parser.add_argument(
        "--details", 
        action="store_true", 
        help="Show detailed information including problem statements"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of problems displayed"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["verified", "full", "lite"], 
        default="verified", 
        help="Which SWE-bench dataset to use (default: verified)"
    )
    
    args = parser.parse_args()

    # Initialize console
    console = Console()

    # Map dataset choice to actual dataset name
    dataset_name = {
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
        "lite": "princeton-nlp/SWE-bench_Lite",
    }[args.dataset]

    # Load the SWE-bench dataset
    console.print(f"Loading SWE-bench dataset ({args.dataset})...")
    try:
        swebench_dataset = load_dataset(dataset_name)["test"].to_pandas()
    except Exception as e:
        console.print(f"[bold red]Error loading dataset: {str(e)}[/bold red]")
        sys.exit(1)

    console.print(f"Found {len(swebench_dataset)} problems in the dataset.")

    # Filter by search term if provided
    if args.search:
        search_term = args.search.lower()
        filtered_dataset = swebench_dataset[
            swebench_dataset["instance_id"].str.lower().str.contains(search_term) | 
            swebench_dataset["problem_statement"].str.lower().str.contains(search_term)
        ]
        console.print(f"Found {len(filtered_dataset)} problems matching search term '{args.search}'.")
    else:
        filtered_dataset = swebench_dataset

    # Apply limit if provided
    if args.limit and args.limit < len(filtered_dataset):
        display_dataset = filtered_dataset.iloc[:args.limit]
        console.print(f"Displaying first {args.limit} problems.")
    else:
        display_dataset = filtered_dataset

    # Display problems
    if args.details:
        # Detailed view with problem statements
        for i, problem in enumerate(display_dataset.itertuples()):
            panel = Panel(
                f"[bold blue]ID:[/bold blue] {problem.instance_id}\n\n"
                f"[bold green]Problem Statement:[/bold green]\n{problem.problem_statement}",
                title=f"Problem {i+1}/{len(display_dataset)}",
                border_style="blue",
                padding=(1, 2),
            )
            console.print(panel)
            
            # Add a separator between problems
            if i < len(display_dataset) - 1:
                console.print("---")
    else:
        # Simple table view with just IDs
        table = Table(title=f"SWE-bench Problems ({args.dataset})")
        table.add_column("Index", justify="right", style="cyan")
        table.add_column("Problem ID", style="green")
        
        for i, problem in enumerate(display_dataset.itertuples()):
            table.add_row(str(i+1), problem.instance_id)
        
        console.print(table)

    # Print usage examples
    console.print("\n[bold]Usage examples:[/bold]")
    console.print("To run specific problems:")
    console.print(f"  python run_agent_on_swebench_problem.py --problem-ids", end=" ")
    
    # Show example IDs from the filtered dataset (up to 3)
    example_ids = display_dataset["instance_id"].tolist()[:3]
    for id in example_ids:
        console.print(f'"{id}"', end=" ")
    
    console.print("--num-candidate-solutions 2")

if __name__ == "__main__":
    main()
