#!/usr/bin/env python
"""
runner.py

Runs edited.py 10 times with save_data=True and varying sample comments.
"""

import subprocess
import sys

def run_edited_py(task='2AF', num_runs=10, variable='orthogonal'):
    """Run edited.py multiple times with different sample comments."""
    for sample_num in range(num_runs):
        comment = f"{task} {variable} sample_{sample_num}"
        cmd = [
            sys.executable,
            'BioRNN_RichLazy/edited.py',
            '--save_data', 'True',
            '--task', task,
            '--var_name', variable,
            '--comment', comment
        ]
        
        print(f"\n{'='*60}")
        print(f"Running sample {sample_num} of {num_runs}")
        print(f"Task: {task}, Comment: {comment}")
        print(f"{'='*60}\n")
        
        result = subprocess.run(cmd, cwd='.')
        
        if result.returncode != 0:
            print(f"Error: edited.py exited with code {result.returncode}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"All {num_runs} runs completed successfully!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run edited.py multiple times')
    parser.add_argument('--task', default='2AF', type=str, choices=['2AF', 'DMS', 'CXT'],
                        help='Task to run')
    parser.add_argument('--num_runs', default=10, type=int,
                                help='Number of times to run edited.py')
    parser.add_argument('--var_name', default='orthogonal', type=str, choices=['rr', 'spectral', 'orthogonal'],
                        help='Variable name for initialization')
    
    args = parser.parse_args()
    
    run_edited_py(task=args.task, num_runs=args.num_runs, variable=args.var_name)
