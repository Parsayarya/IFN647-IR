#!/usr/bin/env python3
"""
Main orchestrator for the Information Retrieval system.
Runs BM25, LMRM, PRRM, and statistical analysis in sequence.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_directories():
    """Create necessary output directories."""
    directories = [
        "outputs",
        "outputs/BM25",
        "outputs/LMRM", 
        "outputs/PRRM"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def check_required_files():
    """Check if all required input files exist."""
    required_files = [
        "data/Queries-1.txt",
        "data/common-english-words.txt"
    ]
    
    required_dirs = [
        "data/DataSets",
        "data/EvaluationBenchmark"
    ]
    
    missing_items = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_items.append(f"File: {file_path}")
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_items.append(f"Directory: {dir_path}")
    
    if missing_items:
        print("Missing required files/directories:")
        for item in missing_items:
            print(f"   - {item}")
        return False
    
    print("All required files and directories found")
    return True

def run_bm25():
    """Run BM25 model."""
    print("\n" + "="*50)
    print("RUNNING BM25 MODEL")
    print("="*50)
    
    try:
        # Change to src directory and run BM25
        os.chdir("src")
        result = subprocess.run([sys.executable, "run_bm25.py"], 
                              capture_output=True, text=True, check=True)
        print("BM25 completed successfully")
        print("BM25 Output:")
        print(result.stdout)
        if result.stderr:
            print("BM25 Warnings/Errors:")
            print(result.stderr)
        os.chdir("..")
        return True
    except subprocess.CalledProcessError as e:
        print(f"BM25 failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        os.chdir("..")
        return False

def run_lmrm():
    """Run LMRM model."""
    print("\n" + "="*50)
    print("RUNNING LMRM MODEL")
    print("="*50)
    
    try:
        # Change to src directory and run LMRM
        os.chdir("src")
        result = subprocess.run([sys.executable, "run_lmrm.py"], 
                              capture_output=True, text=True, check=True)
        print("LMRM completed successfully")
        print("LMRM Output:")
        print(result.stdout)
        if result.stderr:
            print("LMRM Warnings/Errors:")
            print(result.stderr)
        os.chdir("..")
        return True
    except subprocess.CalledProcessError as e:
        print(f"LMRM failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        os.chdir("..")
        return False

def run_prrm():
    """Run PRRM model."""
    print("\n" + "="*50)
    print("RUNNING PRRM MODEL")
    print("="*50)
    
    try:
        # Change to src directory and run PRRM
        os.chdir("src")
        result = subprocess.run([sys.executable, "run_prrm.py"], 
                              capture_output=True, text=True, check=True)
        print("PRRM completed successfully")
        print("PRRM Output:")
        print(result.stdout)
        if result.stderr:
            print("PRRM Warnings/Errors:")
            print(result.stderr)
        os.chdir("..")
        return True
    except subprocess.CalledProcessError as e:
        print(f"PRRM failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        os.chdir("..")
        return False

def run_statistical_analysis():
    """Run statistical analysis and t-tests."""
    print("\n" + "="*50)
    print("RUNNING STATISTICAL ANALYSIS")
    print("="*50)
    
    try:
        # Change to src directory and run statistical analysis
        os.chdir("src")
        result = subprocess.run([sys.executable, "statistical_analysis.py"], 
                              capture_output=True, text=True, check=True)
        print("Statistical analysis completed successfully")
        print("Statistical Analysis Output:")
        print(result.stdout)
        if result.stderr:
            print("Statistical Analysis Warnings/Errors:")
            print(result.stderr)
        os.chdir("..")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Statistical analysis failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        os.chdir("..")
        return False

def copy_outputs():
    """Copy all outputs to organized output directories."""
    print("\n" + "="*50)
    print("ORGANIZING OUTPUTS")
    print("="*50)
    
    # Define source and destination mappings
    output_mappings = [
        # BM25 outputs
        ("data/RankingOutputs_BM25", "outputs/BM25/rankings"),
        ("data/EvaluationOutputs", "outputs/BM25/evaluations"),
        
        # LMRM outputs
        ("src/RankingOutputs_LMRM", "outputs/LMRM/rankings"),
        
        # PRRM outputs
        ("src/RankingOutputs_PRRM", "outputs/PRRM/rankings"),
        ("src/PRRM_Evaluation_Results.csv", "outputs/PRRM/PRRM_Evaluation_Results.csv"),
    ]
    
    for source, destination in output_mappings:
        if os.path.exists(source):
            try:
                if os.path.isdir(source):
                    if os.path.exists(destination):
                        shutil.rmtree(destination)
                    shutil.copytree(source, destination)
                    print(f"Copied directory: {source} to {destination}")
                else:
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.copy2(source, destination)
                    print(f"Copied file: {source} to {destination}")
            except Exception as e:
                print(f"Failed to copy {source}: {e}")
        else:
            print(f"Source not found: {source}")

def print_summary():
    """Print a summary of what was generated."""
    print("\n" + "="*50)
    print("EXECUTION SUMMARY")
    print("="*50)
    
    print("\nGenerated outputs:")
    print("outputs/")
    print("  ├── BM25/")
    print("  │   ├── rankings/          # BM25 ranking files")
    print("  │   └── evaluations/       # BM25 evaluation metrics")
    print("  ├── LMRM/")
    print("  │   ├── rankings/          # LMRM ranking files")
    print("  │   └── LMRM_Evaluation_Results.csv")
    print("  └── PRRM/")
    print("      ├── rankings/          # PRRM ranking files")
    print("      └── PRRM_Evaluation_Results.csv")
    
    print("\nKey files to check:")
    print("- outputs/BM25/evaluations/BM25_*.csv - BM25 performance metrics")
    print("- outputs/PRRM/PRRM_Evaluation_Results.csv - PRRM performance results")
    print("- outputs/LMRM/LMRM_Evaluation_Results.csv - LMRM performance results")
    print("- Statistical analysis results were printed to console")

def main():
    """Main orchestrator function."""
    print("Starting Information Retrieval System Pipeline")
    print("="*100)
    
    # Check if we're in the correct directory
    if not os.path.exists("src") or not os.path.exists("data"):
        print("Error: Please run this script from the project root directory")
        print("   Expected structure:")
        print("   ├── main.py (this file)")
        print("   ├── src/")
        print("   └── data/")
        return False
    
    # Setup
    setup_directories()
    
    if not check_required_files():
        print("\n Cannot proceed without required files. Please check the folder structure.")
        return False
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Run BM25
    if run_bm25():
        success_count += 1
    else:
        print("BM25 failed. Cannot proceed with pipeline.")
        return False
    
    # Step 2: Run LMRM  
    if run_lmrm():
        success_count += 1
    else:
        print("LMRM failed. Cannot proceed with PRRM.")
        return False
    
    # Step 3: Run PRRM (requires BM25 and LMRM outputs)
    if run_prrm():
        success_count += 1
    else:
        print("PRRM failed. Statistical analysis may be incomplete.")
    
    # Step 4: Run statistical analysis
    if run_statistical_analysis():
        success_count += 1
    
    # Organize outputs
    copy_outputs()
    
    # Print summary
    print_summary()
    
    print(f"\n Pipeline completed: {success_count}/{total_steps} steps successful")
    
    if success_count == total_steps:
        print("All components completed successfully")
        return True
    else:
        print("Some components failed. Check the logs above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)