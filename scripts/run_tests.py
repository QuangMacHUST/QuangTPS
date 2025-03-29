#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test runner for QuangTPS.

This script runs all tests for the QuangTPS system and generates a report.
It can also be used to run specific test modules or test cases.
"""

import os
import sys
import argparse
import unittest
import time
import logging
from datetime import datetime

# Add the project root to the Python path if running from scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest


def setup_logging(log_file=None):
    """Set up logging for test runs.
    
    Args:
        log_file: Optional path to a log file
    """
    log_level = logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure the root logger
    logging.basicConfig(level=log_level, format=log_format)
    
    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def run_unittest_tests(test_path=None, pattern='test_*.py', verbosity=1):
    """Run tests using the unittest framework.
    
    Args:
        test_path: Path to the directory containing tests
        pattern: Pattern to match test files
        verbosity: Verbosity level for test output
        
    Returns:
        TestResult object
    """
    if test_path is None:
        test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'quangtps', 'tests')
    
    loader = unittest.TestLoader()
    
    if os.path.isfile(test_path):
        # Run a specific test file
        if test_path.endswith('.py'):
            # Get the module name from the file path
            sys.path.insert(0, os.path.dirname(test_path))
            module_name = os.path.basename(test_path)[:-3]
            suite = loader.loadTestsFromName(module_name)
        else:
            print(f"Error: {test_path} is not a Python file")
            return None
    else:
        # Run all tests in the directory
        suite = loader.discover(test_path, pattern=pattern)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def run_pytest_tests(test_path=None, options=None):
    """Run tests using pytest.
    
    Args:
        test_path: Path to the directory or file containing tests
        options: Additional pytest options
        
    Returns:
        Exit code from pytest
    """
    if test_path is None:
        test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'quangtps', 'tests')
    
    # Build pytest arguments
    args = [test_path]
    
    if options:
        args.extend(options)
    
    # Run pytest
    return pytest.main(args)


def generate_test_report(result, output_file=None, format='text'):
    """Generate a test report.
    
    Args:
        result: TestResult object from unittest or exit code from pytest
        output_file: Output file for the report
        format: Report format ('text' or 'html')
        
    Returns:
        Path to the report file if output_file is provided, None otherwise
    """
    if isinstance(result, unittest.TestResult):
        # Generate report for unittest result
        report = []
        report.append("=" * 80)
        report.append(f"QuangTPS Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append(f"Tests run: {result.testsRun}")
        report.append(f"Failures: {len(result.failures)}")
        report.append(f"Errors: {len(result.errors)}")
        report.append(f"Skipped: {len(result.skipped)}")
        report.append("")
        
        # Details of failures
        if result.failures:
            report.append("FAILURES")
            report.append("-" * 80)
            for test, trace in result.failures:
                report.append(f"{test}")
                report.append(f"{trace}")
                report.append("")
        
        # Details of errors
        if result.errors:
            report.append("ERRORS")
            report.append("-" * 80)
            for test, trace in result.errors:
                report.append(f"{test}")
                report.append(f"{trace}")
                report.append("")
        
        # Join the report lines
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            return output_file
        else:
            print(report_text)
            return None
    else:
        # Generate report for pytest exit code
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"pytest exit code: {result}\n")
            return output_file
        else:
            print(f"pytest exit code: {result}")
            return None


def main():
    """Main function for the test runner."""
    parser = argparse.ArgumentParser(description='Run tests for QuangTPS')
    parser.add_argument('--path', help='Path to test directory or file')
    parser.add_argument('--pattern', default='test_*.py', help='Pattern to match test files')
    parser.add_argument('--engine', choices=['unittest', 'pytest'], default='unittest',
                       help='Test engine to use')
    parser.add_argument('--report', help='Output file for test report')
    parser.add_argument('--format', choices=['text', 'html'], default='text',
                       help='Report format')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity')
    parser.add_argument('--log', help='Log file for test output')
    parser.add_argument('--pytest-args', nargs=argparse.REMAINDER,
                       help='Additional arguments to pass to pytest')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log)
    
    # Record start time
    start_time = time.time()
    
    # Run tests
    if args.engine == 'unittest':
        result = run_unittest_tests(args.path, args.pattern, args.verbose)
    else:
        result = run_pytest_tests(args.path, args.pytest_args)
    
    # Record end time
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print elapsed time
    print(f"\nTest execution time: {elapsed:.2f} seconds")
    
    # Generate report if requested
    if args.report:
        report_file = generate_test_report(result, args.report, args.format)
        if report_file:
            print(f"Test report written to {report_file}")
    
    # Return appropriate exit code
    if args.engine == 'unittest':
        if result.wasSuccessful():
            return 0
        else:
            return 1
    else:
        return result


if __name__ == '__main__':
    sys.exit(main()) 