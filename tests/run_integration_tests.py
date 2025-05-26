#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration Test Runner for Eden AI Coding Platform

This script runs all integration tests for the Eden AI Coding Platform and
provides detailed reporting on component integration status.
"""

import unittest
import sys
import os
import time
import argparse
import json
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import test classes
from test_integration import TestPlatformIntegration, TestErrorHandling, TestCrossComponentCommunication
from test_dashboard_integration import TestDashboardIntegration
from test_ai_super_agent import TestAISuperAgent

class IntegrationTestReport:
    """Generates and displays reports for integration test results"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.end_time = None
    
    def add_result(self, test_name, success, details=None):
        """Add a test result to the report"""
        self.results[test_name] = {
            'success': success,
            'details': details or {}
        }
    
    def finish(self):
        """Mark the end of testing"""
        self.end_time = time.time()
    
    def get_summary(self):
        """Get a summary of the test results"""
        total = len(self.results)
        successful = sum(1 for result in self.results.values() if result['success'])
        failed = total - successful
        duration = round(self.end_time - self.start_time, 2) if self.end_time else 0
        
        return {
            'total_tests': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'duration_seconds': duration
        }
    
    def print_report(self):
        """Print a formatted report of the test results"""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("EDEN AI CODING PLATFORM - INTEGRATION TEST REPORT")
        print("=" * 80)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print("=" * 80)
        print("\nDETAILED RESULTS:")
        
        # Group results by component
        components = {}
        for test_name, result in self.results.items():
            component = test_name.split('.')[0] if '.' in test_name else 'Unknown'
            if component not in components:
                components[component] = []
            components[component].append((test_name, result))
        
        # Print results by component
        for component, tests in components.items():
            print(f"\n{component}:")
            for test_name, result in tests:
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"  {status} - {test_name}")
                if not result['success'] and 'error' in result['details']:
                    print(f"    Error: {result['details']['error']}")
        
        print("\n" + "=" * 80)
        if summary['failed'] == 0:
            print("🎉 ALL INTEGRATION TESTS PASSED! The platform is functioning correctly.")
        else:
            print(f"⚠️  {summary['failed']} TESTS FAILED! Some components may not be integrating correctly.")
        print("=" * 80 + "\n")
    
    def save_report(self, filename):
        """Save the report as a JSON file"""
        report_data = {
            'summary': self.get_summary(),
            'results': self.results,
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Report saved to {filename}")


def load_tests(test_classes, pattern=None):
    """Load test classes into a test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        if pattern:
            tests = loader.loadTestsFromTestCase(test_class)
            for test in tests:
                if pattern.lower() in test.id().lower():
                    suite.addTest(test)
        else:
            suite.addTest(loader.loadTestsFromTestCase(test_class))
    
    return suite


def run_tests(args):
    """Run the integration tests and generate a report"""
    # Initialize the test report
    report = IntegrationTestReport()
    
    # Define all test classes
    all_test_classes = [
        TestAISuperAgent,
        TestPlatformIntegration,
        TestErrorHandling,
        TestCrossComponentCommunication,
        TestDashboardIntegration
    ]
    
    # Filter test classes if component is specified
    if args.component:
        component = args.component.lower()
        test_classes = [cls for cls in all_test_classes 
                       if component in cls.__name__.lower()]
    else:
        test_classes = all_test_classes
    
    # Load tests
    suite = load_tests(test_classes, args.pattern)
    
    # Create a custom result handler to gather successes too
    class SuccessTrackingResult(unittest.TestResult):
        def __init__(self):
            super().__init__()
            self.successes = []
        
        def addSuccess(self, test):
            super().addSuccess(test)
            self.successes.append(test)
    
    # Run tests with result tracking
    result = SuccessTrackingResult()
    
    # Try running tests with additional error handling
    try:
        suite.run(result)
    except Exception as e:
        print(f"Error running test suite: {e}")
    
    # Process results
    for test in result.failures + result.errors:
        test_name = test[0].id()
        error_message = test[1]
        report.add_result(test_name, False, {'error': error_message})
    
    # Add successful tests
    for test in result.successes if hasattr(result, 'successes') else []:
        report.add_result(test.id(), True)
    
    # Infer successful tests if not explicitly tracked
    if not hasattr(result, 'successes'):
        run_tests = set(t[0].id() for t in result.failures + result.errors)
        for test in suite:
            if test.id() not in run_tests:
                report.add_result(test.id(), True)
    
    # Finish the report
    report.finish()
    
    # Print and save the report if requested
    report.print_report()
    if args.report_file:
        report.save_report(args.report_file)
    
    return report.get_summary()['failed'] == 0


def main():
    """Main entry point for the test runner"""
    parser = argparse.ArgumentParser(description="Run integration tests for Eden AI Coding Platform")
    parser.add_argument('--component', help='Filter tests by component name')
    parser.add_argument('--pattern', help='Filter tests by pattern in test name')
    parser.add_argument('--report-file', help='Save test report to specified file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose test output')
    args = parser.parse_args()
    
    # Set verbosity level
    if args.verbose:
        unittest.main(verbosity=2, exit=False)
    
    # Run the tests
    success = run_tests(args)
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
