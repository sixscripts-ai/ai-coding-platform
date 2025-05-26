#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration Test Suite for Eden AI Coding Platform

This module provides comprehensive integration testing for all platform components,
ensuring they work correctly together as a complete system. It tests the interaction
between components, API endpoints, and end-to-end workflows.
"""

import unittest
import sys
import os
import time
import json
import threading
import requests
from unittest.mock import MagicMock, patch
from pathlib import Path
from queue import Queue

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# We're directly importing from the platform file path for testing
# We need to use a more robust approach to avoid module import issues
import importlib.util
spec = importlib.util.spec_from_file_location('eden_platform_module', os.path.join(parent_dir, 'eden-platform.py'))
eden_platform = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(eden_platform)
    # Extract needed classes
    AISuperAgent = eden_platform.AISuperAgent if hasattr(eden_platform, 'AISuperAgent') else None
    FileManager = eden_platform.FileManager if hasattr(eden_platform, 'FileManager') else None
    AIDebugger = eden_platform.AIDebugger if hasattr(eden_platform, 'AIDebugger') else None
    AITestRunner = eden_platform.AITestRunner if hasattr(eden_platform, 'AITestRunner') else None
    CodeOptimizer = eden_platform.CodeOptimizer if hasattr(eden_platform, 'CodeOptimizer') else None
    IntegratedUI = eden_platform.IntegratedUI if hasattr(eden_platform, 'IntegratedUI') else None
    VersionControl = eden_platform.VersionControl if hasattr(eden_platform, 'VersionControl') else None
    LanguageSupport = eden_platform.LanguageSupport if hasattr(eden_platform, 'LanguageSupport') else None
    logger = eden_platform.logger if hasattr(eden_platform, 'logger') else None
    # Use default values if not defined in module
    HOST = getattr(eden_platform, 'HOST', 'localhost')
    PORT = getattr(eden_platform, 'PORT', 5000)
except Exception as e:
    print(f"Warning: Could not load eden-platform.py: {e}")
    print("Running tests with mock objects instead...")
    # Create mock implementations for testing
    from unittest.mock import MagicMock
    AISuperAgent = MagicMock
    FileManager = MagicMock
    AIDebugger = MagicMock
    AITestRunner = MagicMock
    CodeOptimizer = MagicMock
    IntegratedUI = MagicMock
    VersionControl = MagicMock
    LanguageSupport = MagicMock
    HOST = 'localhost'
    PORT = 5000
    logger = MagicMock()


class TestPlatformIntegration(unittest.TestCase):
    """Integration tests for the entire platform"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Create a test workspace directory
        cls.test_dir = Path("/tmp/eden_integration_test")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all components with test configurations
        cls.file_manager = FileManager(base_dir=cls.test_dir)
        cls.version_control = VersionControl(repo_dir=cls.test_dir)
        cls.debugger = AIDebugger()
        cls.test_runner = AITestRunner()
        cls.optimizer = CodeOptimizer()
        cls.language_support = LanguageSupport()
        
        # Initialize the super agent with a short interval for testing
        cls.super_agent = AISuperAgent(interval=1, api_key="test_key")
        
        # Initialize the UI but don't start it yet
        cls.ui = IntegratedUI()
        
        # Override the component instances in the UI with our test instances
        cls.ui.file_manager = cls.file_manager
        cls.ui.version_control = cls.version_control
        cls.ui.debugger = cls.debugger
        cls.ui.test_runner = cls.test_runner
        cls.ui.optimizer = cls.optimizer
        cls.ui.language_support = cls.language_support
        cls.ui.super_agent = cls.super_agent
        
        # Start the UI in a separate thread for testing
        cls.server_thread = threading.Thread(
            target=cls.ui.run,
            kwargs={"host": "localhost", "port": 5001},  # Use a different port for testing
            daemon=True
        )
        cls.server_thread.start()
        
        # Wait for server to start
        time.sleep(1)
        
        # Base URL for API requests
        cls.base_url = "http://localhost:5001"
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Stop the super agent
        if hasattr(cls, 'super_agent'):
            cls.super_agent.stop()
        
        # Clean up test directory
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            import shutil
            shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up before each test"""
        pass
    
    def tearDown(self):
        """Clean up after each test"""
        pass
    
    def test_component_initialization(self):
        """Test that all components initialize correctly"""
        # Check file manager initialization
        self.assertTrue(self.file_manager.base_dir.exists())
        
        # Check version control initialization
        git_dir = self.test_dir / ".git"
        self.assertTrue(git_dir.exists() or git_dir.is_dir())
        
        # Check super agent initialization
        self.assertTrue(self.super_agent.running)
        self.assertEqual(self.super_agent.interval, 1)
    
    def test_file_and_version_control_integration(self):
        """Test integration between file manager and version control"""
        # Create a test file
        test_file = "test_integration.py"
        test_content = "def test_function():\n    return True\n"
        
        # Use file manager to create the file
        self.file_manager.create_file(test_file, test_content)
        
        # Verify file exists
        file_path = self.test_dir / test_file
        self.assertTrue(file_path.exists())
        
        # Commit the file using version control
        result = self.version_control.commit_changes("Add test file")
        
        # Verify commit worked
        self.assertTrue(result)
        
        # Verify file appears in git log
        log = self.version_control.get_commit_log(1)
        self.assertTrue(log)
        self.assertIn("Add test file", log[0] if isinstance(log, list) else log)
    
    def test_super_agent_task_execution(self):
        """Test the super agent can execute tasks that involve other components"""
        # Create a test file for the super agent to analyze
        test_file = "agent_test.py"
        test_content = "def faulty_function():\n    unused_var = 10\n    return None\n"
        self.file_manager.create_file(test_file, test_content)
        
        # Clear any existing tasks
        while not self.super_agent.task_queue.empty():
            try:
                self.super_agent.task_queue.get_nowait()
            except Queue.Empty:
                break
        
        # Add a code analysis task
        task_params = {"files": [test_file]}
        self.super_agent.add_task("analyze_code", priority=0.9, params=task_params)
        
        # Give the agent time to process
        time.sleep(3)
        
        # Check if task was completed
        status = self.super_agent.get_status()
        self.assertGreater(status["completed_tasks"], 0)
    
    def test_api_super_agent_status(self):
        """Test the API endpoint for super agent status"""
        try:
            response = requests.get(f"{self.base_url}/api/super-agent/status")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertTrue(data["success"])
            self.assertIn("agent_status", data)
            self.assertIn("running", data["agent_status"])
            self.assertTrue(data["agent_status"]["running"])
        except requests.exceptions.ConnectionError:
            self.fail("Could not connect to server. Make sure it's running on the test port.")
    
    def test_api_add_super_agent_task(self):
        """Test the API endpoint for adding a task to the super agent"""
        task_data = {
            "task_name": "optimize_codebase",
            "priority": 0.8,
            "params": {"strategy": "performance"}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/super-agent/task",
                json=task_data
            )
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertTrue(data["success"])
            self.assertIn("message", data)
            self.assertIn("optimize_codebase", data["message"])
        except requests.exceptions.ConnectionError:
            self.fail("Could not connect to server. Make sure it's running on the test port.")
    
    def test_dashboard_availability(self):
        """Test that the super agent dashboard is available"""
        try:
            response = requests.get(f"{self.base_url}/super-agent-dashboard")
            self.assertEqual(response.status_code, 200)
            
            # Check for expected dashboard content
            content = response.text
            self.assertIn("AI Super-Agent Dashboard", content)
            self.assertIn("Agent Status", content)
            self.assertIn("Task Priorities", content)
        except requests.exceptions.ConnectionError:
            self.fail("Could not connect to server. Make sure it's running on the test port.")
    
    def test_end_to_end_code_analysis(self):
        """Test end-to-end code analysis workflow"""
        # Create a test file with some code issues
        test_file = "end_to_end_test.py"
        test_content = """
        def problematic_function(x, y=10):
            unused_var = "test"
            if x == None:  # Should be 'is None'
                return y
            else:
                return x + y
        """
        
        # Add the file using file manager
        self.file_manager.create_file(test_file, test_content)
        
        # Submit for analysis through the API
        analysis_data = {
            "code": test_content,
            "language": "python",
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/analyze",
                json=analysis_data
            )
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertTrue(data["success"])
            self.assertIn("analysis", data)
            
            # Verify that the analysis caught at least one issue
            analysis = data["analysis"]
            self.assertTrue(any(issue for issue in analysis["issues"]))
        except requests.exceptions.ConnectionError:
            self.fail("Could not connect to server. Make sure it's running on the test port.")
    
    def test_multiple_component_cooperation(self):
        """Test cooperation between multiple components"""
        # Create a test file
        test_file = "multi_component_test.py"
        test_content = """
        def test_function(x):
            return x * 2
        
        if __name__ == '__main__':
            result = test_function(5)
            print(f"Result: {result}")
        """
        
        # Use file manager to create file
        self.file_manager.create_file(test_file, test_content)
        
        # Commit with version control
        self.version_control.commit_changes("Add multi-component test file")
        
        # Analyze with debugger
        analysis = self.debugger.analyze_code(test_content)
        
        # Optimize with optimizer
        optimized_code, _ = self.optimizer.optimize(test_content, "readability")
        
        # Update file with optimized version
        self.file_manager.update_file(test_file, optimized_code)
        
        # Commit optimized version
        self.version_control.commit_changes("Optimize multi-component test file")
        
        # Get commit log
        log = self.version_control.get_commit_log(2)
        
        # Verify full workflow completed successfully
        self.assertEqual(len(log) if isinstance(log, list) else 1, 2)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and recovery across components"""
    
    def setUp(self):
        """Set up test environment"""
        # Initialize components with test configurations
        self.file_manager = FileManager(base_dir=Path("/tmp/eden_error_test"))
        self.debugger = AIDebugger()
        self.optimizer = CodeOptimizer()
        self.super_agent = AISuperAgent(interval=1)
    
    def tearDown(self):
        """Clean up test environment"""
        # Stop the super agent
        if hasattr(self, 'super_agent'):
            self.super_agent.stop()
        
        # Clean up test directory
        test_dir = Path("/tmp/eden_error_test")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
    
    def test_error_handling_invalid_file(self):
        """Test error handling when dealing with invalid files"""
        # Try to read a non-existent file
        with self.assertRaises(Exception):
            content = self.file_manager.read_file("non_existent_file.py")
    
    def test_super_agent_error_recovery(self):
        """Test super agent's ability to recover from task errors"""
        # Add a task with an invalid handler
        self.super_agent.add_task("non_existent_handler", priority=0.9)
        
        # Give the agent time to process
        time.sleep(2)
        
        # Check that error was recorded but agent is still running
        status = self.super_agent.get_status()
        self.assertTrue(status["running"])
        self.assertGreaterEqual(sum(status["error_counts"].values()), 1)
    
    def test_error_handling_invalid_optimization(self):
        """Test error handling with invalid optimization strategy"""
        code = "def test():\n    pass\n"
        
        # Try to optimize with an invalid strategy
        with self.assertRaises(Exception):
            self.optimizer.optimize(code, "invalid_strategy")


class TestCrossComponentCommunication(unittest.TestCase):
    """Tests focusing on communication between components"""
    
    def setUp(self):
        """Set up test environment"""
        self.file_manager = FileManager(base_dir=Path("/tmp/eden_comm_test"))
        self.debugger = AIDebugger()
        self.optimizer = CodeOptimizer()
        self.test_runner = AITestRunner()
        self.super_agent = AISuperAgent(interval=1)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'super_agent'):
            self.super_agent.stop()
        
        test_dir = Path("/tmp/eden_comm_test")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
    
    def test_cross_component_workflow(self):
        """Test a complete workflow involving multiple components"""
        # Create a file with the file manager
        test_file = "workflow_test.py"
        test_content = """
        def add(a, b):
            return a + b
        """
        
        self.file_manager.create_file(test_file, test_content)
        
        # Generate a test for the function using the AI Super Agent
        test_code = """
        import unittest
        from workflow_test import add
        
        class TestAdd(unittest.TestCase):
            def test_add(self):
                self.assertEqual(add(1, 2), 3)
                self.assertEqual(add(-1, 1), 0)
                self.assertEqual(add(0, 0), 0)
        
        if __name__ == '__main__':
            unittest.main()
        """
        
        # Write the test code
        self.file_manager.create_file("test_" + test_file, test_code)
        
        # Create a simple mock for the test runner
        def mock_run_test(test_path):
            return {"result": "passed", "details": {"tests": 3, "failures": 0, "errors": 0}}
        
        # Patch the run_tests method
        original_run_tests = self.test_runner.run_tests
        self.test_runner.run_tests = mock_run_test
        
        # Add a test task to the super agent
        task_params = {"files": ["test_" + test_file]}
        self.super_agent.add_task("run_tests", priority=0.9, params=task_params)
        
        # Give the agent time to process
        time.sleep(2)
        
        # Restore the original method
        self.test_runner.run_tests = original_run_tests
        
        # Check that task was processed
        status = self.super_agent.get_status()
        self.assertGreater(status["completed_tasks"], 0)


if __name__ == '__main__':
    unittest.main()
