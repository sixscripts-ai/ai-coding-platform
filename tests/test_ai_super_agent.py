#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for the AISuperAgent class

This module provides comprehensive testing for the AISuperAgent class,
ensuring that all task handlers, AI-driven methods, and utility functions
work correctly and integrate properly with the rest of the platform.
"""

import unittest
import sys
import os
import time
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from queue import Queue
try:
    from _thread import LockType as Lock
except ImportError:  # Fallback for older Python versions
    from threading import Lock
from collections import defaultdict

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Try to import the AISuperAgent class
try:
    from eden_platform import AISuperAgent, FileManager, AIDebugger, AITestRunner, CodeOptimizer, logger
except ImportError:
    # If module naming is different, try this import path
    try:
        from eden_platform import AISuperAgent, FileManager, AIDebugger, AITestRunner, CodeOptimizer, logger
    except ImportError:
        # If still not found, use direct import from file
        import importlib.util
        spec = importlib.util.spec_from_file_location("eden_platform", os.path.join(parent_dir, "eden-platform.py"))
        eden_platform = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eden_platform)
        # Now extract the needed classes
        AISuperAgent = eden_platform.AISuperAgent
        FileManager = eden_platform.FileManager
        AIDebugger = eden_platform.AIDebugger
        AITestRunner = eden_platform.AITestRunner
        CodeOptimizer = eden_platform.CodeOptimizer
        logger = eden_platform.logger


class TestAISuperAgent(unittest.TestCase):
    """Test cases for the AISuperAgent class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock dependencies
        self.mock_file_manager = MagicMock(spec=FileManager)
        self.mock_file_manager.base_dir = Path("/tmp/test_workspace")
        self.mock_file_manager.list_files.return_value = ["test1.py", "test2.py"]
        self.mock_file_manager.read_file.return_value = "def test_function():\n    return True\n"
        
        self.mock_ai_debugger = MagicMock(spec=AIDebugger)
        self.mock_ai_debugger.analyze_code.return_value = "No issues found"
        
        self.mock_ai_test_runner = MagicMock(spec=AITestRunner)
        self.mock_ai_test_runner.run_tests.return_value = "All tests passed"
        
        self.mock_code_optimizer = MagicMock(spec=CodeOptimizer)
        self.mock_code_optimizer.optimize.return_value = ("def test_function():\n    return True\n", ["No optimizations needed"])
        
        # Create test instance
        with patch('eden_platform_main.file_manager', self.mock_file_manager), \
             patch('eden_platform_main.ai_debugger', self.mock_ai_debugger), \
             patch('eden_platform_main.ai_test_runner', self.mock_ai_test_runner), \
             patch('eden_platform_main.code_optimizer', self.mock_code_optimizer):
            self.agent = AISuperAgent(interval=1, api_key="test_key")
            # Don't actually start the thread
            self.agent.running = True
            # Make sure queue is empty
            self.agent.task_queue = Queue()
    
    def tearDown(self):
        """Clean up after each test method"""
        self.agent.stop()
    
    def test_initialization(self):
        """Test proper initialization of the AISuperAgent"""
        self.assertEqual(self.agent.interval, 1)
        self.assertEqual(self.agent.api_key, "test_key")
        self.assertTrue(isinstance(self.agent.task_queue, Queue))
        self.assertTrue(isinstance(self.agent.lock, Lock))
        self.assertTrue(self.agent.running)
        self.assertTrue(isinstance(self.agent.task_handlers, dict))
        self.assertTrue(isinstance(self.agent.last_run_times, dict))
        self.assertTrue(isinstance(self.agent.performance_metrics, dict))
        self.assertTrue(isinstance(self.agent.priorities, dict))
        self.assertTrue(isinstance(self.agent.error_counts, defaultdict))
        self.assertTrue(isinstance(self.agent.learning_data, list))
    
    def test_add_task(self):
        """Test adding a task to the queue"""
        # Add a task
        self.agent.add_task("run_tests", priority=0.8, interval=300)
        
        # Check queue size
        self.assertEqual(self.agent.task_queue.qsize(), 1)
        
        # Get the task and check its properties
        task = self.agent.task_queue.get()
        self.assertEqual(task["name"], "run_tests")
        self.assertEqual(task["priority"], 0.8)
        self.assertEqual(task["interval"], 300)
        self.assertIn("scheduled_time", task)
        self.assertIn("params", task)
        
    def test_process_task_queue(self):
        """Test processing tasks from the queue"""
        # Add a test task
        self.agent.add_task("run_tests", priority=0.9)
        
        # Mock _execute_task method
        self.agent._execute_task = MagicMock(return_value=True)
        
        # Process the queue
        self.agent._process_task_queue()
        
        # Check that _execute_task was called
        self.agent._execute_task.assert_called_once()
        args, kwargs = self.agent._execute_task.call_args
        self.assertEqual(args[0]["name"], "run_tests")
    
    def test_execute_task(self):
        """Test execution of a task"""
        # Create a test task
        task = {
            "name": "run_tests",
            "priority": 0.9,
            "interval": 300,
            "scheduled_time": time.time(),
            "params": {}
        }
        
        # Mock the task handler
        self.agent._handle_testing = MagicMock(return_value=True)
        
        # Execute the task
        result = self.agent._execute_task(task)
        
        # Check the result and handler call
        self.assertTrue(result)
        self.agent._handle_testing.assert_called_once_with(task)
        
        # Check task metrics are updated
        self.assertIn("run_tests", self.agent.performance_metrics)
        self.assertIn("execution_count", self.agent.performance_metrics["run_tests"])
        self.assertIn("success_rate", self.agent.performance_metrics["run_tests"])
        
    def test_handle_testing(self):
        """Test the testing task handler"""
        # Create a test task
        task = {
            "name": "run_tests",
            "priority": 0.9,
            "interval": 300,
            "scheduled_time": time.time(),
            "params": {}
        }
        
        # Ensure the mocked test runner is accessible
        with patch('eden_platform_main.ai_test_runner', self.mock_ai_test_runner):
            # Execute the handler
            result = self.agent._handle_testing(task)
            
            # Check result
            self.assertTrue(result)
            # Verify mock was called
            self.mock_ai_test_runner.run_tests.assert_called_once()
    
    def test_handle_optimization(self):
        """Test the code optimization task handler"""
        # Create a test task
        task = {
            "name": "optimize_codebase",
            "priority": 0.7,
            "interval": 3600,
            "scheduled_time": time.time(),
            "params": {"files": ["test1.py"]}
        }
        
        # Ensure the mocked dependencies are accessible
        with patch('eden_platform_main.file_manager', self.mock_file_manager), \
             patch('eden_platform_main.code_optimizer', self.mock_code_optimizer):
            # Execute the handler
            result = self.agent._handle_optimization(task)
            
            # Check result
            self.assertFalse(result)  # Should be False because our mock returns the same code
            # Verify mocks were called
            self.mock_file_manager.read_file.assert_called_once_with("test1.py")
            self.mock_code_optimizer.optimize.assert_called_once()
    
    def test_handle_code_analysis(self):
        """Test the code analysis task handler"""
        # Create a test task
        task = {
            "name": "analyze_code",
            "priority": 0.6,
            "interval": 3600,
            "scheduled_time": time.time(),
            "params": {"files": ["test1.py"]}
        }
        
        # Ensure the mocked dependencies are accessible
        with patch('eden_platform_main.file_manager', self.mock_file_manager), \
             patch('eden_platform_main.ai_debugger', self.mock_ai_debugger):
            # Execute the handler
            result = self.agent._handle_code_analysis(task)
            
            # Check result
            self.assertTrue(result)
            # Verify mocks were called
            self.mock_file_manager.read_file.assert_called_once_with("test1.py")
            self.mock_ai_debugger.analyze_code.assert_called_once()
    
    def test_ai_assisted_optimization(self):
        """Test AI-assisted code optimization"""
        code = "def test_function():\n    return True\n"
        
        # Mock OpenAI API
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].text = "```python\ndef test_function():\n    return True  # Optimized\n```\n\n1. Added comment\n"
        mock_openai.Completion.create.return_value = mock_response
        
        # Test with mocked OpenAI
        with patch.dict('sys.modules', {'openai': mock_openai}):
            optimized_code, improvements = self.agent._ai_assisted_optimization(code)
            
            # Check results
            self.assertIn("# Optimized", optimized_code)
            self.assertEqual(len(improvements), 1)
            self.assertEqual(improvements[0], "Added comment")
            
            # Verify OpenAI was called with correct parameters
            mock_openai.Completion.create.assert_called_once()
            args, kwargs = mock_openai.Completion.create.call_args
            self.assertEqual(kwargs["model"], "gpt-4")
            self.assertTrue("optimize" in kwargs["prompt"])
            self.assertTrue(code in kwargs["prompt"])
    
    def test_get_system_load(self):
        """Test system load monitoring"""
        # Test with psutil available
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 75.0  # 75% CPU usage
        
        with patch.dict('sys.modules', {'psutil': mock_psutil}):
            load = self.agent._get_system_load()
            self.assertEqual(load, 0.75)  # 75% should convert to 0.75
            mock_psutil.cpu_percent.assert_called_once()
        
        # Test without psutil
        with patch.dict('sys.modules', {'psutil': None}):
            load = self.agent._get_system_load()
            self.assertEqual(load, 0.5)  # Default value


if __name__ == '__main__':
    unittest.main()
