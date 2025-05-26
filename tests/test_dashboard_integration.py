#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashboard Integration Test Suite for Eden AI Coding Platform

This module tests the integration between the AISuperAgent and the dashboard 
UI components, verifying that data flows correctly through all endpoints.
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

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import using a more robust approach to avoid module import issues
import importlib.util
spec = importlib.util.spec_from_file_location('eden_platform_module', os.path.join(parent_dir, 'eden-platform.py'))
eden_platform = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(eden_platform)
    # Extract needed classes
    AISuperAgent = eden_platform.AISuperAgent if hasattr(eden_platform, 'AISuperAgent') else None
    IntegratedUI = eden_platform.IntegratedUI if hasattr(eden_platform, 'IntegratedUI') else None
except Exception as e:
    print(f"Warning: Could not load eden-platform.py: {e}")
    print("Running tests with mock objects instead...")
    # Create mock implementations for testing
    from unittest.mock import MagicMock
    AISuperAgent = MagicMock
    IntegratedUI = MagicMock


class TestDashboardIntegration(unittest.TestCase):
    """Integration tests for the Dashboard UI and AISuperAgent"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Initialize super agent with a short interval for testing
        cls.super_agent = AISuperAgent(interval=1, api_key="test_key")
        
        # Initialize the UI but don't start the actual server
        cls.ui = IntegratedUI()
        cls.ui.super_agent = cls.super_agent
        
        # Start a testing server in a separate thread
        cls.server_thread = threading.Thread(
            target=cls.ui.run,
            kwargs={"host": "localhost", "port": 5002, "debug": False},
            daemon=True
        )
        cls.server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Base URL for API requests
        cls.base_url = "http://localhost:5002"
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Stop the super agent
        if hasattr(cls, 'super_agent'):
            cls.super_agent.stop()
    
    def test_dashboard_endpoint(self):
        """Test the dashboard endpoint returns the correct HTML"""
        try:
            response = requests.get(f"{self.base_url}/super-agent-dashboard")
            self.assertEqual(response.status_code, 200)
            
            # Verify the correct page is returned
            self.assertIn("AI Super-Agent Dashboard", response.text)
            self.assertIn("Agent Status", response.text)
            self.assertIn("Task Queue", response.text)
        except requests.exceptions.ConnectionError:
            self.fail("Could not connect to server. Make sure it's running on the test port.")
    
    def test_status_endpoint(self):
        """Test the status API endpoint returns the correct data"""
        try:
            response = requests.get(f"{self.base_url}/api/super-agent/status")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertTrue(data["success"])
            self.assertIn("agent_status", data)
            self.assertIn("running", data["agent_status"])
            self.assertTrue(data["agent_status"]["running"])
            
            # Verify the status contains all required fields
            required_fields = [
                "running", "queue_size", "completed_tasks", 
                "task_history", "error_counts", "performance_metrics"
            ]
            
            for field in required_fields:
                self.assertIn(field, data["agent_status"])
                
        except requests.exceptions.ConnectionError:
            self.fail("Could not connect to server. Make sure it's running on the test port.")
    
    def test_task_submission_and_performance(self):
        """Test adding a task and verifying it appears in performance metrics"""
        # Add a task
        task_data = {
            "task_name": "optimize_codebase",
            "priority": 0.8,
            "params": {"files": ["test.py"]}
        }
        
        try:
            # Add the task
            response = requests.post(
                f"{self.base_url}/api/super-agent/task",
                json=task_data
            )
            self.assertEqual(response.status_code, 200)
            
            # Wait for task to be processed
            time.sleep(3)
            
            # Check performance endpoint for the task
            response = requests.get(f"{self.base_url}/api/super-agent/performance")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertTrue(data["success"])
            
            # Verify task appears in history
            task_found = False
            if "task_history" in data:
                for task in data["task_history"]:
                    if task.get("name") == "optimize_codebase":
                        task_found = True
                        break
            
            self.assertTrue(task_found, "Added task not found in task history")
            
        except requests.exceptions.ConnectionError:
            self.fail("Could not connect to server. Make sure it's running on the test port.")
    
    def test_dashboard_data_flow(self):
        """Test that data flows correctly from the agent to the dashboard"""
        # First add some tasks to generate data
        task_names = ["analyze_code", "run_tests", "optimize_codebase"]
        
        for i, task_name in enumerate(task_names):
            task_data = {
                "task_name": task_name,
                "priority": 0.5 + (i * 0.1),  # Increasing priorities
                "params": {"index": i}
            }
            
            response = requests.post(
                f"{self.base_url}/api/super-agent/task",
                json=task_data
            )
            self.assertEqual(response.status_code, 200)
        
        # Wait for tasks to be processed
        time.sleep(5)
        
        # Fetch the dashboard HTML to check data is being displayed
        response = requests.get(f"{self.base_url}/super-agent-dashboard")
        self.assertEqual(response.status_code, 200)
        
        dashboard_html = response.text
        
        # Check for task names in the dashboard
        for task_name in task_names:
            self.assertIn(task_name, dashboard_html, 
                         f"Task {task_name} not found in dashboard HTML")
        
        # Check performance data is represented
        response = requests.get(f"{self.base_url}/api/super-agent/performance")
        performance_data = response.json()
        
        # Verify important performance data appears in the dashboard
        if "success_rate" in performance_data:
            success_rate_str = str(round(performance_data["success_rate"]))
            self.assertIn(success_rate_str, dashboard_html)
        
        if "system_load" in performance_data:
            # Convert to percentage and look for in dashboard
            load_pct = str(int(performance_data["system_load"] * 100))
            self.assertIn(load_pct, dashboard_html)


if __name__ == '__main__':
    unittest.main()
