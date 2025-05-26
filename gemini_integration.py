import google.generativeai as genai
import os
import json
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from flask import Markup
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiAnalytics:
    """Gemini AI-powered analytics for the Eden AI Coding Platform"""
    
    def __init__(self, api_key=None):
        """Initialize Gemini Analytics with API key"""
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', 'AIzaSyC6R6I6eT0JHrUiw_W5vBJk2NxJMZ4uDzk')
        self.analytics_data = {}
        
        # Model configuration settings
        self.models = {
            'flash': {
                'id': 'gemini-2.0-flash',
                'description': 'Fastest model for text generation, ideal for quick analysis',
                'max_tokens': 8192,
                'strengths': ['speed', 'conciseness', 'efficiency'],
                'instance': None
            },
            'pro': {
                'id': 'gemini-2.0-pro',
                'description': 'Balanced model for advanced reasoning and complex code analysis',
                'max_tokens': 32768,
                'strengths': ['reasoning', 'code analysis', 'nuanced responses'],
                'instance': None
            },
            'vision': {
                'id': 'gemini-2.0-vision',
                'description': 'Vision-capable model for analyzing charts, diagrams, and visual content',
                'max_tokens': 16384,
                'strengths': ['image analysis', 'visual understanding', 'chart interpretation'],
                'instance': None
            }
        }
        
        # Task to model mapping for automatic model selection
        self.task_model_mapping = {
            'code_analysis': 'pro',
            'performance_metrics': 'flash',
            'security_analysis': 'pro',
            'documentation': 'pro',
            'quick_insights': 'flash',
            'visualization': 'vision',
            'chart_analysis': 'vision',
            'real_time_feedback': 'flash',
            'tutorial_generation': 'pro'
        }
        
        # Usage tracking
        self.model_usage = {model: 0 for model in self.models}
        self.setup()
        
    def setup(self):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            
            # Initialize all model instances
            for model_key, model_config in self.models.items():
                self.models[model_key]['instance'] = genai.GenerativeModel(model_config['id'])
            
            # Set default model as flash for backward compatibility
            self.model = self.models['flash']['instance']
            
            logger.info("Gemini API configured successfully with multiple models")
            self.generate_sample_analytics_data()
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {str(e)}")
            
    def select_model(self, task=None, manual_selection=None):
        """Select the appropriate Gemini model based on task or manual selection
        
        Args:
            task: The type of task being performed (e.g., 'code_analysis', 'documentation')
            manual_selection: Manual override to select a specific model ('flash', 'pro', 'vision')
            
        Returns:
            The selected Gemini model instance
        """
        try:
            # 1. Manual selection takes precedence if provided and valid
            if manual_selection and manual_selection in self.models:
                selected = manual_selection
                logger.info(f"Manually selected Gemini model: {selected}")
            
            # 2. Task-based selection if task is provided and has a mapping
            elif task and task in self.task_model_mapping:
                selected = self.task_model_mapping[task]
                logger.info(f"Task-based Gemini model selection for '{task}': {selected}")
            
            # 3. Default to 'flash' model if no valid selection criteria
            else:
                selected = 'flash'  # Default to fastest model
                logger.info(f"Defaulting to Gemini model: {selected}")
                
            # Update usage statistics
            self.model_usage[selected] += 1
            
            # Return the appropriate model instance
            return self.models[selected]['instance']
            
        except Exception as e:
            logger.error(f"Error selecting model: {str(e)}, defaulting to 'flash'")
            return self.models['flash']['instance']  # Fallback to default model
    
    def get_model_usage_stats(self):
        """Get statistics about model usage"""
        total = sum(self.model_usage.values()) or 1  # Avoid division by zero
        stats = {
            'total_calls': total,
            'model_breakdown': {
                model: {
                    'count': count,
                    'percentage': round((count / total) * 100, 2)
                } for model, count in self.model_usage.items()
            },
            'recommended_model': max(self.model_usage, key=self.model_usage.get)
        }
        return stats
    
    def generate_documentation(self, code, language="python", doc_type="function"):
        """Generate documentation for code using Gemini AI
        
        Args:
            code: The source code to document
            language: Programming language of the code
            doc_type: Type of documentation to generate (function, class, module, etc.)
            
        Returns:
            Dictionary with generated documentation and metadata
        """
        try:
            # Select the appropriate model for documentation generation
            model = self.select_model(task='documentation')
            
            prompt = f"""Generate comprehensive {doc_type} documentation for this {language} code.
            Follow standard {language} documentation conventions.
            Include:
            1. Purpose and overview
            2. Parameters/arguments with types and descriptions
            3. Return values with types and descriptions
            4. Exceptions/errors that might be raised
            5. Usage examples
            6. Notes on edge cases or limitations
            
            Here is the {language} code to document:
            ```{language}
            {code}
            ```
            
            Return only the documentation in the standard format for {language}."""
            
            # Generate documentation
            response = model.generate_content(prompt)
            
            # Process and format the documentation
            doc_content = response.text.strip()
            
            # Return formatted documentation and metadata
            return {
                'success': True,
                'documentation': doc_content,
                'language': language,
                'doc_type': doc_type,
                'model_used': 'pro',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def generate_interactive_tutorial(self, topic, language="python", skill_level="intermediate"):
        """Generate an interactive tutorial for learning code concepts
        
        Args:
            topic: The programming concept or feature to create a tutorial for
            language: Programming language for the tutorial
            skill_level: Target skill level (beginner, intermediate, advanced)
            
        Returns:
            Dictionary with tutorial content structured as interactive steps
        """
        try:
            # Select the appropriate model for tutorial generation
            model = self.select_model(task='tutorial_generation')
            
            prompt = f"""Create an interactive, step-by-step tutorial for {skill_level} developers to learn about '{topic}' in {language}.
            
            Structure the tutorial as follows:
            1. Title and overview of what will be learned
            2. Prerequisites (knowledge, tools, etc.)
            3. 5-8 sequential interactive steps where each step includes:
               - Clear explanation of the concept
               - Code example showing the concept
               - An exercise for the learner to practice
               - Expected output or solution for verification
            4. Summary of what was learned
            5. Further resources or next steps
            
            Make the tutorial engaging, practical, and focused on hands-on learning.
            Ensure code examples are correct, efficient, and follow best practices for {language}."""
            
            # Generate tutorial content
            response = model.generate_content(prompt)
            
            # Process the response and structure as interactive tutorial
            tutorial_text = response.text
            
            # Structure the tutorial into sections
            tutorial = {
                'success': True,
                'title': f"Interactive {topic} Tutorial for {language}",
                'skill_level': skill_level,
                'content': tutorial_text,
                'model_used': 'pro',
                'timestamp': datetime.now().isoformat(),
                'interactive_elements': True
            }
            
            return tutorial
            
        except Exception as e:
            logger.error(f"Error generating tutorial: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_sample_analytics_data(self):
        """Generate sample analytics data for dashboard visualization"""
        # Generate time-series data for the past 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(31)]
        
        # User activity metrics
        self.analytics_data['user_activity'] = {
            'dates': dates,
            'code_submissions': [random.randint(50, 200) for _ in dates],
            'optimizations': [random.randint(20, 100) for _ in dates],
            'tests_run': [random.randint(100, 300) for _ in dates],
            'ai_interactions': [random.randint(150, 400) for _ in dates]
        }
        
        # Performance metrics
        self.analytics_data['performance'] = {
            'response_times': [random.uniform(0.1, 0.5) for _ in dates],
            'error_rates': [random.uniform(0.01, 0.05) for _ in dates],
            'optimization_scores': [random.uniform(70, 95) for _ in dates],
            'test_coverage': [random.uniform(75, 98) for _ in dates],
        }
        
        # Language distribution
        self.analytics_data['languages'] = {
            'Python': 45,
            'JavaScript': 25,
            'Java': 15,
            'C++': 8,
            'Go': 4,
            'Rust': 3
        }
        
        # AI task distribution
        self.analytics_data['ai_tasks'] = {
            'Code Optimization': 35,
            'Bug Detection': 25,
            'Test Generation': 20,
            'Documentation': 10,
            'Refactoring': 10
        }
        
        logger.info("Sample analytics data generated successfully")
    
    async def analyze_code(self, code, language="python"):
        """Analyze code using Gemini AI"""
        try:
            prompt = f"""Analyze this {language} code and provide insights. Focus on:
            1. Code quality and best practices
            2. Potential performance issues
            3. Security vulnerabilities
            4. Suggested improvements
            
            Code to analyze:
            ```{language}
            {code}
            ```
            
            Provide a structured response with clear sections."""
            
            response = await self.model.generate_content_async(prompt)
            return {
                'success': True,
                'analysis': response.text,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing code with Gemini: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    async def analyze_performance(self, code, language="python"):
        """Analyze code performance using Gemini AI"""
        try:
            prompt = f"""Perform a detailed performance analysis of this {language} code.
            Focus specifically on:
            1. Time complexity analysis (Big O notation)
            2. Space complexity analysis
            3. Bottlenecks and performance hotspots
            4. Resource usage inefficiencies
            5. Concrete optimization recommendations with code examples
            
            Code to analyze:
            ```{language}
            {code}
            ```
            
            Provide specific, actionable recommendations for performance improvements.
            Format your response in clear sections with headers."""
            
            response = await self.model.generate_content_async(prompt)
            return {
                'success': True,
                'analysis': response.text,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing code performance with Gemini: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    async def analyze_security(self, code, language="python"):
        """Analyze code security using Gemini AI"""
        try:
            prompt = f"""Perform a comprehensive security analysis of this {language} code.
            Focus specifically on:
            1. Security vulnerabilities (e.g., OWASP Top 10 if applicable)
            2. Input validation issues
            3. Authentication and authorization weaknesses
            4. Data handling and privacy concerns
            5. Secure coding best practices violations
            6. Concrete security recommendations with code examples
            
            Code to analyze:
            ```{language}
            {code}
            ```
            
            Provide specific, actionable recommendations for security improvements.
            Format your response in clear sections with headers.
            Include severity levels (Critical, High, Medium, Low) for each issue identified."""
            
            response = await self.model.generate_content_async(prompt)
            return {
                'success': True,
                'analysis': response.text,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing code security with Gemini: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def get_code_performance_visualization(self, code, language="python"):
        """Generate performance visualization data for code"""
        try:
            # Generate simulated performance metrics based on code complexity
            # In a real implementation, this would use actual performance profiling
            code_lines = len(code.splitlines())
            complexity_score = min(100, max(10, code_lines / 10))  # Simple heuristic
            
            # Generate metrics for visualization
            metrics = {
                'time_complexity': random.choice(['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)', 'O(2ⁿ)']),
                'memory_usage': round(random.uniform(5, 100), 2),  # MB
                'cpu_usage': round(random.uniform(10, 90), 2),  # %
                'execution_time': round(random.uniform(0.001, 2.0), 3),  # seconds
                'throughput': round(random.uniform(100, 10000), 0),  # ops/sec
                'complexity_score': round(complexity_score, 2)
            }
            
            # Generate visualization data for Plotly charts
            visualization_data = {
                'resource_usage': {
                    'data': [
                        {
                            'values': [metrics['cpu_usage'], metrics['memory_usage'], 100 - metrics['cpu_usage'] - metrics['memory_usage']],
                            'labels': ['CPU Usage (%)', 'Memory Usage (MB)', 'Available Resources'],
                            'type': 'pie',
                            'hole': 0.4,
                            'textinfo': 'label+percent',
                            'textposition': 'outside',
                            'marker': {
                                'colors': ['#3498db', '#e74c3c', '#ecf0f1']
                            }
                        }
                    ],
                    'layout': {
                        'title': 'Resource Usage Distribution',
                        'height': 400,
                        'width': 500
                    }
                },
                'performance_metrics': {
                    'data': [
                        {
                            'type': 'bar',
                            'x': ['Complexity', 'Throughput (K ops/s)', 'Execution Time (ms)'],
                            'y': [metrics['complexity_score'], metrics['throughput']/100, metrics['execution_time']*1000],
                            'marker': {
                                'color': ['#9b59b6', '#2ecc71', '#f39c12']
                            }
                        }
                    ],
                    'layout': {
                        'title': 'Key Performance Indicators',
                        'height': 400,
                        'width': 500
                    }
                }
            }
            
            # Add optimization recommendations
            if metrics['complexity_score'] > 50:
                recommendation = "High complexity detected. Consider refactoring for improved performance."
            elif metrics['memory_usage'] > 70:
                recommendation = "High memory usage. Consider optimizing data structures or memory management."
            elif metrics['execution_time'] > 1.0:
                recommendation = "Long execution time. Consider algorithmic improvements or caching strategies."
            else:
                recommendation = "Code performance is within acceptable parameters. No immediate optimization needed."
            
            return {
                'metrics': metrics,
                'visualizations': visualization_data,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating code performance visualization: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def generate_insights(self, data_type, timeframe="week"):
        """Generate AI-powered insights based on analytics data"""
        try:
            # Prepare context data based on data_type
            context = json.dumps(self.analytics_data.get(data_type, {}))
            
            prompt = f"""Based on the following {data_type} analytics data for the last {timeframe}, 
            generate 3-5 key insights and actionable recommendations for improving development practices:
            
            {context}
            
            Provide specific, data-driven insights that would be valuable for developers and managers."""
            
            response = await self.model.generate_content_async(prompt)
            return {
                'success': True,
                'insights': response.text,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating insights with Gemini: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_activity_chart(self):
        """Generate HTML for activity chart using Plotly"""
        try:
            data = self.analytics_data['user_activity']
            df = pd.DataFrame({
                'Date': data['dates'],
                'Code Submissions': data['code_submissions'],
                'Optimizations': data['optimizations'],
                'Tests Run': data['tests_run'],
                'AI Interactions': data['ai_interactions']
            })
            
            # Melt the dataframe for easier plotting
            df_melted = pd.melt(df, id_vars=['Date'], var_name='Metric', value_name='Count')
            
            # Create figure
            fig = px.line(df_melted, x='Date', y='Count', color='Metric', 
                         title='Platform Activity Over Time',
                         template='plotly_white')
            
            # Update layout
            fig.update_layout(
                autosize=True,
                height=400,
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return Markup(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            logger.error(f"Error generating activity chart: {str(e)}")
            return f"<div class='error'>Error generating chart: {str(e)}</div>"
    
    def get_language_distribution_chart(self):
        """Generate HTML for language distribution chart"""
        try:
            languages = self.analytics_data['languages']
            labels = list(languages.keys())
            values = list(languages.values())
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                marker_colors=px.colors.qualitative.Plotly
            )])
            
            fig.update_layout(
                title_text='Programming Language Distribution',
                autosize=True,
                height=350,
                margin=dict(l=0, r=0, b=0, t=40),
            )
            
            return Markup(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            logger.error(f"Error generating language chart: {str(e)}")
            return f"<div class='error'>Error generating chart: {str(e)}</div>"
    
    def get_performance_metrics_chart(self):
        """Generate HTML for performance metrics chart"""
        try:
            data = self.analytics_data['performance']
            dates = self.analytics_data['user_activity']['dates']
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=dates,
                y=data['response_times'],
                name="Response Time (s)",
                line=dict(color='#636EFA')
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=data['error_rates'],
                name="Error Rate",
                line=dict(color='#EF553B'),
                yaxis="y2"
            ))
            
            # Create axis objects
            fig.update_layout(
                title="API Performance Metrics",
                xaxis=dict(title="Date"),
                yaxis=dict(
                    title="Response Time (s)",
                    titlefont=dict(color="#636EFA"),
                    tickfont=dict(color="#636EFA")
                ),
                yaxis2=dict(
                    title="Error Rate",
                    titlefont=dict(color="#EF553B"),
                    tickfont=dict(color="#EF553B"),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                autosize=True,
                height=350,
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return Markup(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            logger.error(f"Error generating performance chart: {str(e)}")
            return f"<div class='error'>Error generating chart: {str(e)}</div>"
    
    def get_ai_task_distribution_chart(self):
        """Generate HTML for AI task distribution chart"""
        try:
            tasks = self.analytics_data['ai_tasks']
            labels = list(tasks.keys())
            values = list(tasks.values())
            
            fig = go.Figure([go.Bar(
                x=labels,
                y=values,
                text=values,
                textposition='auto',
                marker_color='#2c3e50'
            )])
            
            fig.update_layout(
                title_text='AI Task Distribution',
                xaxis_title="Task Type",
                yaxis_title="Percentage",
                autosize=True,
                height=350,
                margin=dict(l=0, r=0, b=0, t=40),
            )
            
            return Markup(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            logger.error(f"Error generating AI task chart: {str(e)}")
            return f"<div class='error'>Error generating chart: {str(e)}</div>"
    
    def get_sample_ai_insights(self):
        """Generate sample AI insights for the dashboard"""
        insights = [
            {
                "title": "Code Quality Trend",
                "content": "Code quality metrics have improved by 23% over the past month, with a notable decrease in security vulnerabilities and technical debt."
            },
            {
                "title": "Performance Optimization",
                "content": "AI analysis suggests that refactoring the database query methods could improve overall system performance by up to 35%."
            },
            {
                "title": "Usage Patterns",
                "content": "User interaction data shows most developers use the code completion feature 3x more frequently than documentation generation."
            },
            {
                "title": "Security Enhancement",
                "content": "Automatic security scans identified 3 potential vulnerabilities in third-party dependencies. Consider updating to newer versions."
            },
            {
                "title": "Resource Optimization",
                "content": "The analysis indicates that 65% of performance issues stem from inefficient database queries. Implementing the suggested query optimizations could yield a 40% performance improvement."
            }
        ]
        return insights
    
    def generate_code_dependency_visualization(self, code_files):
        """Generate interactive 3D visualization of code dependencies
        
        Args:
            code_files: Dictionary mapping file paths to their content
            
        Returns:
            HTML string with interactive 3D visualization
        """
        try:
            # Select the appropriate model for visualization
            model = self.select_model(task='visualization')
        except Exception as e:
            logger.error(f"Error selecting model for visualization: {str(e)}")
            raise
            
        try:
            
            # Extract imports and function calls from code files
            dependencies = []
            nodes = []
            file_idx_map = {}
            
            # Process each file to extract nodes
            for i, (file_path, content) in enumerate(code_files.items()):
                file_idx_map[file_path] = i
                # Add file node
                nodes.append({
                    'id': i,
                    'name': file_path.split('/')[-1],
                    'type': 'file',
                    'size': len(content),
                    'file_path': file_path
                })
                
                # Use Gemini AI to analyze imports and dependencies
                prompt = f"""Analyze this Python code and extract:
                1. All import statements
                2. All function definitions
                3. All class definitions
                4. All function calls to other functions inside this file
                
                Format the response as a JSON with keys: 'imports', 'functions', 'classes', and 'internal_calls'
                
                Code to analyze:
                ```python
                {content}
                ```"""
                
                response = model.generate_content(prompt)
                
                try:
                    # Parse the response to extract dependency information
                    analysis = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
                    
                    # Process imports to find external dependencies
                    for imp in analysis.get('imports', []):
                        dependencies.append({
                            'source': i,
                            'target': -1,  # External dependency
                            'type': 'import',
                            'name': imp
                        })
                    
                    # Add function and class nodes
                    for func in analysis.get('functions', []):
                        nodes.append({
                            'id': len(nodes),
                            'name': func,
                            'type': 'function',
                            'parent_file': i,
                            'file_path': file_path
                        })
                        
                    for cls in analysis.get('classes', []):
                        nodes.append({
                            'id': len(nodes),
                            'name': cls,
                            'type': 'class',
                            'parent_file': i,
                            'file_path': file_path
                        })
                        
                except Exception as e:
                    logger.error(f"Error parsing dependency analysis response: {str(e)}")
            
            # Find cross-file dependencies by looking for imported module names in file contents
            for i, (file_path, content) in enumerate(code_files.items()):
                for j, (other_file, _) in enumerate(code_files.items()):
                    if i != j:
                        other_file_name = other_file.split('/')[-1].split('.')[0]
                        if other_file_name in content:
                            dependencies.append({
                                'source': i,
                                'target': j,
                                'type': 'file_dependency',
                                'strength': content.count(other_file_name)
                            })
            
            # Generate 3D visualization using Plotly
            x_nodes = [i % 5 * 3 for i in range(len(nodes))]  # Distribute nodes in 3D space
            y_nodes = [i // 5 * 3 for i in range(len(nodes))]
            z_nodes = [(node['type'] == 'file') * 2 for node in nodes]  # Files at z=2, functions/classes at z=0
            
            node_colors = [
                '#E91E63' if node['type'] == 'file' else 
                '#2196F3' if node['type'] == 'function' else 
                '#4CAF50' for node in nodes
            ]
            
            # Create the 3D network visualization
            fig = go.Figure()
            
            # Add nodes
            fig.add_trace(go.Scatter3d(
                x=x_nodes,
                y=y_nodes,
                z=z_nodes,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=node_colors,
                    opacity=0.8
                ),
                text=[node['name'] for node in nodes],
                textposition='top center',
                hoverinfo='text',
                hovertext=[f"Type: {node['type']}\nName: {node['name']}\nPath: {node.get('file_path', 'N/A')}" for node in nodes]
            ))
            
            # Add edges (dependencies)
            edge_x = []
            edge_y = []
            edge_z = []
            for dep in dependencies:
                if dep['target'] >= 0:  # Skip external dependencies for clarity
                    src_idx, tgt_idx = dep['source'], dep['target']
                    # Add line segment for each edge
                    edge_x.extend([x_nodes[src_idx], x_nodes[tgt_idx], None])
                    edge_y.extend([y_nodes[src_idx], y_nodes[tgt_idx], None])
                    edge_z.extend([z_nodes[src_idx], z_nodes[tgt_idx], None])
            
            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(
                    color='rgba(125, 125, 125, 0.5)',
                    width=1
                ),
                hoverinfo='none'
            ))
            
            # Layout configuration
            fig.update_layout(
                title='Interactive 3D Code Dependency Visualization',
                scene=dict(
                    xaxis=dict(showticklabels=False, title=''),
                    yaxis=dict(showticklabels=False, title=''),
                    zaxis=dict(showticklabels=False, title=''),
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                height=700,
                scene_camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
            
            # Add legend trace for node types
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=0),
                name='File',
                showlegend=True
            ))
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=0),
                name='Function',
                showlegend=True
            ))
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=0),
                name='Class',
                showlegend=True
            ))
            
            # Return HTML with full interactive visualization
            dep_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Add additional insights about the codebase structure
            insights_prompt = f"""Based on the code dependency analysis, generate 3-5 insights about:
            1. The modularity of the codebase
            2. Potential areas of high coupling that might benefit from refactoring
            3. The overall architecture and design patterns evident
            
            Format the response as a JSON with an 'insights' key containing an array of insight strings."""
            
            insights_response = model.generate_content(insights_prompt)
            try:
                insights_data = json.loads(insights_response.text.strip().replace('```json', '').replace('```', ''))
                insights_html = '<div class="code-insights"><h3>AI-Generated Code Structure Insights</h3><ul>'
                for insight in insights_data.get('insights', []):
                    insights_html += f'<li>{insight}</li>'
                insights_html += '</ul></div>'
                
                # Combine visualization with insights
                return dep_html + insights_html
                
            except Exception as e:
                logger.error(f"Error parsing insights: {str(e)}")
                return dep_html
                
        except Exception as e:
            logger.error(f"Error generating code dependency visualization: {str(e)}")
            return f"<div class='error'>Error generating visualization: {str(e)}</div>"
    
    def generate_realtime_performance_dashboard(self, metrics_data=None):
        """Generate real-time performance dashboard visualization
        
        Args:
            metrics_data: Dictionary containing performance metrics data (optional)
            
        Returns:
            HTML string with real-time performance dashboard
        """
        try:
            # If no data provided, generate sample data
            if not metrics_data:
                # Generate sample real-time data with realistic metrics
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=3)
                times = [(start_time + timedelta(minutes=i*5)).strftime('%H:%M') for i in range(37)]  # 3 hours in 5-min intervals
                
                # CPU utilization (%)
                cpu_data = [round(random.uniform(5, 85), 1) for _ in range(37)]
                # Add some realistic patterns
                for i in range(1, 37):
                    # Smooth transitions
                    cpu_data[i] = min(95, max(1, cpu_data[i-1] + random.uniform(-15, 15)))
                    # Add occasional spikes
                    if random.random() > 0.9:
                        cpu_data[i] = min(98, cpu_data[i] + random.uniform(20, 40))
                
                # Memory usage (MB)
                base_memory = 500  # Base memory usage
                memory_data = [round(base_memory + random.uniform(0, 1500), 1) for _ in range(37)]
                # Ensure memory generally increases over time with occasional GC
                for i in range(1, 37):
                    # Generally increases
                    memory_data[i] = memory_data[i-1] + random.uniform(-100, 300)
                    # Occasional garbage collection
                    if random.random() > 0.9:
                        memory_data[i] = max(base_memory, memory_data[i] - random.uniform(400, 800))
                
                # Response time (ms)
                response_data = [round(random.uniform(50, 300), 1) for _ in range(37)]
                # Make response time correlate somewhat with CPU
                for i in range(37):
                    response_data[i] += (cpu_data[i] / 100) * random.uniform(50, 200)
                
                # Request count (per 5 min)
                request_data = [int(random.uniform(10, 200)) for _ in range(37)]
                # Add pattern of increasing usage during "business hours"
                hour_factors = {0: 0.3, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2, 
                               6: 0.4, 7: 0.6, 8: 0.8, 9: 1.0, 10: 1.2, 11: 1.3,
                               12: 1.2, 13: 1.1, 14: 1.0, 15: 1.1, 16: 1.0, 17: 0.9,
                               18: 0.8, 19: 0.7, 20: 0.6, 21: 0.5, 22: 0.4, 23: 0.3}
                for i in range(37):
                    hour = int(times[i].split(':')[0])
                    request_data[i] = int(request_data[i] * hour_factors.get(hour, 1.0))
                
                # Error rate (%)
                error_data = [round(random.uniform(0, 0.5) + (cpu_data[i]/100)*random.uniform(0, 5), 2) for i in range(37)]
                # Spike error rate when CPU is very high
                for i in range(37):
                    if cpu_data[i] > 90:
                        error_data[i] += random.uniform(5, 15)
                
                # Compile metrics data
                metrics_data = {
                    'timestamps': times,
                    'cpu_utilization': cpu_data,
                    'memory_usage': memory_data,
                    'response_time': response_data,
                    'request_count': request_data,
                    'error_rate': error_data
                }
            
            # Create a multiple subplot figure
            fig = go.Figure()
            
            # CPU Utilization Time Series
            fig.add_trace(go.Scatter(
                x=metrics_data['timestamps'],
                y=metrics_data['cpu_utilization'],
                mode='lines+markers',
                name='CPU Utilization (%)',
                line=dict(color='#E91E63', width=2),
                hovertemplate='%{y:.1f}%<extra></extra>'
            ))
            
            # Memory Usage
            fig.add_trace(go.Scatter(
                x=metrics_data['timestamps'],
                y=metrics_data['memory_usage'],
                mode='lines+markers',
                name='Memory Usage (MB)',
                line=dict(color='#2196F3', width=2),
                visible='legendonly',  # Hide initially but available in legend
                hovertemplate='%{y:.1f} MB<extra></extra>'
            ))
            
            # Response Time
            fig.add_trace(go.Scatter(
                x=metrics_data['timestamps'],
                y=metrics_data['response_time'],
                mode='lines+markers',
                name='Response Time (ms)',
                line=dict(color='#4CAF50', width=2),
                visible='legendonly',  # Hide initially but available in legend
                hovertemplate='%{y:.1f} ms<extra></extra>'
            ))
            
            # Request Count
            fig.add_trace(go.Bar(
                x=metrics_data['timestamps'],
                y=metrics_data['request_count'],
                name='Requests (per 5 min)',
                marker_color='#FF9800',
                opacity=0.7,
                visible='legendonly',  # Hide initially but available in legend
                hovertemplate='%{y} requests<extra></extra>'
            ))
            
            # Error Rate
            fig.add_trace(go.Scatter(
                x=metrics_data['timestamps'],
                y=metrics_data['error_rate'],
                mode='lines+markers',
                name='Error Rate (%)',
                line=dict(color='#F44336', width=2),
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
            
            # Add range slider and buttons for time selection
            fig.update_layout(
                title='Real-time Performance Monitoring',
                height=500,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                margin=dict(l=10, r=10, t=70, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(240,240,240,0.5)',
                font=dict(family='Arial', size=12),
                xaxis=dict(
                    title='Time',
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.2)'
                ),
                yaxis=dict(
                    title='Value',
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.2)'
                ),
                updatemenus=[
                    dict(
                        type='buttons',
                        direction='right',
                        buttons=[
                            dict(method='relayout',
                                args=['xaxis.autorange', True],
                                label='Reset Zoom'),
                            dict(method='update',
                                args=[{'visible': [True, True, True, True, True]}],
                                label='Show All'),
                            dict(method='update',
                                args=[{'visible': [True, False, False, False, True]}],
                                label='Performance Focus')
                        ],
                        pad={'r': 10, 't': 10},
                        showactive=True,
                        x=0.1,
                        xanchor='left',
                        y=1.1,
                        yanchor='top'
                    ),
                ],
                hovermode='x unified'
            )
            
            # Generate the HTML for the dashboard
            dashboard_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            # Create a gauge chart for current system status
            current_cpu = metrics_data['cpu_utilization'][-1]
            cpu_color = '#4CAF50' if current_cpu < 50 else '#FF9800' if current_cpu < 80 else '#F44336'
            
            gauge_fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=current_cpu,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current CPU Utilization"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': cpu_color},
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(76, 175, 80, 0.2)'},
                        {'range': [50, 80], 'color': 'rgba(255, 152, 0, 0.3)'},
                        {'range': [80, 100], 'color': 'rgba(244, 67, 54, 0.4)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            gauge_fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            gauge_html = gauge_fig.to_html(full_html=False, include_plotlyjs=False)
            
            # Add a summary of system health
            current_memory = metrics_data['memory_usage'][-1]
            current_response = metrics_data['response_time'][-1]
            current_error = metrics_data['error_rate'][-1]
            
            health_status = "Good" if current_cpu < 70 and current_error < 1 else "Warning" if current_cpu < 85 and current_error < 5 else "Critical"
            health_color = "#4CAF50" if health_status == "Good" else "#FF9800" if health_status == "Warning" else "#F44336"
            
            summary_html = f"""
            <div style="margin-top: 20px; padding: 15px; background-color: rgba(240,240,240,0.5); border-radius: 5px;">
                <h3 style="margin-top: 0; color: {health_color};">System Health: {health_status}</h3>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 150px; margin: 10px;">
                        <h4>Current Metrics</h4>
                        <ul style="list-style-type: none; padding-left: 0;">
                            <li><strong>CPU:</strong> {current_cpu:.1f}%</li>
                            <li><strong>Memory:</strong> {current_memory:.1f} MB</li>
                            <li><strong>Response Time:</strong> {current_response:.1f} ms</li>
                            <li><strong>Error Rate:</strong> {current_error:.2f}%</li>
                        </ul>
                    </div>
                    <div style="flex: 1; min-width: 150px; margin: 10px;">
                        <h4>Recommendations</h4>
                        <ul>
            """
            
            # Generate recommendations based on metrics
            if health_status == "Good":
                summary_html += """
                            <li>System is operating normally.</li>
                            <li>Continue monitoring for any changes.</li>
                            <li>Consider optimizing request patterns during peak hours.</li>
                        </ul>
                    </div>
                </div>
            </div>"""
            elif health_status == "Warning":
                summary_html += """
                            <li>Monitor increasing resource usage.</li>
                            <li>Consider scaling resources if trends continue upward.</li>
                            <li>Investigate response time increases.</li>
                        </ul>
                    </div>
                </div>
            </div>"""
            else:
                summary_html += """
                            <li><strong>Critical: Immediate attention required!</strong></li>
                            <li>Scale up resources or reduce load immediately.</li>
                            <li>Investigate error rate spikes and CPU bottlenecks.</li>
                            <li>Consider enabling auto-scaling policies.</li>
                        </ul>
                    </div>
                </div>
            </div>"""
            
            # Combine all components
            dashboard_with_gauge = f"""
            <div class="performance-dashboard">
                <div style="display: flex; flex-wrap: wrap;">
                    <div style="flex: 3; min-width: 300px;">
                        {dashboard_html}
                    </div>
                    <div style="flex: 1; min-width: 250px;">
                        {gauge_html}
                    </div>
                </div>
                {summary_html}
                <div style="font-size: 12px; text-align: right; margin-top: 10px; color: #666;">
                    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            """
            
            return dashboard_with_gauge
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {str(e)}")
            return f"<div class='error'>Error generating performance dashboard: {str(e)}</div>"

    def get_sample_ai_insights(self):
        """Generate sample AI insights for the dashboard"""
        insights = [
            {
                "title": "Code Quality Trends",
                "content": "There's been a 15% improvement in overall code quality scores over the past week. This correlates with increased usage of the AI optimization feature. Consider promoting this feature to more users."
            },
            {
                "title": "Testing Efficiency",
                "content": "Users who leverage AI-generated tests complete their development cycles 30% faster on average. Automated test generation has identified 23% more edge cases than manually written tests."
            },
            {
                "title": "Language-Specific Insights",
                "content": "Python usage has increased by 12% this month. Consider adding more Python-specific optimization rules and templates to better serve this growing user segment."
            },
            {
                "title": "Performance Bottlenecks",
                "content": "The analysis indicates that 65% of performance issues stem from inefficient database queries. Implementing the suggested query optimizations could yield a 40% performance improvement."
            }
        ]
        return insights
