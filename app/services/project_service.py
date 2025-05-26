import os
import json
import uuid
import shutil
import datetime
from flask import current_app

def get_projects_file():
    """
    Get the path to the projects metadata file
    """
    return os.path.join(current_app.instance_path, 'projects.json')

def load_projects():
    """
    Load projects metadata from the JSON file
    """
    projects_file = get_projects_file()
    
    if not os.path.exists(projects_file):
        # Initialize with empty projects list
        with open(projects_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return []
    
    with open(projects_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_projects(projects):
    """
    Save projects metadata to the JSON file
    """
    projects_file = get_projects_file()
    
    with open(projects_file, 'w', encoding='utf-8') as f:
        json.dump(projects, f, indent=2)

def list_projects():
    """
    List all projects
    """
    try:
        return load_projects()
    except Exception as e:
        current_app.logger.error(f"Error listing projects: {str(e)}")
        raise

def get_project(project_id):
    """
    Get a specific project by ID
    """
    try:
        projects = load_projects()
        for project in projects:
            if project['id'] == project_id:
                return project
        
        raise ValueError(f"Project {project_id} not found")
    except Exception as e:
        current_app.logger.error(f"Error getting project: {str(e)}")
        raise

def create_project(name, description='', template='blank'):
    """
    Create a new project
    """
    try:
        projects = load_projects()
        
        # Generate a unique ID
        project_id = str(uuid.uuid4())
        
        # Create project metadata
        now = datetime.datetime.now().isoformat()
        project = {
            'id': project_id,
            'name': name,
            'description': description,
            'created_at': now,
            'updated_at': now,
            'template': template
        }
        
        # Create project directory
        project_dir = os.path.join(current_app.instance_path, 'projects', project_id)
        os.makedirs(project_dir, exist_ok=True)
        
        # Apply template if specified
        if template != 'blank':
            apply_template(project_dir, template)
        
        # Add to projects list and save
        projects.append(project)
        save_projects(projects)
        
        return project
    except Exception as e:
        current_app.logger.error(f"Error creating project: {str(e)}")
        raise

def apply_template(project_dir, template):
    """
    Apply a template to a new project
    """
    templates = {
        'python': create_python_template,
        'web': create_web_template,
        'react': create_react_template,
        'node': create_node_template
    }
    
    if template in templates:
        templates[template](project_dir)
    else:
        raise ValueError(f"Unknown template: {template}")

def create_python_template(project_dir):
    """
    Create a basic Python project structure
    """
    # Create directories
    os.makedirs(os.path.join(project_dir, 'src'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'tests'), exist_ok=True)
    
    # Create files
    with open(os.path.join(project_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write("# Python Project\n\nA Python project created with AI Coding Platform.")
    
    with open(os.path.join(project_dir, 'requirements.txt'), 'w', encoding='utf-8') as f:
        f.write("# Project dependencies\n")
    
    with open(os.path.join(project_dir, 'src', '__init__.py'), 'w', encoding='utf-8') as f:
        pass
    
    with open(os.path.join(project_dir, 'src', 'main.py'), 'w', encoding='utf-8') as f:
        f.write("""def main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()\n""")
    
    with open(os.path.join(project_dir, 'tests', '__init__.py'), 'w', encoding='utf-8') as f:
        pass
    
    with open(os.path.join(project_dir, 'tests', 'test_main.py'), 'w', encoding='utf-8') as f:
        f.write("""import unittest\nfrom src.main import main\n\nclass TestMain(unittest.TestCase):\n    def test_main(self):\n        # TODO: Write actual tests\n        self.assertTrue(True)\n\nif __name__ == "__main__":\n    unittest.main()\n""")

def create_web_template(project_dir):
    """
    Create a basic web project structure (HTML, CSS, JS)
    """
    # Create directories
    os.makedirs(os.path.join(project_dir, 'css'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'js'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'images'), exist_ok=True)
    
    # Create files
    with open(os.path.join(project_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Web Project</title>\n    <link rel="stylesheet" href="css/style.css">\n</head>\n<body>\n    <header>\n        <h1>Web Project</h1>\n    </header>\n    <main>\n        <p>Welcome to your new web project!</p>\n    </main>\n    <footer>\n        <p>Created with AI Coding Platform</p>\n    </footer>\n    <script src="js/main.js"></script>\n</body>\n</html>\n""")
    
    with open(os.path.join(project_dir, 'css', 'style.css'), 'w', encoding='utf-8') as f:
        f.write("""body {\n    font-family: Arial, sans-serif;\n    line-height: 1.6;\n    margin: 0;\n    padding: 0;\n    color: #333;\n}\n\nheader, footer {\n    background-color: #f4f4f4;\n    text-align: center;\n    padding: 1rem;\n}\n\nmain {\n    padding: 2rem;\n    max-width: 800px;\n    margin: 0 auto;\n}\n""")
    
    with open(os.path.join(project_dir, 'js', 'main.js'), 'w', encoding='utf-8') as f:
        f.write("""// Main JavaScript file\n\ndocument.addEventListener('DOMContentLoaded', () => {\n    console.log('Web project initialized');\n});\n""")
    
    with open(os.path.join(project_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write("# Web Project\n\nA web project created with AI Coding Platform.")

def create_react_template(project_dir):
    """
    Create a basic React project structure
    """
    # Create directories
    os.makedirs(os.path.join(project_dir, 'public'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'src'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'src', 'components'), exist_ok=True)
    
    # Create files
    with open(os.path.join(project_dir, 'package.json'), 'w', encoding='utf-8') as f:
        f.write("""{\n  "name": "react-project",\n  "version": "0.1.0",\n  "private": true,\n  "dependencies": {\n    "react": "^17.0.2",\n    "react-dom": "^17.0.2",\n    "react-scripts": "5.0.0"\n  },\n  "scripts": {\n    "start": "react-scripts start",\n    "build": "react-scripts build",\n    "test": "react-scripts test",\n    "eject": "react-scripts eject"\n  },\n  "eslintConfig": {\n    "extends": [\n      "react-app",\n      "react-app/jest"\n    ]\n  },\n  "browserslist": {\n    "production": [\n      ">0.2%",\n      "not dead",\n      "not op_mini all"\n    ],\n    "development": [\n      "last 1 chrome version",\n      "last 1 firefox version",\n      "last 1 safari version"\n    ]\n  }\n}\n""")
    
    with open(os.path.join(project_dir, 'public', 'index.html'), 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1" />\n    <title>React App</title>\n</head>\n<body>\n    <noscript>You need to enable JavaScript to run this app.</noscript>\n    <div id="root"></div>\n</body>\n</html>\n""")
    
    with open(os.path.join(project_dir, 'src', 'index.js'), 'w', encoding='utf-8') as f:
        f.write("""import React from 'react';\nimport ReactDOM from 'react-dom';\nimport './index.css';\nimport App from './App';\n\nReactDOM.render(\n  <React.StrictMode>\n    <App />\n  </React.StrictMode>,\n  document.getElementById('root')\n);\n""")
    
    with open(os.path.join(project_dir, 'src', 'index.css'), 'w', encoding='utf-8') as f:
        f.write("""body {\n  margin: 0;\n  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',\n    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',\n    sans-serif;\n  -webkit-font-smoothing: antialiased;\n  -moz-osx-font-smoothing: grayscale;\n}\n\ncode {\n  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',\n    monospace;\n}\n""")
    
    with open(os.path.join(project_dir, 'src', 'App.js'), 'w', encoding='utf-8') as f:
        f.write("""import React from 'react';\nimport './App.css';\n\nfunction App() {\n  return (\n    <div className="App">\n      <header className="App-header">\n        <h1>React Project</h1>\n        <p>Welcome to your new React project!</p>\n      </header>\n    </div>\n  );\n}\n\nexport default App;\n""")
    
    with open(os.path.join(project_dir, 'src', 'App.css'), 'w', encoding='utf-8') as f:
        f.write(""".App {\n  text-align: center;\n}\n\n.App-header {\n  background-color: #282c34;\n  min-height: 100vh;\n  display: flex;\n  flex-direction: column;\n  align-items: center;\n  justify-content: center;\n  font-size: calc(10px + 2vmin);\n  color: white;\n}\n""")
    
    with open(os.path.join(project_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write("# React Project\n\nA React project created with AI Coding Platform.")

def create_node_template(project_dir):
    """
    Create a basic Node.js project structure
    """
    # Create directories
    os.makedirs(os.path.join(project_dir, 'src'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'tests'), exist_ok=True)
    
    # Create files
    with open(os.path.join(project_dir, 'package.json'), 'w', encoding='utf-8') as f:
        f.write("""{\n  "name": "node-project",\n  "version": "1.0.0",\n  "description": "A Node.js project created with AI Coding Platform",\n  "main": "src/index.js",\n  "scripts": {\n    "start": "node src/index.js",\n    "test": "mocha tests/**/*.js"\n  },\n  "dependencies": {\n    "express": "^4.17.1"\n  },\n  "devDependencies": {\n    "mocha": "^9.1.3",\n    "chai": "^4.3.4"\n  }\n}\n""")
    
    with open(os.path.join(project_dir, 'src', 'index.js'), 'w', encoding='utf-8') as f:
        f.write("""const express = require('express');\nconst app = express();\nconst port = process.env.PORT || 3000;\n\napp.get('/', (req, res) => {\n  res.send('Hello from Node.js!');\n});\n\napp.listen(port, () => {\n  console.log(`Server running at http://localhost:${port}`);\n});\n""")
    
    with open(os.path.join(project_dir, 'tests', 'index.test.js'), 'w', encoding='utf-8') as f:
        f.write("""const { expect } = require('chai');\n\ndescribe('Sample Test', () => {\n  it('should pass', () => {\n    expect(true).to.be.true;\n  });\n});\n""")
    
    with open(os.path.join(project_dir, '.gitignore'), 'w', encoding='utf-8') as f:
        f.write("""node_modules/\nnpm-debug.log\n.env\n""")
    
    with open(os.path.join(project_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write("# Node.js Project\n\nA Node.js project created with AI Coding Platform.")
