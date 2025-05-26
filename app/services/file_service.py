import os
import json
import shutil
from flask import current_app

def get_project_path(project_id):
    """
    Get the absolute path to a project directory
    """
    project_dir = os.path.join(current_app.instance_path, 'projects', project_id)
    if not os.path.exists(project_dir):
        raise FileNotFoundError(f"Project {project_id} not found")
    return project_dir

def list_files(project_id, path='/'):
    """
    List files and directories at the specified path within a project
    """
    try:
        project_dir = get_project_path(project_id)
        target_path = os.path.normpath(os.path.join(project_dir, path.lstrip('/'))) 
        
        # Security check to prevent directory traversal
        if not target_path.startswith(project_dir):
            raise ValueError("Invalid path: Attempted directory traversal")
        
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Path {path} not found in project {project_id}")
        
        items = []
        for item in os.listdir(target_path):
            item_path = os.path.join(target_path, item)
            is_dir = os.path.isdir(item_path)
            
            # Get relative path from project root
            rel_path = os.path.relpath(item_path, project_dir)
            if os.name == 'nt':  # Windows
                rel_path = rel_path.replace('\\', '/')
            
            # Skip hidden files and directories
            if item.startswith('.'):
                continue
                
            items.append({
                'name': item,
                'path': '/' + rel_path,
                'type': 'directory' if is_dir else 'file',
                'size': 0 if is_dir else os.path.getsize(item_path),
                'modified': os.path.getmtime(item_path)
            })
        
        # Sort directories first, then files, both alphabetically
        return sorted(items, key=lambda x: (0 if x['type'] == 'directory' else 1, x['name'].lower()))
    
    except Exception as e:
        current_app.logger.error(f"Error listing files: {str(e)}")
        raise

def read_file(project_id, file_path):
    """
    Read the contents of a file within a project
    """
    try:
        project_dir = get_project_path(project_id)
        target_path = os.path.normpath(os.path.join(project_dir, file_path.lstrip('/'))) 
        
        # Security check to prevent directory traversal
        if not target_path.startswith(project_dir):
            raise ValueError("Invalid path: Attempted directory traversal")
        
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"File {file_path} not found in project {project_id}")
        
        if os.path.isdir(target_path):
            raise IsADirectoryError(f"{file_path} is a directory, not a file")
        
        with open(target_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    except Exception as e:
        current_app.logger.error(f"Error reading file: {str(e)}")
        raise

def write_file(project_id, file_path, content):
    """
    Write content to a file within a project
    """
    try:
        project_dir = get_project_path(project_id)
        target_path = os.path.normpath(os.path.join(project_dir, file_path.lstrip('/'))) 
        
        # Security check to prevent directory traversal
        if not target_path.startswith(project_dir):
            raise ValueError("Invalid path: Attempted directory traversal")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    
    except Exception as e:
        current_app.logger.error(f"Error writing file: {str(e)}")
        raise

def create_file(project_id, file_path, file_type='file'):
    """
    Create a new file or directory within a project
    """
    try:
        project_dir = get_project_path(project_id)
        target_path = os.path.normpath(os.path.join(project_dir, file_path.lstrip('/'))) 
        
        # Security check to prevent directory traversal
        if not target_path.startswith(project_dir):
            raise ValueError("Invalid path: Attempted directory traversal")
        
        if os.path.exists(target_path):
            raise FileExistsError(f"{file_path} already exists in project {project_id}")
        
        if file_type == 'directory':
            os.makedirs(target_path, exist_ok=True)
        else:
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # Create an empty file
            with open(target_path, 'w', encoding='utf-8') as f:
                pass
        
        return True
    
    except Exception as e:
        current_app.logger.error(f"Error creating file: {str(e)}")
        raise

def delete_file(project_id, file_path):
    """
    Delete a file or directory within a project
    """
    try:
        project_dir = get_project_path(project_id)
        target_path = os.path.normpath(os.path.join(project_dir, file_path.lstrip('/'))) 
        
        # Security check to prevent directory traversal
        if not target_path.startswith(project_dir):
            raise ValueError("Invalid path: Attempted directory traversal")
        
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"{file_path} not found in project {project_id}")
        
        if os.path.isdir(target_path):
            shutil.rmtree(target_path)
        else:
            os.remove(target_path)
        
        return True
    
    except Exception as e:
        current_app.logger.error(f"Error deleting file: {str(e)}")
        raise
