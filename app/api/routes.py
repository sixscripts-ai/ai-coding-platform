from flask import jsonify, request, current_app
import os
import json
import subprocess
from app.api import api_bp
from app.services.ai_service import generate_code, analyze_code, complete_code
from app.services.file_service import list_files, read_file, write_file, create_file, delete_file
from app.services.project_service import create_project, list_projects, get_project

# AI Assistant routes
@api_bp.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt')
    language = data.get('language', 'python')
    context = data.get('context', '')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        generated_code = generate_code(prompt, language, context)
        return jsonify({'code': generated_code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    code = data.get('code')
    language = data.get('language', 'python')
    
    if not code:
        return jsonify({'error': 'Code is required'}), 400
    
    try:
        analysis = analyze_code(code, language)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/complete', methods=['POST'])
def complete():
    data = request.get_json()
    code = data.get('code')
    cursor_position = data.get('cursor_position')
    language = data.get('language', 'python')
    
    if not code or cursor_position is None:
        return jsonify({'error': 'Code and cursor position are required'}), 400
    
    try:
        completion = complete_code(code, cursor_position, language)
        return jsonify({'completion': completion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# File operations routes
@api_bp.route('/files', methods=['GET'])
def get_files():
    project_id = request.args.get('project_id')
    path = request.args.get('path', '/')
    
    if not project_id:
        return jsonify({'error': 'Project ID is required'}), 400
    
    try:
        files = list_files(project_id, path)
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/files/read', methods=['GET'])
def get_file_content():
    project_id = request.args.get('project_id')
    file_path = request.args.get('path')
    
    if not project_id or not file_path:
        return jsonify({'error': 'Project ID and file path are required'}), 400
    
    try:
        content = read_file(project_id, file_path)
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/files/write', methods=['POST'])
def save_file_content():
    data = request.get_json()
    project_id = data.get('project_id')
    file_path = data.get('path')
    content = data.get('content')
    
    if not project_id or not file_path or content is None:
        return jsonify({'error': 'Project ID, file path, and content are required'}), 400
    
    try:
        write_file(project_id, file_path, content)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/files/create', methods=['POST'])
def create_new_file():
    data = request.get_json()
    project_id = data.get('project_id')
    file_path = data.get('path')
    file_type = data.get('type', 'file')  # 'file' or 'directory'
    
    if not project_id or not file_path:
        return jsonify({'error': 'Project ID and file path are required'}), 400
    
    try:
        create_file(project_id, file_path, file_type)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/files/delete', methods=['DELETE'])
def delete_existing_file():
    project_id = request.args.get('project_id')
    file_path = request.args.get('path')
    
    if not project_id or not file_path:
        return jsonify({'error': 'Project ID and file path are required'}), 400
    
    try:
        delete_file(project_id, file_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Project management routes
@api_bp.route('/projects', methods=['GET'])
def get_projects():
    try:
        projects = list_projects()
        return jsonify({'projects': projects})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/projects/<project_id>', methods=['GET'])
def get_project_details(project_id):
    try:
        project = get_project(project_id)
        return jsonify(project)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/projects', methods=['POST'])
def create_new_project():
    data = request.get_json()
    name = data.get('name')
    description = data.get('description', '')
    template = data.get('template', 'blank')
    
    if not name:
        return jsonify({'error': 'Project name is required'}), 400
    
    try:
        project = create_project(name, description, template)
        return jsonify(project)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Terminal integration
@api_bp.route('/terminal/execute', methods=['POST'])
def execute_command():
    data = request.get_json()
    command = data.get('command')
    project_id = data.get('project_id')
    working_dir = data.get('working_dir', '/')
    
    if not command or not project_id:
        return jsonify({'error': 'Command and project ID are required'}), 400
    
    try:
        # Get the project directory
        project_dir = os.path.join(current_app.instance_path, 'projects', project_id)
        working_path = os.path.join(project_dir, working_dir.lstrip('/'))
        
        # Execute the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_path
        )
        
        stdout, stderr = process.communicate()
        
        return jsonify({
            'stdout': stdout.decode('utf-8'),
            'stderr': stderr.decode('utf-8'),
            'exit_code': process.returncode
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
