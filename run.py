#!/usr/bin/env python3

import os
import sys
import webbrowser
from threading import Timer
from app import create_app

def open_browser(port):
    webbrowser.open(f'http://localhost:{port}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app = create_app()  
    
    # Open browser after a short delay
    Timer(1.5, open_browser, [port]).start()
    
    # Start the server
    app.run(host='0.0.0.0', port=port, debug=True)
