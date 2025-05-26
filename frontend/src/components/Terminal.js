import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

const TerminalContainer = styled.div`
  height: ${props => props.height}px;
  min-height: 100px;
  background-color: #1e1e1e;
  border-top: 1px solid #333;
  display: flex;
  flex-direction: column;
  position: relative;
`;

const ResizeHandle = styled.div`
  height: 5px;
  width: 100%;
  position: absolute;
  top: 0;
  left: 0;
  cursor: row-resize;
  background-color: transparent;
  &:hover {
    background-color: #0078d7;
  }
  z-index: 10;
`;

const TerminalHeader = styled.div`
  background-color: #252526;
  color: #cccccc;
  padding: 5px 10px;
  font-size: 12px;
  border-bottom: 1px solid #333;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const TerminalTitle = styled.div`
  font-weight: bold;
`;

const TerminalControls = styled.div`
  display: flex;
  gap: 8px;
`;

const ControlButton = styled.button`
  background: none;
  border: none;
  color: #cccccc;
  cursor: pointer;
  font-size: 12px;
  padding: 2px 5px;
  
  &:hover {
    color: white;
  }
`;

const TerminalContent = styled.div`
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  padding: 8px;
`;

const TerminalOutput = styled.div`
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.4;
  color: #cccccc;
  white-space: pre-wrap;
  flex: 1;
  overflow-y: auto;
`;

const TerminalInput = styled.div`
  display: flex;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 14px;
  color: #cccccc;
`;

const Prompt = styled.span`
  color: #4ec9b0;
  margin-right: 8px;
`;

const Input = styled.input`
  background: transparent;
  border: none;
  color: #cccccc;
  font-family: inherit;
  font-size: inherit;
  flex: 1;
  outline: none;
`;

const Terminal = ({ height, setHeight, activeProject }) => {
  const [history, setHistory] = useState([
    { type: 'output', content: 'Welcome to AI Coding Platform Terminal' },
    { type: 'output', content: 'Type "help" for a list of available commands.' },
  ]);
  const [inputValue, setInputValue] = useState('');
  const resizeHandleRef = useRef(null);
  const terminalContentRef = useRef(null);
  
  // Scroll to bottom when history changes
  useEffect(() => {
    if (terminalContentRef.current) {
      terminalContentRef.current.scrollTop = terminalContentRef.current.scrollHeight;
    }
  }, [history]);
  
  // Handle terminal resizing
  useEffect(() => {
    const handle = resizeHandleRef.current;
    if (!handle) return;
    
    let startY;
    let startHeight;
    
    const onMouseDown = (e) => {
      startY = e.clientY;
      startHeight = height;
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };
    
    const onMouseMove = (e) => {
      // Moving up decreases height
      const newHeight = startHeight - (e.clientY - startY);
      if (newHeight >= 100 && newHeight <= 500) {
        setHeight(newHeight);
      }
    };
    
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    
    handle.addEventListener('mousedown', onMouseDown);
    
    return () => {
      handle.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, [height, setHeight]);
  
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };
  
  const handleInputSubmit = (e) => {
    if (e.key === 'Enter') {
      // Add command to history
      const newEntry = { type: 'command', content: `$ ${inputValue}` };
      
      // Process command and generate output
      let output = '';
      
      switch (inputValue.trim().toLowerCase()) {
        case 'help':
          output = 'Available commands:\n  help - Display this help message\n  clear - Clear the terminal\n  ls - List files\n  pwd - Show current directory';
          break;
        case 'clear':
          setHistory([]);
          setInputValue('');
          return;
        case 'ls':
          output = 'file1.js\nfile2.js\nREADME.md';
          break;
        case 'pwd':
          output = activeProject ? `/projects/${activeProject.name}` : '/';
          break;
        default:
          output = `Command not found: ${inputValue}`;
      }
      
      const outputEntry = { type: 'output', content: output };
      setHistory([...history, newEntry, outputEntry]);
      setInputValue('');
    }
  };
  
  const clearTerminal = () => {
    setHistory([]);
  };
  
  return (
    <TerminalContainer height={height}>
      <ResizeHandle ref={resizeHandleRef} />
      
      <TerminalHeader>
        <TerminalTitle>
          {activeProject ? `Terminal: ${activeProject.name}` : 'Terminal'}
        </TerminalTitle>
        <TerminalControls>
          <ControlButton onClick={clearTerminal}>Clear</ControlButton>
        </TerminalControls>
      </TerminalHeader>
      
      <TerminalContent ref={terminalContentRef}>
        <TerminalOutput>
          {history.map((item, index) => (
            <div key={index}>{item.content}</div>
          ))}
        </TerminalOutput>
        
        <TerminalInput>
          <Prompt>$</Prompt>
          <Input 
            type="text" 
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleInputSubmit}
            autoFocus
          />
        </TerminalInput>
      </TerminalContent>
    </TerminalContainer>
  );
};

export default Terminal;

