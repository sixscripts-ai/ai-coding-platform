import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const EditorContainer = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  height: 100%;
  background-color: #1e1e1e;
  color: #d4d4d4;
  overflow: hidden;
`;

const EditorHeader = styled.div`
  background-color: #2d2d2d;
  padding: 8px 16px;
  font-size: 14px;
  border-bottom: 1px solid #3e3e3e;
  display: flex;
  align-items: center;
`;

const FileName = styled.div`
  font-weight: bold;
  margin-right: auto;
`;

const EditorContent = styled.div`
  flex: 1;
  overflow: hidden;
  position: relative;
`;

const StyledTextarea = styled.textarea`
  width: 100%;
  height: 100%;
  background-color: #1e1e1e;
  color: #d4d4d4;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.5;
  padding: 16px;
  border: none;
  resize: none;
  outline: none;
  white-space: pre;
  overflow: auto;
  tab-size: 2;
`;

const NoFileSelected = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 16px;
  color: #6d6d6d;
`;

const Editor = ({ activeFile }) => {
  const [content, setContent] = useState('');
  
  // Initialize content when active file changes
  useEffect(() => {
    if (activeFile) {
      // In a real app, this would fetch the file content from a service
      // For now, just use some dummy content based on the file name
      const dummyContent = activeFile.name.endsWith('.js') 
        ? "// JavaScript file\nconsole.log('Hello, world!');\n\nfunction example() {\n  return 'This is an example';\n}"
        : activeFile.name.endsWith('.css')
        ? "/* CSS file */\n\nbody {\n  margin: 0;\n  padding: 0;\n  font-family: sans-serif;\n}"
        : "# Generic file\n\nThis is a sample file.";
      
      setContent(dummyContent);
    } else {
      setContent('');
    }
  }, [activeFile]);
  
  const handleContentChange = (e) => {
    setContent(e.target.value);
  };
  
  if (!activeFile) {
    return (
      <EditorContainer>
        <NoFileSelected>No file selected</NoFileSelected>
      </EditorContainer>
    );
  }
  
  return (
    <EditorContainer>
      <EditorHeader>
        <FileName>{activeFile.name}</FileName>
      </EditorHeader>
      <EditorContent>
        <StyledTextarea 
          value={content}
          onChange={handleContentChange}
          spellCheck="false"
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
        />
      </EditorContent>
    </EditorContainer>
  );
};

export default Editor;

