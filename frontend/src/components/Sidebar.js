import React, { useRef, useEffect } from 'react';
import styled from 'styled-components';

// Styled components
const SidebarContainer = styled.div`
  width: ${props => props.width}px;
  height: 100%;
  background-color: #252526;
  color: #e1e1e1;
  border-right: 1px solid #333;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
`;

const ResizeHandle = styled.div`
  width: 5px;
  height: 100%;
  position: absolute;
  right: 0;
  top: 0;
  cursor: col-resize;
  background-color: transparent;
  &:hover {
    background-color: #0078d7;
  }
`;

const SidebarHeader = styled.div`
  padding: 12px;
  font-weight: bold;
  border-bottom: 1px solid #333;
  background-color: #333;
`;

const FileList = styled.ul`
  list-style: none;
  padding: 0;
  margin: 0;
  overflow-y: auto;
  flex: 1;
`;

const FileItem = styled.li`
  padding: 8px 12px;
  cursor: pointer;
  &:hover {
    background-color: #2a2d2e;
  }
  &.active {
    background-color: #37373d;
  }
`;

const Sidebar = ({ width, setWidth, activeProject, setActiveFile }) => {
  const resizeHandleRef = useRef(null);
  
  // Handle sidebar resizing
  useEffect(() => {
    const handle = resizeHandleRef.current;
    if (!handle) return;
    
    let startX;
    let startWidth;
    
    const onMouseDown = (e) => {
      startX = e.clientX;
      startWidth = width;
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };
    
    const onMouseMove = (e) => {
      const newWidth = startWidth + e.clientX - startX;
      if (newWidth > 100 && newWidth < 500) {
        setWidth(newWidth);
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
  }, [width, setWidth]);
  
  // Sample files for demonstration
  const sampleFiles = activeProject ? [
    { id: 1, name: 'index.js', path: '/index.js' },
    { id: 2, name: 'App.js', path: '/App.js' },
    { id: 3, name: 'styles.css', path: '/styles.css' }
  ] : [];
  
  const handleFileClick = (file) => {
    setActiveFile && setActiveFile(file);
  };
  
  return (
    <SidebarContainer width={width}>
      <SidebarHeader>
        {activeProject ? activeProject.name || 'Project Files' : 'No Project Selected'}
      </SidebarHeader>
      
      <FileList>
        {sampleFiles.map(file => (
          <FileItem 
            key={file.id} 
            onClick={() => handleFileClick(file)}
          >
            {file.name}
          </FileItem>
        ))}
      </FileList>
      
      <ResizeHandle ref={resizeHandleRef} />
    </SidebarContainer>
  );
};

export default Sidebar;

