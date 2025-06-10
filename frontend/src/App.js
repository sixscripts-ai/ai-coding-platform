import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import styled from 'styled-components';

// Components
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Editor from './components/Editor';
import Terminal from './components/Terminal';
import ProjectSelector from './components/ProjectSelector';
import AIAssistant from './components/AIAssistant';
import WelcomePage from './components/WelcomePage';
import SuperAgentDashboard from './components/SuperAgentDashboard';

// Styles
import './App.css';

const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background-color: #1e1e1e;
  color: #f0f0f0;
`;

const MainContainer = styled.div`
  display: flex;
  flex: 1;
  overflow: hidden;
`;

const ContentContainer = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
`;

const WorkspaceContainer = styled.div`
  display: flex;
  flex: 1;
  overflow: hidden;
`;

function App() {
  const [activeProject, setActiveProject] = useState(null);
  const [activeFile, setActiveFile] = useState(null);
  const [sidebarWidth, setSidebarWidth] = useState(250);
  const [terminalHeight, setTerminalHeight] = useState(200);
  const [showTerminal, setShowTerminal] = useState(true);
  const [showAIAssistant, setShowAIAssistant] = useState(false);
  const [aiAssistantWidth, setAIAssistantWidth] = useState(300);
  
  const toggleTerminal = () => {
    setShowTerminal(!showTerminal);
  };
  
  const toggleAIAssistant = () => {
    setShowAIAssistant(!showAIAssistant);
  };
  
  return (
    <Router>
      <AppContainer>
        <Navbar 
          activeProject={activeProject}
          toggleTerminal={toggleTerminal}
          toggleAIAssistant={toggleAIAssistant}
        />
        <MainContainer>
          <Sidebar 
            width={sidebarWidth} 
            setWidth={setSidebarWidth}
            activeProject={activeProject}
            setActiveFile={setActiveFile}
          />
          <ContentContainer>
            <Routes>
              <Route path="/" element={
                activeProject ? (
                  <WorkspaceContainer>
                    <Editor activeFile={activeFile} />
                    {showAIAssistant && (
                      <AIAssistant 
                        width={aiAssistantWidth}
                        setWidth={setAIAssistantWidth}
                        activeFile={activeFile}
                      />
                    )}
                  </WorkspaceContainer>
                ) : (
                  <WelcomePage setActiveProject={setActiveProject} />
                )
              } />
              <Route path="/projects" element={
                <ProjectSelector
                  setActiveProject={setActiveProject}
                />
              } />
              <Route path="/dashboard" element={<SuperAgentDashboard />} />
            </Routes>
            {showTerminal && activeProject && (
              <Terminal 
                height={terminalHeight} 
                setHeight={setTerminalHeight}
                activeProject={activeProject}
              />
            )}
          </ContentContainer>
        </MainContainer>
      </AppContainer>
    </Router>
  );
}

export default App;
