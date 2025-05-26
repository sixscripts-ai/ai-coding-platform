import React, { useState } from 'react';
import styled from 'styled-components';
import { useNavigate } from 'react-router-dom';

const PageContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: #1e1e1e;
  color: #e1e1e1;
  padding: 20px;
  overflow-y: auto;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
`;

const Title = styled.h1`
  font-size: 24px;
  margin: 0;
`;

const Button = styled.button`
  background-color: #0078d7;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  font-size: 14px;
  cursor: pointer;
  display: flex;
  align-items: center;
  
  &:hover {
    background-color: #0069c0;
  }
  
  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(0, 120, 215, 0.5);
  }
`;

const ProjectsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
`;

const ProjectCard = styled.div`
  background-color: #252526;
  border: 1px solid ${props => props.selected ? '#0078d7' : '#333'};
  border-radius: 6px;
  padding: 16px;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
  
  ${props => props.selected && `
    box-shadow: 0 0 0 2px #0078d7;
  `}
`;

const ProjectName = styled.h3`
  margin-top: 0;
  margin-bottom: 8px;
  font-size: 18px;
`;

const ProjectDescription = styled.p`
  color: #cccccc;
  font-size: 14px;
  margin-bottom: 16px;
`;

const ProjectMeta = styled.div`
  display: flex;
  justify-content: space-between;
  color: #8e8e8e;
  font-size: 12px;
`;

const ActionButtons = styled.div`
  display: flex;
  justify-content: flex-end;
  margin-top: 30px;
  gap: 10px;
`;

const ProjectSelector = ({ setActiveProject }) => {
  const navigate = useNavigate();
  const [selectedProject, setSelectedProject] = useState(null);
  
  // Sample project data
  const projects = [
    {
      id: 'project-1',
      name: 'React Weather App',
      description: 'A simple weather application built with React.',
      language: 'JavaScript',
      lastModified: '2 days ago'
    },
    {
      id: 'project-2',
      name: 'Personal Portfolio',
      description: 'A portfolio website showcasing my projects and skills.',
      language: 'HTML/CSS/JavaScript',
      lastModified: '1 week ago'
    },
    {
      id: 'project-3',
      name: 'Python Data Analysis',
      description: 'Data analysis scripts for processing CSV files.',
      language: 'Python',
      lastModified: '3 days ago'
    },
    {
      id: 'project-4',
      name: 'Node.js API',
      description: 'RESTful API built with Node.js and Express.',
      language: 'JavaScript',
      lastModified: '5 days ago'
    }
  ];
  
  const handleProjectClick = (project) => {
    setSelectedProject(project);
  };
  
  const handleOpenProject = () => {
    if (selectedProject) {
      setActiveProject(selectedProject);
      navigate('/');
    }
  };
  
  const handleCreateNewProject = () => {
    // In a real app, this would open a dialog to create a new project
    const newProject = {
      id: `project-${projects.length + 1}`,
      name: `New Project ${projects.length + 1}`,
      description: 'A new project.',
      language: 'JavaScript',
      lastModified: 'Just now'
    };
    
    setSelectedProject(newProject);
  };
  
  return (
    <PageContainer>
      <Header>
        <Title>Projects</Title>
        <Button onClick={handleCreateNewProject}>Create New Project</Button>
      </Header>
      
      <ProjectsGrid>
        {projects.map(project => (
          <ProjectCard
            key={project.id}
            selected={selectedProject && selectedProject.id === project.id}
            onClick={() => handleProjectClick(project)}
          >
            <ProjectName>{project.name}</ProjectName>
            <ProjectDescription>{project.description}</ProjectDescription>
            <ProjectMeta>
              <span>{project.language}</span>
              <span>Modified: {project.lastModified}</span>
            </ProjectMeta>
          </ProjectCard>
        ))}
      </ProjectsGrid>
      
      <ActionButtons>
        <Button 
          onClick={() => navigate('/')}
          style={{ backgroundColor: '#555' }}
        >
          Cancel
        </Button>
        <Button 
          onClick={handleOpenProject}
          disabled={!selectedProject}
          style={{ opacity: selectedProject ? 1 : 0.5 }}
        >
          Open Project
        </Button>
      </ActionButtons>
    </PageContainer>
  );
};

export default ProjectSelector;

