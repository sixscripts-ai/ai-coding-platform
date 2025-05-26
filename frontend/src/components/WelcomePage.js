import React from 'react';
import styled from 'styled-components';
import { useNavigate } from 'react-router-dom';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  background-color: #1e1e1e;
  color: #e1e1e1;
  padding: 20px;
`;

const WelcomeCard = styled.div`
  background-color: #252526;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  padding: 40px;
  max-width: 600px;
  width: 100%;
  text-align: center;
`;

const Logo = styled.div`
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 10px;
  color: #0078d7;
`;

const Title = styled.h1`
  font-size: 28px;
  margin-bottom: 20px;
`;

const Description = styled.p`
  font-size: 16px;
  color: #cccccc;
  margin-bottom: 30px;
  line-height: 1.5;
`;

const ButtonContainer = styled.div`
  display: flex;
  gap: 15px;
  justify-content: center;
`;

const Button = styled.button`
  background-color: ${props => props.primary ? '#0078d7' : '#333333'};
  color: white;
  border: none;
  border-radius: 4px;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: ${props => props.primary ? '#0066b8' : '#444444'};
  }
`;

const SampleProjects = styled.div`
  margin-top: 40px;
  width: 100%;
`;

const SampleTitle = styled.h3`
  font-size: 18px;
  margin-bottom: 15px;
  text-align: left;
`;

const ProjectGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 15px;
`;

const ProjectCard = styled.div`
  background-color: #333333;
  border-radius: 6px;
  padding: 15px;
  cursor: pointer;
  transition: transform 0.2s, background-color 0.2s;
  
  &:hover {
    background-color: #3c3c3c;
    transform: translateY(-2px);
  }
`;

const ProjectName = styled.div`
  font-weight: bold;
  margin-bottom: 8px;
`;

const ProjectDescription = styled.div`
  font-size: 13px;
  color: #aaaaaa;
`;

const WelcomePage = ({ setActiveProject }) => {
  const navigate = useNavigate();
  
  // Sample projects
  const sampleProjects = [
    {
      id: 'sample-1',
      name: 'React Todo App',
      description: 'A simple todo application',
      language: 'JavaScript'
    },
    {
      id: 'sample-2',
      name: 'Portfolio Site',
      description: 'Personal portfolio website',
      language: 'HTML/CSS/JS'
    },
    {
      id: 'sample-3',
      name: 'Weather Dashboard',
      description: 'Weather forecast application',
      language: 'JavaScript'
    }
  ];
  
  const handleCreateProject = () => {
    // In a real app, this would open a dialog to create a new project
    const newProject = {
      id: 'new-project',
      name: 'New Project',
      description: 'A new project created by the user',
      language: 'JavaScript'
    };
    
    setActiveProject(newProject);
  };
  
  const handleOpenProject = () => {
    // Navigate to project selection screen
    navigate('/projects');
  };
  
  const handleSampleProjectClick = (project) => {
    setActiveProject(project);
  };
  
  return (
    <Container>
      <WelcomeCard>
        <Logo>AI Coding Platform</Logo>
        <Title>Welcome to Your Coding Workspace</Title>
        <Description>
          Get started with intelligent coding assistance, optimized workflows, and AI-powered development tools.
          Create a new project or open an existing one to begin.
        </Description>
        
        <ButtonContainer>
          <Button primary onClick={handleCreateProject}>Create New Project</Button>
          <Button onClick={handleOpenProject}>Open Project</Button>
        </ButtonContainer>
        
        <SampleProjects>
          <SampleTitle>Sample Projects</SampleTitle>
          <ProjectGrid>
            {sampleProjects.map(project => (
              <ProjectCard 
                key={project.id}
                onClick={() => handleSampleProjectClick(project)}
              >
                <ProjectName>{project.name}</ProjectName>
                <ProjectDescription>{project.description}</ProjectDescription>
              </ProjectCard>
            ))}
          </ProjectGrid>
        </SampleProjects>
      </WelcomeCard>
    </Container>
  );
};

export default WelcomePage;

