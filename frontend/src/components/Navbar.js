import React from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';

const NavbarContainer = styled.div`
  display: flex;
  align-items: center;
  background-color: #1e1e1e;
  color: #ffffff;
  height: 50px;
  border-bottom: 1px solid #333;
  padding: 0 16px;
`;

const Logo = styled.div`
  font-size: 20px;
  font-weight: bold;
  margin-right: 24px;
`;

const ProjectName = styled.div`
  font-size: 16px;
  margin-right: auto;
  color: #ccc;
`;

const NavItems = styled.div`
  display: flex;
  align-items: center;
`;

const NavButton = styled.button`
  background-color: transparent;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 6px 12px;
  margin-left: 8px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  
  &:hover {
    background-color: #333;
  }
  
  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(30, 144, 255, 0.5);
  }
`;

const NavLink = styled(Link)`
  color: #fff;
  text-decoration: none;
  padding: 6px 12px;
  border-radius: 4px;
  
  &:hover {
    background-color: #333;
  }
`;

const Navbar = ({ activeProject, toggleTerminal, toggleAIAssistant }) => {
  return (
    <NavbarContainer>
      <Logo>AI Coding Platform</Logo>
      
      <ProjectName>
        {activeProject ? activeProject.name || 'Untitled Project' : 'No Project Selected'}
      </ProjectName>
      
      <NavItems>
        <NavLink to="/projects">Projects</NavLink>
        <NavLink to="/dashboard">Dashboard</NavLink>

        <NavButton onClick={toggleTerminal}>
          Toggle Terminal
        </NavButton>
        
        <NavButton onClick={toggleAIAssistant}>
          Toggle AI Assistant
        </NavButton>
      </NavItems>
    </NavbarContainer>
  );
};

export default Navbar;

