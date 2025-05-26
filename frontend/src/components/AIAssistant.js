import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

const AssistantContainer = styled.div`
  width: ${props => props.width}px;
  height: 100%;
  background-color: #252526;
  color: #e1e1e1;
  border-left: 1px solid #333;
  display: flex;
  flex-direction: column;
  position: relative;
`;

const ResizeHandle = styled.div`
  width: 5px;
  height: 100%;
  position: absolute;
  left: 0;
  top: 0;
  cursor: col-resize;
  background-color: transparent;
  &:hover {
    background-color: #0078d7;
  }
  z-index: 10;
`;

const AssistantHeader = styled.div`
  background-color: #333;
  padding: 10px 15px;
  font-weight: bold;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #444;
`;

const Title = styled.div`
  font-size: 14px;
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  color: #999;
  cursor: pointer;
  font-size: 16px;
  
  &:hover {
    color: #fff;
  }
`;

const ChatContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 15px;
  display: flex;
  flex-direction: column;
  gap: 15px;
`;

const Message = styled.div`
  max-width: 80%;
  padding: 10px 15px;
  border-radius: 10px;
  font-size: 14px;
  line-height: 1.4;
  
  ${props => props.isUser ? `
    align-self: flex-end;
    background-color: #0078d7;
    color: white;
    border-bottom-right-radius: 0;
  ` : `
    align-self: flex-start;
    background-color: #3c3c3c;
    color: #e1e1e1;
    border-bottom-left-radius: 0;
  `}
`;

const InputContainer = styled.div`
  padding: 15px;
  border-top: 1px solid #444;
  display: flex;
  gap: 10px;
`;

const MessageInput = styled.input`
  flex: 1;
  background-color: #3c3c3c;
  color: #e1e1e1;
  border: 1px solid #555;
  border-radius: 4px;
  padding: 8px 12px;
  font-size: 14px;
  
  &:focus {
    outline: none;
    border-color: #0078d7;
  }
`;

const SendButton = styled.button`
  background-color: #0078d7;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 12px;
  cursor: pointer;
  
  &:hover {
    background-color: #0069c0;
  }
  
  &:disabled {
    background-color: #555;
    cursor: not-allowed;
  }
`;

const SuggestionsList = styled.div`
  padding: 15px;
  background-color: #2d2d2d;
  border-top: 1px solid #444;
`;

const SuggestionTitle = styled.div`
  font-size: 12px;
  color: #999;
  margin-bottom: 10px;
`;

const Suggestion = styled.div`
  padding: 8px 12px;
  background-color: #3c3c3c;
  border-radius: 4px;
  margin-bottom: 8px;
  font-size: 13px;
  cursor: pointer;
  
  &:hover {
    background-color: #4c4c4c;
  }
`;

const AIAssistant = ({ width, setWidth, activeFile }) => {
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your AI coding assistant. How can I help you today?", isUser: false }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const resizeHandleRef = useRef(null);
  const messagesEndRef = useRef(null);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Handle assistant panel resizing
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
      // Moving left increases width
      const newWidth = startWidth + (startX - e.clientX);
      if (newWidth > 200 && newWidth < 600) {
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
  
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };
  
  const handleSendMessage = () => {
    if (!inputValue.trim()) return;
    
    // Add user message
    const newMessage = {
      id: messages.length + 1,
      text: inputValue,
      isUser: true
    };
    setMessages([...messages, newMessage]);
    setInputValue('');
    
    // Simulate AI typing
    setIsTyping(true);
    
    // Generate AI response based on user input and active file
    setTimeout(() => {
      let aiResponse = "";
      
      if (inputValue.toLowerCase().includes('help')) {
        aiResponse = "I can help you with coding questions, debugging, or suggesting improvements to your code. What would you like assistance with?";
      } else if (inputValue.toLowerCase().includes('error') || inputValue.toLowerCase().includes('bug')) {
        aiResponse = "I can help you debug. Could you share the error message or describe the issue you're experiencing?";
      } else if (activeFile && inputValue.toLowerCase().includes(activeFile.name.toLowerCase())) {
        aiResponse = `I see you're asking about ${activeFile.name}. What specific part of this file would you like help with?`;
      } else {
        aiResponse = "I understand you're looking for assistance. Could you provide more details about what you need help with?";
      }
      
      setIsTyping(false);
      
      setMessages(prev => [...prev, {
        id: prev.length + 2,
        text: aiResponse,
        isUser: false
      }]);
    }, 1000);
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  // Generate suggestions based on active file
  const getSuggestions = () => {
    if (!activeFile) return [];
    
    // Return different suggestions based on file type
    if (activeFile.name.endsWith('.js')) {
      return [
        "Need help with JavaScript syntax?",
        "How to optimize this function?",
        "Explain React hooks"
      ];
    } else if (activeFile.name.endsWith('.css')) {
      return [
        "How to center a div?",
        "Best practices for responsive design",
        "CSS Grid vs Flexbox"
      ];
    } else {
      return [
        "How can I help with this file?",
        "Need documentation for this code?",
        "Suggest improvements"
      ];
    }
  };
  
  const suggestions = getSuggestions();
  
  return (
    <AssistantContainer width={width}>
      <ResizeHandle ref={resizeHandleRef} />
      
      <AssistantHeader>
        <Title>AI Assistant</Title>
        <CloseButton>&times;</CloseButton>
      </AssistantHeader>
      
      <ChatContainer>
        <MessagesContainer>
          {messages.map(message => (
            <Message key={message.id} isUser={message.isUser}>
              {message.text}
            </Message>
          ))}
          {isTyping && (
            <Message isUser={false}>
              Typing...
            </Message>
          )}
          <div ref={messagesEndRef} />
        </MessagesContainer>
        
        <InputContainer>
          <MessageInput
            type="text"
            placeholder="Ask the AI assistant..."
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
          />
          <SendButton 
            onClick={handleSendMessage} 
            disabled={!inputValue.trim() || isTyping}
          >
            Send
          </SendButton>
        </InputContainer>
        
        {suggestions.length > 0 && (
          <SuggestionsList>
            <SuggestionTitle>Suggestions:</SuggestionTitle>
            {suggestions.map((suggestion, index) => (
              <Suggestion 
                key={index}
                onClick={() => {
                  setInputValue(suggestion);
                }}
              >
                {suggestion}
              </Suggestion>
            ))}
          </SuggestionsList>
        )}
      </ChatContainer>
    </AssistantContainer>
  );
};

export default AIAssistant;

