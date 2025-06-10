import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const Container = styled.div`
  padding: 20px;
  color: #e1e1e1;
`;

const Title = styled.h2`
  margin-bottom: 20px;
`;

const MetricsList = styled.ul`
  list-style: none;
  padding: 0;
`;

const MetricItem = styled.li`
  margin-bottom: 8px;
`;

function SuperAgentDashboard() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get('/api/super-agent/status');
        if (response.data && response.data.agent_status) {
          setStatus(response.data.agent_status);
        }
      } catch (err) {
        console.error('Failed to load status', err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!status) {
    return <Container>Loading...</Container>;
  }

  return (
    <Container>
      <Title>AI Super-Agent Dashboard</Title>
      <MetricsList>
        <MetricItem>Running: {String(status.running)}</MetricItem>
        <MetricItem>Queue Size: {status.queue_size}</MetricItem>
        <MetricItem>Completed Tasks: {status.completed_tasks}</MetricItem>
        <MetricItem>Uptime: {status.uptime}s</MetricItem>
      </MetricsList>
    </Container>
  );
}

export default SuperAgentDashboard;
