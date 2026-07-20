import React, { useState, useEffect } from 'react';
import { WebSocket } from 'websockets';
import Plot from 'react-plotly.js';

interface AnalyticsData {
  documentVolume: number[];
  clauseAdoption: Record<string, number>;
  userEngagement: {
    activeUsers: number;
    interactions: number;
  };
  apiUsage: Record<string, number>;
}

const AnalyticsDashboard: React.FC = () => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Establish WebSocket connection for real-time updates
    const ws = new WebSocket('ws://your-analytics-endpoint');
    setWebsocket(ws);

    ws.onmessage = (event) => {
      const data: AnalyticsData = JSON.parse(event.data);
      setAnalyticsData(data);
    };

    return () => {
      ws.close();
    };
  }, []);

  if (!analyticsData) return <div>Loading analytics...</div>;

  return (
    <div className="analytics-dashboard">
      <h1>Business Intelligence Dashboard</h1>
      
      <section className="document-volume">
        <h2>Document Processing Volume</h2>
        <Plot
          data={[{
            x: Array.from({length: analyticsData.documentVolume.length}, (_, i) => i),
            y: analyticsData.documentVolume,
            type: 'scatter',
            mode: 'lines+markers',
            marker: {color: 'blue'}
          }]}
          layout={{title: 'Document Volume Trend'}}
        />
      </section>

      <section className="clause-adoption">
        <h2>Clause Adoption Trends</h2>
        <Plot
          data={[{
            x: Object.keys(analyticsData.clauseAdoption),
            y: Object.values(analyticsData.clauseAdoption),
            type: 'bar',
            marker: {color: 'green'}
          }]}
          layout={{title: 'Clause Type Adoption'}}
        />
      </section>

      <section className="user-engagement">
        <h2>User Engagement</h2>
        <div>
          <p>Active Users: {analyticsData.userEngagement.activeUsers}</p>
          <p>Total Interactions: {analyticsData.userEngagement.interactions}</p>
        </div>
      </section>

      <section className="api-usage">
        <h2>API Usage Distribution</h2>
        <Plot
          data={[{
            labels: Object.keys(analyticsData.apiUsage),
            values: Object.values(analyticsData.apiUsage),
            type: 'pie'
          }]}
          layout={{title: 'API Endpoint Usage'}}
        />
      </section>
    </div>
  );
};

export default AnalyticsDashboard;