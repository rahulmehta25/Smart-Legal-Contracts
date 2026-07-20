import React, { useState, useEffect } from 'react';
import { CalendarDaysIcon, ArrowTrendingUpIcon, ChartBarIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import AdminLayout from '../../src/components/admin/AdminLayout';
import AnalyticsChart from '../../src/components/admin/AnalyticsChart';
import MetricsCard from '../../src/components/admin/MetricsCard';
import GeographicChart from '../../src/components/admin/GeographicChart';
import UsageHeatmap from '../../src/components/admin/UsageHeatmap';
import RealtimeMetrics from '../../src/components/admin/RealtimeMetrics';

interface AnalyticsData {
  totalAnalyses: number;
  accuracyRate: number;
  avgProcessingTime: number;
  arbitrationDetectionRate: number;
  userGrowthRate: number;
  revenueGrowthRate: number;
  systemUptime: number;
  apiRequestsToday: number;
}

interface ChartData {
  labels: string[];
  datasets: any[];
}

const AnalyticsPage: React.FC = () => {
  const [analytics, setAnalytics] = useState<AnalyticsData>({
    totalAnalyses: 0,
    accuracyRate: 0,
    avgProcessingTime: 0,
    arbitrationDetectionRate: 0,
    userGrowthRate: 0,
    revenueGrowthRate: 0,
    systemUptime: 0,
    apiRequestsToday: 0
  });
  const [usageChart, setUsageChart] = useState<ChartData | null>(null);
  const [accuracyChart, setAccuracyChart] = useState<ChartData | null>(null);
  const [revenueChart, setRevenueChart] = useState<ChartData | null>(null);
  const [dateRange, setDateRange] = useState('7d');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchAnalytics();
  }, [dateRange]);

  const fetchAnalytics = async () => {
    setIsLoading(true);
    // Mock API call - replace with actual implementation
    setTimeout(() => {
      // Generate mock data based on date range
      const days = dateRange === '7d' ? 7 : dateRange === '30d' ? 30 : 90;
      const labels = Array.from({ length: days }, (_, i) => {
        const date = new Date();
        date.setDate(date.getDate() - (days - 1 - i));
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      });

      setAnalytics({
        totalAnalyses: 45632 + Math.floor(Math.random() * 1000),
        accuracyRate: 94.5 + Math.random() * 3,
        avgProcessingTime: 2.3 + Math.random() * 0.5,
        arbitrationDetectionRate: 23.7 + Math.random() * 5,
        userGrowthRate: 12.5 + Math.random() * 5,
        revenueGrowthRate: 18.3 + Math.random() * 7,
        systemUptime: 99.8 + Math.random() * 0.2,
        apiRequestsToday: 8934 + Math.floor(Math.random() * 1000)
      });

      // Usage chart data
      setUsageChart({
        labels,
        datasets: [
          {
            label: 'Document Analyses',
            data: labels.map(() => Math.floor(Math.random() * 200) + 50),
            borderColor: '#3B82F6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true
          },
          {
            label: 'API Requests',
            data: labels.map(() => Math.floor(Math.random() * 500) + 100),
            borderColor: '#10B981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true
          }
        ]
      });

      // Accuracy chart data
      setAccuracyChart({
        labels,
        datasets: [
          {
            label: 'Detection Accuracy',
            data: labels.map(() => 92 + Math.random() * 6),
            borderColor: '#8B5CF6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            fill: true
          },
          {
            label: 'Processing Speed',
            data: labels.map(() => 1.5 + Math.random() * 2),
            borderColor: '#F59E0B',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            fill: true
          }
        ]
      });

      // Revenue chart data
      setRevenueChart({
        labels,
        datasets: [
          {
            label: 'Revenue',
            data: labels.map(() => Math.floor(Math.random() * 10000) + 5000),
            borderColor: '#06B6D4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            fill: true
          }
        ]
      });

      setIsLoading(false);
    }, 1000);
  };

  const handleExportReport = () => {
    // Generate and download analytics report
    console.log('Exporting analytics report...');
  };

  const metricsCards = [
    {
      id: 'total-analyses',
      title: 'Total Analyses',
      value: analytics.totalAnalyses.toLocaleString(),
      change: `+${analytics.userGrowthRate.toFixed(1)}%`,
      trend: 'up' as const,
      icon: <ChartBarIcon className="h-6 w-6" id="total-analyses-icon" />,
      description: 'vs last period'
    },
    {
      id: 'accuracy-rate',
      title: 'Accuracy Rate',
      value: `${analytics.accuracyRate.toFixed(1)}%`,
      change: '+2.3%',
      trend: 'up' as const,
      icon: <ArrowTrendingUpIcon className="h-6 w-6" id="accuracy-rate-icon" />,
      description: 'detection accuracy'
    },
    {
      id: 'avg-processing-time',
      title: 'Avg Processing Time',
      value: `${analytics.avgProcessingTime.toFixed(1)}s`,
      change: '-0.2s',
      trend: 'up' as const,
      icon: <CalendarDaysIcon className="h-6 w-6" id="processing-time-icon" />,
      description: 'per document'
    },
    {
      id: 'arbitration-rate',
      title: 'Arbitration Detection',
      value: `${analytics.arbitrationDetectionRate.toFixed(1)}%`,
      change: `+${(analytics.arbitrationDetectionRate - 20).toFixed(1)}%`,
      trend: 'up' as const,
      icon: <ChartBarIcon className="h-6 w-6" id="arbitration-rate-icon" />,
      description: 'detection rate'
    }
  ];

  return (
    <AdminLayout>
      <div className="space-y-6" id="analytics-page-container">
        {/* Header */}
        <div className="sm:flex sm:items-center sm:justify-between" id="analytics-header">
          <div id="analytics-title-section">
            <h1 className="text-2xl font-semibold text-gray-900" id="analytics-title">
              Analytics Dashboard
            </h1>
            <p className="mt-1 text-sm text-gray-500" id="analytics-subtitle">
              System performance and usage insights
            </p>
          </div>
          <div className="mt-4 flex space-x-3 sm:mt-0" id="analytics-actions">
            <select
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-blue-500 focus:border-blue-500"
              id="date-range-selector"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </select>
            <button
              onClick={handleExportReport}
              className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              id="export-report-btn"
            >
              <ArrowDownTrayIcon className="h-4 w-4 mr-2" id="export-icon" />
              Export Report
            </button>
          </div>
        </div>

        {/* Real-time Metrics */}
        <div className="bg-white rounded-lg border border-gray-200 p-6" id="realtime-metrics-container">
          <h3 className="text-lg font-medium text-gray-900 mb-4" id="realtime-metrics-title">
            Real-time Metrics
          </h3>
          <RealtimeMetrics />
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4" id="key-metrics-grid">
          {metricsCards.map((card) => (
            <MetricsCard
              key={card.id}
              {...card}
              isLoading={isLoading}
            />
          ))}
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2" id="charts-section">
          {/* Usage Analytics */}
          <div className="bg-white rounded-lg border border-gray-200 p-6" id="usage-analytics-container">
            <div className="flex items-center justify-between mb-4" id="usage-chart-header">
              <h3 className="text-lg font-medium text-gray-900" id="usage-chart-title">
                Usage Analytics
              </h3>
              <div className="flex space-x-2" id="usage-chart-legend">
                <div className="flex items-center" id="documents-legend">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mr-2" id="documents-legend-color"></div>
                  <span className="text-sm text-gray-600" id="documents-legend-text">Documents</span>
                </div>
                <div className="flex items-center" id="api-legend">
                  <div className="w-3 h-3 bg-green-500 rounded-full mr-2" id="api-legend-color"></div>
                  <span className="text-sm text-gray-600" id="api-legend-text">API Requests</span>
                </div>
              </div>
            </div>
            {usageChart && <AnalyticsChart data={usageChart} type="line" />}
          </div>

          {/* Accuracy Trends */}
          <div className="bg-white rounded-lg border border-gray-200 p-6" id="accuracy-trends-container">
            <div className="flex items-center justify-between mb-4" id="accuracy-chart-header">
              <h3 className="text-lg font-medium text-gray-900" id="accuracy-chart-title">
                Performance Trends
              </h3>
              <div className="flex space-x-2" id="accuracy-chart-legend">
                <div className="flex items-center" id="accuracy-legend">
                  <div className="w-3 h-3 bg-purple-500 rounded-full mr-2" id="accuracy-legend-color"></div>
                  <span className="text-sm text-gray-600" id="accuracy-legend-text">Accuracy (%)</span>
                </div>
                <div className="flex items-center" id="speed-legend">
                  <div className="w-3 h-3 bg-amber-500 rounded-full mr-2" id="speed-legend-color"></div>
                  <span className="text-sm text-gray-600" id="speed-legend-text">Speed (s)</span>
                </div>
              </div>
            </div>
            {accuracyChart && <AnalyticsChart data={accuracyChart} type="line" />}
          </div>
        </div>

        {/* Revenue and Geographic Data */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3" id="revenue-geographic-section">
          {/* Revenue Chart */}
          <div className="lg:col-span-2 bg-white rounded-lg border border-gray-200 p-6" id="revenue-chart-container">
            <h3 className="text-lg font-medium text-gray-900 mb-4" id="revenue-chart-title">
              Revenue Trends
            </h3>
            {revenueChart && <AnalyticsChart data={revenueChart} type="bar" />}
          </div>

          {/* Geographic Distribution */}
          <div className="bg-white rounded-lg border border-gray-200 p-6" id="geographic-chart-container">
            <h3 className="text-lg font-medium text-gray-900 mb-4" id="geographic-chart-title">
              Geographic Usage
            </h3>
            <GeographicChart />
          </div>
        </div>

        {/* Usage Heatmap */}
        <div className="bg-white rounded-lg border border-gray-200 p-6" id="usage-heatmap-container">
          <h3 className="text-lg font-medium text-gray-900 mb-4" id="usage-heatmap-title">
            Usage Patterns Heatmap
          </h3>
          <p className="text-sm text-gray-600 mb-4" id="usage-heatmap-description">
            Document analysis activity by hour and day of the week
          </p>
          <UsageHeatmap />
        </div>

        {/* System Performance Metrics */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3" id="system-metrics-section">
          <div className="bg-white rounded-lg border border-gray-200 p-6" id="uptime-metric">
            <div className="flex items-center justify-between" id="uptime-header">
              <h4 className="text-sm font-medium text-gray-900" id="uptime-title">System Uptime</h4>
              <div className="w-2 h-2 bg-green-500 rounded-full" id="uptime-indicator"></div>
            </div>
            <div className="mt-2" id="uptime-content">
              <div className="text-3xl font-semibold text-gray-900" id="uptime-value">
                {analytics.systemUptime.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500 mt-1" id="uptime-description">
                Last 30 days
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6" id="api-requests-metric">
            <div className="flex items-center justify-between" id="api-requests-header">
              <h4 className="text-sm font-medium text-gray-900" id="api-requests-title">API Requests Today</h4>
              <ArrowTrendingUpIcon className="h-5 w-5 text-green-500" id="api-requests-trend-icon" />
            </div>
            <div className="mt-2" id="api-requests-content">
              <div className="text-3xl font-semibold text-gray-900" id="api-requests-value">
                {analytics.apiRequestsToday.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500 mt-1" id="api-requests-description">
                +{((analytics.apiRequestsToday / 8000 - 1) * 100).toFixed(1)}% vs yesterday
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6" id="error-rate-metric">
            <div className="flex items-center justify-between" id="error-rate-header">
              <h4 className="text-sm font-medium text-gray-900" id="error-rate-title">Error Rate</h4>
              <div className="w-2 h-2 bg-green-500 rounded-full" id="error-rate-indicator"></div>
            </div>
            <div className="mt-2" id="error-rate-content">
              <div className="text-3xl font-semibold text-gray-900" id="error-rate-value">
                0.1%
              </div>
              <div className="text-sm text-gray-500 mt-1" id="error-rate-description">
                Below threshold
              </div>
            </div>
          </div>
        </div>
      </div>
    </AdminLayout>
  );
};

export default AnalyticsPage;