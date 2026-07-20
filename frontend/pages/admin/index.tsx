import React, { useState, useEffect } from 'react';
import { ChartBarIcon, UsersIcon, DocumentTextIcon, CogIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import AdminLayout from '../../src/components/admin/AdminLayout';
import StatsCard from '../../src/components/admin/StatsCard';
import ActivityFeed from '../../src/components/admin/ActivityFeed';
import SystemHealth from '../../src/components/admin/SystemHealth';
import QuickActions from '../../src/components/admin/QuickActions';
import UsageChart from '../../src/components/admin/UsageChart';
import RevenueChart from '../../src/components/admin/RevenueChart';

interface DashboardStats {
  totalUsers: number;
  activeUsers: number;
  totalDocuments: number;
  analysisCount: number;
  revenue: number;
  systemHealth: number;
}

const AdminDashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats>({
    totalUsers: 0,
    activeUsers: 0,
    totalDocuments: 0,
    analysisCount: 0,
    revenue: 0,
    systemHealth: 0
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate API call
    const fetchStats = async () => {
      setIsLoading(true);
      // Mock data - replace with actual API calls
      setTimeout(() => {
        setStats({
          totalUsers: 12847,
          activeUsers: 3892,
          totalDocuments: 45632,
          analysisCount: 8937,
          revenue: 234567,
          systemHealth: 99.2
        });
        setIsLoading(false);
      }, 1000);
    };

    fetchStats();
    
    // Set up real-time updates
    const interval = setInterval(fetchStats, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const statsCards = [
    {
      id: 'total-users',
      title: 'Total Users',
      value: stats.totalUsers.toLocaleString(),
      change: '+12.5%',
      trend: 'up' as const,
      icon: <UsersIcon className="h-6 w-6" id="total-users-icon" />,
      description: 'vs last month'
    },
    {
      id: 'active-users',
      title: 'Active Users',
      value: stats.activeUsers.toLocaleString(),
      change: '+8.2%',
      trend: 'up' as const,
      icon: <UsersIcon className="h-6 w-6" id="active-users-icon" />,
      description: 'last 30 days'
    },
    {
      id: 'total-documents',
      title: 'Documents Processed',
      value: stats.totalDocuments.toLocaleString(),
      change: '+23.1%',
      trend: 'up' as const,
      icon: <DocumentTextIcon className="h-6 w-6" id="documents-icon" />,
      description: 'this month'
    },
    {
      id: 'revenue',
      title: 'Revenue',
      value: `$${(stats.revenue / 1000).toFixed(0)}k`,
      change: '+15.3%',
      trend: 'up' as const,
      icon: <ChartBarIcon className="h-6 w-6" id="revenue-icon" />,
      description: 'monthly recurring'
    }
  ];

  return (
    <AdminLayout>
      <div className="space-y-6" id="admin-dashboard-container">
        {/* Header */}
        <div className="border-b border-gray-200 pb-4" id="dashboard-header">
          <h1 className="text-2xl font-semibold text-gray-900" id="dashboard-title">
            Dashboard Overview
          </h1>
          <p className="mt-1 text-sm text-gray-500" id="dashboard-subtitle">
            Real-time insights and system status
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4" id="stats-grid">
          {statsCards.map((card) => (
            <StatsCard
              key={card.id}
              {...card}
              isLoading={isLoading}
            />
          ))}
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2" id="charts-section">
          <div className="bg-white rounded-lg border border-gray-200 p-6" id="usage-chart-container">
            <h3 className="text-lg font-medium text-gray-900 mb-4" id="usage-chart-title">
              Usage Analytics
            </h3>
            <UsageChart />
          </div>
          
          <div className="bg-white rounded-lg border border-gray-200 p-6" id="revenue-chart-container">
            <h3 className="text-lg font-medium text-gray-900 mb-4" id="revenue-chart-title">
              Revenue Trends
            </h3>
            <RevenueChart />
          </div>
        </div>

        {/* Bottom Section */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3" id="bottom-section">
          {/* System Health */}
          <div className="bg-white rounded-lg border border-gray-200" id="system-health-container">
            <SystemHealth />
          </div>

          {/* Activity Feed */}
          <div className="bg-white rounded-lg border border-gray-200" id="activity-feed-container">
            <ActivityFeed />
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-lg border border-gray-200" id="quick-actions-container">
            <QuickActions />
          </div>
        </div>

        {/* Alerts Section */}
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4" id="alerts-section">
          <div className="flex items-start" id="alert-content">
            <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400 mt-0.5 mr-3" id="alert-icon" />
            <div id="alert-text">
              <h4 className="text-sm font-medium text-yellow-800" id="alert-title">
                System Maintenance Scheduled
              </h4>
              <p className="mt-1 text-sm text-yellow-700" id="alert-message">
                Scheduled maintenance window: Tomorrow 2:00 AM - 4:00 AM UTC. 
                Expected downtime: 15 minutes.
              </p>
            </div>
          </div>
        </div>
      </div>
    </AdminLayout>
  );
};

export default AdminDashboard;