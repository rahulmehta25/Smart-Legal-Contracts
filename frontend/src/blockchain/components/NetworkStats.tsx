import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface NetworkStats {
  blockHeight: number;
  blockTime: number;
  gasPrice: number;
  peerCount: number;
  hashRate: number;
  difficulty: string;
  totalSupply: string;
  marketCap: string;
}

interface NetworkStatsProps {
  stats: NetworkStats | null;
  networkId: string;
  refreshInterval?: number;
}

interface HistoricalData {
  timestamps: string[];
  blockTimes: number[];
  gasPrices: number[];
  blockHeights: number[];
}

export const NetworkStats: React.FC<NetworkStatsProps> = ({
  stats,
  networkId,
  refreshInterval = 30000
}) => {
  const [historicalData, setHistoricalData] = useState<HistoricalData>({
    timestamps: [],
    blockTimes: [],
    gasPrices: [],
    blockHeights: []
  });
  const [selectedChart, setSelectedChart] = useState<'blockTime' | 'gasPrice' | 'blockHeight'>('blockTime');
  const [isLoading, setIsLoading] = useState(false);

  // Update historical data when stats change
  useEffect(() => {
    if (stats) {
      setHistoricalData(prev => {
        const now = new Date().toLocaleTimeString();
        const maxDataPoints = 50;

        const newData = {
          timestamps: [...prev.timestamps, now].slice(-maxDataPoints),
          blockTimes: [...prev.blockTimes, stats.blockTime].slice(-maxDataPoints),
          gasPrices: [...prev.gasPrices, stats.gasPrice].slice(-maxDataPoints),
          blockHeights: [...prev.blockHeights, stats.blockHeight].slice(-maxDataPoints)
        };

        return newData;
      });
    }
  }, [stats]);

  const getNetworkStatus = () => {
    if (!stats) return { status: 'offline', color: 'bg-gray-500' };
    
    if (stats.blockTime > 30) return { status: 'slow', color: 'bg-yellow-500' };
    if (stats.peerCount < 5) return { status: 'degraded', color: 'bg-orange-500' };
    if (stats.gasPrice > 100) return { status: 'congested', color: 'bg-red-500' };
    
    return { status: 'healthy', color: 'bg-green-500' };
  };

  const networkStatus = getNetworkStatus();

  const chartData = {
    labels: historicalData.timestamps,
    datasets: [
      {
        label: selectedChart === 'blockTime' ? 'Block Time (s)' : 
               selectedChart === 'gasPrice' ? 'Gas Price (Gwei)' : 'Block Height',
        data: selectedChart === 'blockTime' ? historicalData.blockTimes :
              selectedChart === 'gasPrice' ? historicalData.gasPrices : historicalData.blockHeights,
        borderColor: selectedChart === 'blockTime' ? 'rgb(59, 130, 246)' :
                     selectedChart === 'gasPrice' ? 'rgb(239, 68, 68)' : 'rgb(34, 197, 94)',
        backgroundColor: selectedChart === 'blockTime' ? 'rgba(59, 130, 246, 0.1)' :
                         selectedChart === 'gasPrice' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        fill: true,
        pointRadius: 2,
        pointHoverRadius: 4
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `${selectedChart === 'blockTime' ? 'Block Time' : 
               selectedChart === 'gasPrice' ? 'Gas Price' : 'Block Height'} Over Time`
      },
    },
    scales: {
      x: {
        display: false
      },
      y: {
        beginAtZero: selectedChart !== 'blockHeight',
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
    interaction: {
      intersect: false,
      mode: 'index' as const
    }
  };

  const formatNumber = (num: number, decimals: number = 2) => {
    if (num >= 1e9) return (num / 1e9).toFixed(decimals) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(decimals) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(decimals) + 'K';
    return num.toFixed(decimals);
  };

  if (!stats) {
    return (
      <div id="network-stats-loading" className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-gray-600">Loading network statistics...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div id="network-stats-container" className="space-y-6">
      {/* Network Overview */}
      <div id="network-overview" className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-6">
          <div id="network-header">
            <h2 className="text-xl font-semibold text-gray-900">Network Statistics</h2>
            <p className="text-sm text-gray-600">
              Real-time metrics for {networkId.charAt(0).toUpperCase() + networkId.slice(1)}
            </p>
          </div>
          
          <div id="network-status" className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${networkStatus.color}`}></div>
            <span className="text-sm font-medium text-gray-700 capitalize">
              {networkStatus.status}
            </span>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div id="key-metrics-grid" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div id="block-height-metric" className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-blue-600 font-medium">Block Height</p>
                <p className="text-2xl font-bold text-blue-900">
                  {formatNumber(stats.blockHeight, 0)}
                </p>
              </div>
              <div className="text-blue-500 text-2xl">🧱</div>
            </div>
          </div>

          <div id="block-time-metric" className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-600 font-medium">Block Time</p>
                <p className="text-2xl font-bold text-green-900">
                  {stats.blockTime.toFixed(1)}s
                </p>
              </div>
              <div className="text-green-500 text-2xl">⏱️</div>
            </div>
          </div>

          <div id="gas-price-metric" className="bg-gradient-to-r from-yellow-50 to-yellow-100 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-yellow-600 font-medium">Gas Price</p>
                <p className="text-2xl font-bold text-yellow-900">
                  {stats.gasPrice.toFixed(1)} Gwei
                </p>
              </div>
              <div className="text-yellow-500 text-2xl">⛽</div>
            </div>
          </div>

          <div id="peer-count-metric" className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-purple-600 font-medium">Peer Count</p>
                <p className="text-2xl font-bold text-purple-900">
                  {stats.peerCount}
                </p>
              </div>
              <div className="text-purple-500 text-2xl">🌐</div>
            </div>
          </div>
        </div>

        {/* Additional Metrics */}
        <div id="additional-metrics" className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <div id="hash-rate-metric" className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 font-medium">Network Hash Rate</p>
            <p className="text-xl font-bold text-gray-900">
              {stats.hashRate > 0 ? formatNumber(stats.hashRate) + ' H/s' : 'N/A'}
            </p>
          </div>

          <div id="difficulty-metric" className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 font-medium">Difficulty</p>
            <p className="text-xl font-bold text-gray-900">
              {stats.difficulty !== '0' ? formatNumber(parseFloat(stats.difficulty)) : 'N/A'}
            </p>
          </div>

          <div id="total-supply-metric" className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600 font-medium">Total Supply</p>
            <p className="text-xl font-bold text-gray-900">
              {stats.totalSupply !== '0' ? formatNumber(parseFloat(stats.totalSupply)) + ' ETH' : 'N/A'}
            </p>
          </div>
        </div>
      </div>

      {/* Historical Charts */}
      <div id="historical-charts" className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Historical Data</h3>
          
          <div id="chart-selector" className="flex space-x-1 bg-gray-100 rounded-lg p-1">
            {[
              { key: 'blockTime', label: 'Block Time', icon: '⏱️' },
              { key: 'gasPrice', label: 'Gas Price', icon: '⛽' },
              { key: 'blockHeight', label: 'Block Height', icon: '🧱' }
            ].map((chart) => (
              <button
                key={chart.key}
                id={`chart-select-${chart.key}`}
                onClick={() => setSelectedChart(chart.key as any)}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  selectedChart === chart.key
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <span>{chart.icon}</span>
                <span>{chart.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Chart Container */}
        <div id="chart-container" className="h-64">
          {historicalData.timestamps.length > 0 ? (
            <Line data={chartData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <p className="text-gray-500">Collecting data...</p>
                <p className="text-sm text-gray-400">Charts will appear as data is gathered</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Network Health Indicators */}
      <div id="network-health" className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Network Health Indicators</h3>
        
        <div id="health-indicators" className="space-y-4">
          <div id="block-time-health" className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <p className="text-sm font-medium text-gray-900">Block Time</p>
              <p className="text-xs text-gray-600">Target: ~12-15 seconds</p>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                stats.blockTime <= 15 ? 'bg-green-500' : 
                stats.blockTime <= 30 ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
              <span className="text-sm font-medium">{stats.blockTime.toFixed(1)}s</span>
            </div>
          </div>

          <div id="gas-price-health" className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <p className="text-sm font-medium text-gray-900">Gas Price</p>
              <p className="text-xs text-gray-600">Network congestion indicator</p>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                stats.gasPrice <= 50 ? 'bg-green-500' : 
                stats.gasPrice <= 100 ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
              <span className="text-sm font-medium">{stats.gasPrice.toFixed(1)} Gwei</span>
            </div>
          </div>

          <div id="peer-count-health" className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <p className="text-sm font-medium text-gray-900">Network Connectivity</p>
              <p className="text-xs text-gray-600">Connected peers</p>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                stats.peerCount >= 10 ? 'bg-green-500' : 
                stats.peerCount >= 5 ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
              <span className="text-sm font-medium">{stats.peerCount} peers</span>
            </div>
          </div>
        </div>
      </div>

      {/* Refresh Status */}
      <div id="refresh-status" className="text-center">
        <p className="text-sm text-gray-500">
          Auto-refreshing every {refreshInterval / 1000} seconds • 
          Last updated: {new Date().toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
};