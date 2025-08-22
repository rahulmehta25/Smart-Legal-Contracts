import React, { useState, useEffect } from 'react';
import { useWeb3 } from '../hooks/useWeb3';
import { useBlockchain } from '../hooks/useBlockchain';
import { BlockDetails } from '../components/BlockDetails';
import { TransactionDetails } from '../components/TransactionDetails';
import { AuditTrailViewer } from '../components/AuditTrailViewer';
import { NetworkStats } from '../components/NetworkStats';
import { SearchBar } from '../components/SearchBar';
import { Alert } from '../components/Alert';

interface ExplorerProps {
  networkId?: string;
  contractAddress?: string;
}

export const BlockchainExplorer: React.FC<ExplorerProps> = ({
  networkId = 'ethereum',
  contractAddress
}) => {
  const [activeTab, setActiveTab] = useState<'blocks' | 'transactions' | 'audit' | 'analytics'>('blocks');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { web3, account, isConnected, connectWallet } = useWeb3();
  const { 
    latestBlocks, 
    recentTransactions, 
    auditRecords, 
    networkStats,
    searchBlockchain,
    subscribeToEvents
  } = useBlockchain(networkId, contractAddress);

  useEffect(() => {
    // Subscribe to real-time blockchain events
    const unsubscribe = subscribeToEvents({
      onNewBlock: (block) => {
        console.log('New block received:', block);
      },
      onNewTransaction: (tx) => {
        console.log('New transaction received:', tx);
      },
      onAuditRecord: (record) => {
        console.log('New audit record:', record);
      }
    });

    return unsubscribe;
  }, [subscribeToEvents]);

  const handleSearch = async (query: string) => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const results = await searchBlockchain(query);
      setSearchResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'blocks':
        return (
          <div id="blocks-tab-content" className="space-y-6">
            <div id="latest-blocks-section">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Latest Blocks</h3>
              <div id="blocks-grid" className="grid gap-4">
                {latestBlocks.map((block) => (
                  <BlockDetails key={block.number} block={block} />
                ))}
              </div>
            </div>
          </div>
        );

      case 'transactions':
        return (
          <div id="transactions-tab-content" className="space-y-6">
            <div id="recent-transactions-section">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Transactions</h3>
              <div id="transactions-list" className="space-y-3">
                {recentTransactions.map((tx) => (
                  <TransactionDetails key={tx.hash} transaction={tx} />
                ))}
              </div>
            </div>
          </div>
        );

      case 'audit':
        return (
          <div id="audit-tab-content" className="space-y-6">
            <AuditTrailViewer 
              auditRecords={auditRecords}
              contractAddress={contractAddress}
              onRecordClick={(record) => {
                console.log('Audit record clicked:', record);
              }}
            />
          </div>
        );

      case 'analytics':
        return (
          <div id="analytics-tab-content" className="space-y-6">
            <NetworkStats 
              stats={networkStats}
              networkId={networkId}
              refreshInterval={30000}
            />
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div id="blockchain-explorer-container" className="min-h-screen bg-gray-50">
      {/* Header */}
      <div id="explorer-header" className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div id="header-content" className="flex items-center justify-between">
            <div id="header-left">
              <h1 className="text-2xl font-bold text-gray-900">Blockchain Explorer</h1>
              <p className="text-sm text-gray-600 mt-1">
                Network: <span className="capitalize font-medium">{networkId}</span>
              </p>
            </div>
            
            <div id="header-right" className="flex items-center space-x-4">
              {/* Wallet Connection */}
              <div id="wallet-connection">
                {isConnected ? (
                  <div id="connected-wallet" className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm text-gray-700">
                      {account?.slice(0, 6)}...{account?.slice(-4)}
                    </span>
                  </div>
                ) : (
                  <button
                    id="connect-wallet-btn"
                    onClick={connectWallet}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Connect Wallet
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Search Bar */}
          <div id="search-section" className="mt-4">
            <SearchBar
              onSearch={handleSearch}
              loading={loading}
              placeholder="Search by block number, transaction hash, or address..."
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div id="explorer-main" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Error Alert */}
        {error && (
          <div id="error-alert" className="mb-6">
            <Alert
              type="error"
              title="Error"
              message={error}
              onDismiss={() => setError(null)}
            />
          </div>
        )}

        {/* Search Results */}
        {searchResults && (
          <div id="search-results" className="mb-6">
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Search Results</h3>
              {searchResults.type === 'block' && (
                <BlockDetails block={searchResults.data} expanded />
              )}
              {searchResults.type === 'transaction' && (
                <TransactionDetails transaction={searchResults.data} expanded />
              )}
              {searchResults.type === 'address' && (
                <div id="address-details">
                  <p className="text-sm text-gray-600">Address: {searchResults.data.address}</p>
                  <p className="text-sm text-gray-600">Balance: {searchResults.data.balance} ETH</p>
                  <p className="text-sm text-gray-600">
                    Transaction Count: {searchResults.data.transactionCount}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div id="tab-navigation" className="mb-6">
          <nav className="flex space-x-1 bg-gray-100 rounded-lg p-1">
            {[
              { id: 'blocks', label: 'Blocks', icon: '🧱' },
              { id: 'transactions', label: 'Transactions', icon: '📝' },
              { id: 'audit', label: 'Audit Trail', icon: '🔍' },
              { id: 'analytics', label: 'Analytics', icon: '📊' }
            ].map((tab) => (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div id="tab-content-container" className="bg-white rounded-lg shadow-sm border p-6">
          {renderTabContent()}
        </div>

        {/* Real-time Status */}
        <div id="realtime-status" className="mt-6">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2"></div>
              <span className="text-sm text-green-800">
                Real-time monitoring active • Last update: {new Date().toLocaleTimeString()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer id="explorer-footer" className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div id="footer-left">
              <p>Blockchain Audit Trail Explorer</p>
            </div>
            <div id="footer-right" className="flex items-center space-x-4">
              <span>Network: {networkId}</span>
              {contractAddress && (
                <span>Contract: {contractAddress.slice(0, 6)}...{contractAddress.slice(-4)}</span>
              )}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};