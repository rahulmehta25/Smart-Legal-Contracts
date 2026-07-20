import React from 'react';
import { formatDistanceToNow } from 'date-fns';

interface Transaction {
  hash: string;
  from: string;
  to: string;
  value: string;
  gas: number;
  gasPrice: string;
  blockNumber: number;
  status: 'pending' | 'confirmed' | 'failed';
  timestamp: number;
}

interface Block {
  number: number;
  hash: string;
  timestamp: number;
  transactions: Transaction[];
  gasUsed: number;
  gasLimit: number;
  miner: string;
  size: number;
}

interface BlockDetailsProps {
  block: Block;
  expanded?: boolean;
  onClick?: () => void;
}

export const BlockDetails: React.FC<BlockDetailsProps> = ({ 
  block, 
  expanded = false, 
  onClick 
}) => {
  const formatHash = (hash: string) => {
    return `${hash.slice(0, 10)}...${hash.slice(-8)}`;
  };

  const formatGasUsage = () => {
    const percentage = (block.gasUsed / block.gasLimit) * 100;
    return {
      percentage: percentage.toFixed(1),
      color: percentage > 80 ? 'bg-red-500' : percentage > 60 ? 'bg-yellow-500' : 'bg-green-500'
    };
  };

  const gasUsage = formatGasUsage();

  return (
    <div 
      id={`block-${block.number}`}
      className={`bg-white border border-gray-200 rounded-lg transition-all duration-200 ${
        onClick ? 'cursor-pointer hover:shadow-md hover:border-blue-300' : ''
      }`}
      onClick={onClick}
    >
      {/* Block Header */}
      <div id={`block-header-${block.number}`} className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div id={`block-number-${block.number}`} className="flex items-center space-x-2">
            <span className="text-lg font-semibold text-gray-900">
              Block #{block.number}
            </span>
            <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
              {block.transactions.length} txns
            </span>
          </div>
          
          <div id={`block-timestamp-${block.number}`} className="text-sm text-gray-600">
            {formatDistanceToNow(new Date(block.timestamp * 1000), { addSuffix: true })}
          </div>
        </div>

        <div id={`block-hash-${block.number}`} className="mb-3">
          <span className="text-sm text-gray-600">Hash: </span>
          <span className="text-sm font-mono text-gray-900">
            {expanded ? block.hash : formatHash(block.hash)}
          </span>
        </div>

        {/* Quick Stats */}
        <div id={`block-stats-${block.number}`} className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div id={`block-miner-${block.number}`} className="text-center">
            <div className="text-xs text-gray-600">Miner</div>
            <div className="text-sm font-medium text-gray-900">
              {formatHash(block.miner)}
            </div>
          </div>
          
          <div id={`block-size-${block.number}`} className="text-center">
            <div className="text-xs text-gray-600">Size</div>
            <div className="text-sm font-medium text-gray-900">
              {(block.size / 1024).toFixed(1)} KB
            </div>
          </div>
          
          <div id={`block-gas-used-${block.number}`} className="text-center">
            <div className="text-xs text-gray-600">Gas Used</div>
            <div className="text-sm font-medium text-gray-900">
              {(block.gasUsed / 1000000).toFixed(2)}M
            </div>
          </div>
          
          <div id={`block-gas-percentage-${block.number}`} className="text-center">
            <div className="text-xs text-gray-600">Gas Usage</div>
            <div className="flex items-center justify-center space-x-1">
              <div className="w-12 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className={`h-full ${gasUsage.color} transition-all duration-300`}
                  style={{ width: `${gasUsage.percentage}%` }}
                />
              </div>
              <span className="text-xs text-gray-600">{gasUsage.percentage}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div id={`block-details-${block.number}`} className="border-t border-gray-200 p-4 bg-gray-50">
          {/* Detailed Information */}
          <div id={`block-info-${block.number}`} className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div id={`block-info-left-${block.number}`}>
              <h4 className="text-sm font-medium text-gray-900 mb-3">Block Information</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Block Number:</span>
                  <span className="font-mono">{block.number}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Timestamp:</span>
                  <span>{new Date(block.timestamp * 1000).toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Hash:</span>
                  <span className="font-mono text-xs break-all">{block.hash}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Miner:</span>
                  <span className="font-mono text-xs">{block.miner}</span>
                </div>
              </div>
            </div>

            <div id={`block-gas-info-${block.number}`}>
              <h4 className="text-sm font-medium text-gray-900 mb-3">Gas Information</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Gas Used:</span>
                  <span>{block.gasUsed.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Gas Limit:</span>
                  <span>{block.gasLimit.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Gas Utilization:</span>
                  <span>{gasUsage.percentage}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Block Size:</span>
                  <span>{(block.size / 1024).toFixed(2)} KB</span>
                </div>
              </div>
            </div>
          </div>

          {/* Transactions */}
          {block.transactions.length > 0 && (
            <div id={`block-transactions-${block.number}`}>
              <h4 className="text-sm font-medium text-gray-900 mb-3">
                Transactions ({block.transactions.length})
              </h4>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {block.transactions.slice(0, 10).map((tx, index) => (
                  <div
                    key={tx.hash}
                    id={`block-tx-${block.number}-${index}`}
                    className="bg-white border border-gray-200 rounded-md p-3"
                  >
                    <div className="flex items-center justify-between">
                      <div id={`tx-hash-${index}`} className="flex-1">
                        <span className="text-xs text-gray-600">Hash: </span>
                        <span className="text-xs font-mono text-blue-600">
                          {formatHash(tx.hash)}
                        </span>
                      </div>
                      <div id={`tx-value-${index}`} className="text-right">
                        <span className="text-sm font-medium">
                          {parseFloat(tx.value).toFixed(4)} ETH
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <div id={`tx-addresses-${index}`} className="text-xs text-gray-600">
                        {formatHash(tx.from)} → {tx.to ? formatHash(tx.to) : 'Contract Creation'}
                      </div>
                      <div id={`tx-status-${index}`}>
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          tx.status === 'confirmed' 
                            ? 'bg-green-100 text-green-800'
                            : tx.status === 'failed'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-yellow-100 text-yellow-800'
                        }`}>
                          {tx.status}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
                
                {block.transactions.length > 10 && (
                  <div id={`more-transactions-${block.number}`} className="text-center py-2">
                    <span className="text-sm text-gray-600">
                      ... and {block.transactions.length - 10} more transactions
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Actions */}
          <div id={`block-actions-${block.number}`} className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex items-center space-x-3">
              <button
                id={`view-on-explorer-${block.number}`}
                onClick={(e) => {
                  e.stopPropagation();
                  window.open(`https://etherscan.io/block/${block.number}`, '_blank');
                }}
                className="text-sm text-blue-600 hover:text-blue-800 underline"
              >
                View on Etherscan
              </button>
              <button
                id={`copy-hash-${block.number}`}
                onClick={(e) => {
                  e.stopPropagation();
                  navigator.clipboard.writeText(block.hash);
                }}
                className="text-sm text-gray-600 hover:text-gray-800 underline"
              >
                Copy Hash
              </button>
              <button
                id={`copy-number-${block.number}`}
                onClick={(e) => {
                  e.stopPropagation();
                  navigator.clipboard.writeText(block.number.toString());
                }}
                className="text-sm text-gray-600 hover:text-gray-800 underline"
              >
                Copy Number
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};