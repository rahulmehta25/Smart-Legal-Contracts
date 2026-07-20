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

interface TransactionDetailsProps {
  transaction: Transaction;
  expanded?: boolean;
  onClick?: () => void;
}

export const TransactionDetails: React.FC<TransactionDetailsProps> = ({
  transaction,
  expanded = false,
  onClick
}) => {
  const formatHash = (hash: string) => {
    return `${hash.slice(0, 10)}...${hash.slice(-8)}`;
  };

  const formatAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const getStatusColor = (status: string) => {
    const colors = {
      'confirmed': 'bg-green-100 text-green-800',
      'pending': 'bg-yellow-100 text-yellow-800',
      'failed': 'bg-red-100 text-red-800'
    };
    return colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const formatGasPrice = (gasPrice: string) => {
    try {
      const gwei = parseFloat(gasPrice) / 1e9;
      return gwei.toFixed(2);
    } catch {
      return '0.00';
    }
  };

  const calculateTransactionFee = () => {
    try {
      const gasPriceWei = parseFloat(transaction.gasPrice);
      const gasUsed = transaction.gas; // In expanded view, this might be actual gas used
      const feeWei = gasPriceWei * gasUsed;
      const feeEth = feeWei / 1e18;
      return feeEth.toFixed(6);
    } catch {
      return '0.000000';
    }
  };

  return (
    <div
      id={`transaction-${transaction.hash.slice(0, 10)}`}
      className={`bg-white border border-gray-200 rounded-lg transition-all duration-200 ${
        onClick ? 'cursor-pointer hover:shadow-md hover:border-blue-300' : ''
      }`}
      onClick={onClick}
    >
      {/* Transaction Header */}
      <div id={`tx-header-${transaction.hash.slice(0, 10)}`} className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div id="tx-hash-section" className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-900">
              {expanded ? transaction.hash : formatHash(transaction.hash)}
            </span>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(transaction.status)}`}>
              {transaction.status}
            </span>
          </div>
          
          <div id="tx-timestamp" className="text-sm text-gray-600">
            {formatDistanceToNow(new Date(transaction.timestamp * 1000), { addSuffix: true })}
          </div>
        </div>

        {/* Transaction Flow */}
        <div id="tx-flow" className="flex items-center justify-between mb-3">
          <div id="tx-from" className="flex items-center space-x-2">
            <span className="text-xs text-gray-600">From:</span>
            <span className="text-sm font-mono text-gray-900">
              {expanded ? transaction.from : formatAddress(transaction.from)}
            </span>
          </div>
          
          <div id="tx-arrow" className="text-gray-400">
            →
          </div>
          
          <div id="tx-to" className="flex items-center space-x-2">
            <span className="text-xs text-gray-600">To:</span>
            <span className="text-sm font-mono text-gray-900">
              {transaction.to 
                ? (expanded ? transaction.to : formatAddress(transaction.to))
                : 'Contract Creation'
              }
            </span>
          </div>
        </div>

        {/* Value and Basic Info */}
        <div id="tx-basic-info" className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div id="tx-value" className="text-center">
            <div className="text-xs text-gray-600">Value</div>
            <div className="text-sm font-medium text-gray-900">
              {parseFloat(transaction.value).toFixed(4)} ETH
            </div>
          </div>
          
          <div id="tx-block" className="text-center">
            <div className="text-xs text-gray-600">Block</div>
            <div className="text-sm font-medium text-gray-900">
              #{transaction.blockNumber}
            </div>
          </div>
          
          <div id="tx-gas-price" className="text-center">
            <div className="text-xs text-gray-600">Gas Price</div>
            <div className="text-sm font-medium text-gray-900">
              {formatGasPrice(transaction.gasPrice)} Gwei
            </div>
          </div>
          
          <div id="tx-fee" className="text-center">
            <div className="text-xs text-gray-600">Fee</div>
            <div className="text-sm font-medium text-gray-900">
              {calculateTransactionFee()} ETH
            </div>
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div id="tx-expanded-details" className="border-t border-gray-200 p-4 bg-gray-50">
          {/* Detailed Information */}
          <div id="tx-detailed-info" className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div id="tx-info-left">
              <h4 className="text-sm font-medium text-gray-900 mb-3">Transaction Details</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Hash:</span>
                  <span className="font-mono text-xs break-all">{transaction.hash}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Status:</span>
                  <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(transaction.status)}`}>
                    {transaction.status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Block Number:</span>
                  <span>{transaction.blockNumber}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Timestamp:</span>
                  <span>{new Date(transaction.timestamp * 1000).toLocaleString()}</span>
                </div>
              </div>
            </div>

            <div id="tx-gas-details">
              <h4 className="text-sm font-medium text-gray-900 mb-3">Gas & Fee Details</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Gas Limit:</span>
                  <span>{transaction.gas.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Gas Price:</span>
                  <span>{formatGasPrice(transaction.gasPrice)} Gwei</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Transaction Fee:</span>
                  <span>{calculateTransactionFee()} ETH</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Value:</span>
                  <span>{parseFloat(transaction.value).toFixed(6)} ETH</span>
                </div>
              </div>
            </div>
          </div>

          {/* Address Information */}
          <div id="tx-addresses" className="mb-6">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Address Information</h4>
            <div className="space-y-3">
              <div id="tx-from-detail" className="bg-white border border-gray-200 rounded-md p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-xs text-gray-600">From</span>
                    <div className="font-mono text-sm text-gray-900">{transaction.from}</div>
                  </div>
                  <button
                    id="copy-from-address"
                    onClick={(e) => {
                      e.stopPropagation();
                      navigator.clipboard.writeText(transaction.from);
                    }}
                    className="text-blue-600 hover:text-blue-800 text-sm"
                  >
                    Copy
                  </button>
                </div>
              </div>

              {transaction.to && (
                <div id="tx-to-detail" className="bg-white border border-gray-200 rounded-md p-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-xs text-gray-600">To</span>
                      <div className="font-mono text-sm text-gray-900">{transaction.to}</div>
                    </div>
                    <button
                      id="copy-to-address"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigator.clipboard.writeText(transaction.to);
                      }}
                      className="text-blue-600 hover:text-blue-800 text-sm"
                    >
                      Copy
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Transaction Analysis */}
          <div id="tx-analysis" className="mb-6">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Transaction Analysis</h4>
            <div className="bg-white border border-gray-200 rounded-md p-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div id="tx-type-analysis">
                  <span className="text-gray-600">Type:</span>
                  <span className="ml-2">
                    {transaction.to ? 'Transfer' : 'Contract Creation'}
                  </span>
                </div>
                <div id="tx-size-analysis">
                  <span className="text-gray-600">Estimated Size:</span>
                  <span className="ml-2">
                    {Math.ceil(transaction.hash.length / 2)} bytes
                  </span>
                </div>
                <div id="tx-priority-analysis">
                  <span className="text-gray-600">Priority:</span>
                  <span className="ml-2">
                    {parseFloat(formatGasPrice(transaction.gasPrice)) > 50 ? 'High' : 
                     parseFloat(formatGasPrice(transaction.gasPrice)) > 20 ? 'Medium' : 'Low'}
                  </span>
                </div>
                <div id="tx-efficiency-analysis">
                  <span className="text-gray-600">Gas Efficiency:</span>
                  <span className="ml-2">
                    {transaction.gas < 21000 ? 'Standard' : 
                     transaction.gas < 100000 ? 'Complex' : 'Heavy'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div id="tx-actions" className="pt-4 border-t border-gray-200">
            <div className="flex items-center space-x-3">
              <button
                id="view-on-explorer"
                onClick={(e) => {
                  e.stopPropagation();
                  window.open(`https://etherscan.io/tx/${transaction.hash}`, '_blank');
                }}
                className="text-sm text-blue-600 hover:text-blue-800 underline"
              >
                View on Etherscan
              </button>
              <button
                id="copy-tx-hash"
                onClick={(e) => {
                  e.stopPropagation();
                  navigator.clipboard.writeText(transaction.hash);
                }}
                className="text-sm text-gray-600 hover:text-gray-800 underline"
              >
                Copy Hash
              </button>
              <button
                id="view-block"
                onClick={(e) => {
                  e.stopPropagation();
                  window.open(`https://etherscan.io/block/${transaction.blockNumber}`, '_blank');
                }}
                className="text-sm text-gray-600 hover:text-gray-800 underline"
              >
                View Block
              </button>
              {transaction.to && (
                <button
                  id="view-address"
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open(`https://etherscan.io/address/${transaction.to}`, '_blank');
                  }}
                  className="text-sm text-gray-600 hover:text-gray-800 underline"
                >
                  View Address
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};