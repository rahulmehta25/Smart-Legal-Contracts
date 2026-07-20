import React, { useState, useEffect } from 'react';
import { formatDistanceToNow } from 'date-fns';

interface AuditRecord {
  documentHash: string;
  analysisResult: any;
  timestamp: number;
  submitter: string;
  decisionType: string;
  complianceStatus: string;
  metadata: any;
  blockNumber: number;
  transactionHash: string;
  isVerified: boolean;
}

interface AuditTrailViewerProps {
  auditRecords: AuditRecord[];
  contractAddress?: string;
  onRecordClick?: (record: AuditRecord) => void;
  showFilters?: boolean;
}

export const AuditTrailViewer: React.FC<AuditTrailViewerProps> = ({
  auditRecords,
  contractAddress,
  onRecordClick,
  showFilters = true
}) => {
  const [filteredRecords, setFilteredRecords] = useState<AuditRecord[]>(auditRecords);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterDecisionType, setFilterDecisionType] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'timestamp' | 'blockNumber'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [expandedRecord, setExpandedRecord] = useState<string | null>(null);

  useEffect(() => {
    let filtered = [...auditRecords];

    // Apply filters
    if (filterStatus !== 'all') {
      filtered = filtered.filter(record => record.complianceStatus === filterStatus);
    }

    if (filterDecisionType !== 'all') {
      filtered = filtered.filter(record => record.decisionType === filterDecisionType);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let valueA = sortBy === 'timestamp' ? a.timestamp : a.blockNumber;
      let valueB = sortBy === 'timestamp' ? b.timestamp : b.blockNumber;

      if (sortOrder === 'asc') {
        return valueA - valueB;
      } else {
        return valueB - valueA;
      }
    });

    setFilteredRecords(filtered);
  }, [auditRecords, filterStatus, filterDecisionType, sortBy, sortOrder]);

  const getStatusColor = (status: string) => {
    const colors = {
      'compliant': 'bg-green-100 text-green-800',
      'non-compliant': 'bg-red-100 text-red-800',
      'pending': 'bg-yellow-100 text-yellow-800',
      'under-review': 'bg-blue-100 text-blue-800'
    };
    return colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const getDecisionTypeColor = (type: string) => {
    const colors = {
      'approved': 'bg-green-100 text-green-800',
      'rejected': 'bg-red-100 text-red-800',
      'conditional': 'bg-yellow-100 text-yellow-800',
      'escalated': 'bg-purple-100 text-purple-800'
    };
    return colors[type as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const toggleRecordExpansion = (documentHash: string) => {
    if (expandedRecord === documentHash) {
      setExpandedRecord(null);
    } else {
      setExpandedRecord(documentHash);
    }
  };

  const handleRecordClick = (record: AuditRecord) => {
    toggleRecordExpansion(record.documentHash);
    if (onRecordClick) {
      onRecordClick(record);
    }
  };

  const uniqueStatuses = [...new Set(auditRecords.map(r => r.complianceStatus))];
  const uniqueDecisionTypes = [...new Set(auditRecords.map(r => r.decisionType))];

  return (
    <div id="audit-trail-viewer" className="space-y-6">
      {/* Header */}
      <div id="audit-trail-header" className="flex items-center justify-between">
        <div id="header-left">
          <h2 className="text-xl font-semibold text-gray-900">Audit Trail</h2>
          <p className="text-sm text-gray-600">
            {filteredRecords.length} of {auditRecords.length} records
          </p>
        </div>

        <div id="header-actions" className="flex items-center space-x-3">
          {/* Sort Controls */}
          <div id="sort-controls" className="flex items-center space-x-2">
            <label className="text-sm text-gray-600">Sort by:</label>
            <select
              id="sort-by-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'timestamp' | 'blockNumber')}
              className="text-sm border border-gray-300 rounded-md px-2 py-1"
            >
              <option value="timestamp">Timestamp</option>
              <option value="blockNumber">Block Number</option>
            </select>
            <button
              id="sort-order-btn"
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              {sortOrder === 'asc' ? '↑' : '↓'}
            </button>
          </div>
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <div id="audit-trail-filters" className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div id="status-filter">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Compliance Status
              </label>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2"
              >
                <option value="all">All Statuses</option>
                {uniqueStatuses.map(status => (
                  <option key={status} value={status}>{status}</option>
                ))}
              </select>
            </div>

            <div id="decision-type-filter">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Decision Type
              </label>
              <select
                value={filterDecisionType}
                onChange={(e) => setFilterDecisionType(e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2"
              >
                <option value="all">All Types</option>
                {uniqueDecisionTypes.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>

            <div id="filter-actions" className="flex items-end">
              <button
                id="clear-filters-btn"
                onClick={() => {
                  setFilterStatus('all');
                  setFilterDecisionType('all');
                }}
                className="text-sm text-gray-600 hover:text-gray-800 underline"
              >
                Clear Filters
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Audit Records */}
      <div id="audit-records-list" className="space-y-3">
        {filteredRecords.length === 0 ? (
          <div id="no-records" className="text-center py-12">
            <div className="text-gray-400 text-lg mb-2">📋</div>
            <p className="text-gray-600">No audit records found</p>
          </div>
        ) : (
          filteredRecords.map((record, index) => (
            <div
              key={record.documentHash}
              id={`audit-record-${index}`}
              className="bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
            >
              {/* Record Header */}
              <div
                id={`record-header-${index}`}
                className="p-4 cursor-pointer"
                onClick={() => handleRecordClick(record)}
              >
                <div className="flex items-center justify-between">
                  <div id={`record-info-${index}`} className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <span className="text-sm font-mono text-gray-600">
                        {record.documentHash.slice(0, 10)}...{record.documentHash.slice(-8)}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(record.complianceStatus)}`}>
                        {record.complianceStatus}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDecisionTypeColor(record.decisionType)}`}>
                        {record.decisionType}
                      </span>
                      {record.isVerified && (
                        <span className="text-green-500 text-sm">✓ Verified</span>
                      )}
                    </div>
                    
                    <div className="flex items-center space-x-4 text-sm text-gray-600">
                      <span>Block #{record.blockNumber}</span>
                      <span>By: {record.submitter.slice(0, 6)}...{record.submitter.slice(-4)}</span>
                      <span>{formatDistanceToNow(new Date(record.timestamp * 1000), { addSuffix: true })}</span>
                    </div>
                  </div>

                  <div id={`record-expand-${index}`} className="flex items-center">
                    <button className="text-gray-400 hover:text-gray-600">
                      {expandedRecord === record.documentHash ? '▼' : '▶'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Expanded Content */}
              {expandedRecord === record.documentHash && (
                <div id={`record-details-${index}`} className="border-t border-gray-200 p-4 bg-gray-50">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Analysis Result */}
                    <div id={`analysis-result-${index}`}>
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Analysis Result</h4>
                      <div className="bg-white rounded-md p-3 border">
                        <pre className="text-xs text-gray-700 whitespace-pre-wrap">
                          {JSON.stringify(record.analysisResult, null, 2)}
                        </pre>
                      </div>
                    </div>

                    {/* Metadata */}
                    <div id={`metadata-${index}`}>
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Metadata</h4>
                      <div className="bg-white rounded-md p-3 border">
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Transaction Hash:</span>
                            <span className="font-mono text-xs">
                              {record.transactionHash.slice(0, 10)}...{record.transactionHash.slice(-8)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Timestamp:</span>
                            <span>{new Date(record.timestamp * 1000).toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Block Number:</span>
                            <span>{record.blockNumber}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Submitter:</span>
                            <span className="font-mono text-xs">{record.submitter}</span>
                          </div>
                        </div>
                        
                        {record.metadata && Object.keys(record.metadata).length > 0 && (
                          <div className="mt-3 pt-3 border-t">
                            <h5 className="text-xs font-medium text-gray-700 mb-2">Additional Metadata</h5>
                            <pre className="text-xs text-gray-600 whitespace-pre-wrap">
                              {JSON.stringify(record.metadata, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div id={`record-actions-${index}`} className="mt-4 pt-4 border-t border-gray-200">
                    <div className="flex items-center space-x-3">
                      <button
                        id={`view-on-explorer-${index}`}
                        onClick={() => window.open(`https://etherscan.io/tx/${record.transactionHash}`, '_blank')}
                        className="text-sm text-blue-600 hover:text-blue-800 underline"
                      >
                        View on Explorer
                      </button>
                      <button
                        id={`copy-hash-${index}`}
                        onClick={() => navigator.clipboard.writeText(record.documentHash)}
                        className="text-sm text-gray-600 hover:text-gray-800 underline"
                      >
                        Copy Document Hash
                      </button>
                      <button
                        id={`verify-record-${index}`}
                        className="text-sm text-green-600 hover:text-green-800 underline"
                      >
                        Verify Record
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Summary Stats */}
      <div id="audit-trail-stats" className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-900 mb-3">Summary Statistics</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div id="total-records-stat" className="text-center">
            <div className="text-2xl font-bold text-blue-600">{auditRecords.length}</div>
            <div className="text-xs text-gray-600">Total Records</div>
          </div>
          <div id="verified-records-stat" className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {auditRecords.filter(r => r.isVerified).length}
            </div>
            <div className="text-xs text-gray-600">Verified</div>
          </div>
          <div id="compliant-records-stat" className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {auditRecords.filter(r => r.complianceStatus === 'compliant').length}
            </div>
            <div className="text-xs text-gray-600">Compliant</div>
          </div>
          <div id="recent-records-stat" className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {auditRecords.filter(r => 
                Date.now() - (r.timestamp * 1000) < 24 * 60 * 60 * 1000
              ).length}
            </div>
            <div className="text-xs text-gray-600">Last 24h</div>
          </div>
        </div>
      </div>
    </div>
  );
};