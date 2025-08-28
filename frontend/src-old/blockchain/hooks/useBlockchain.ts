import { useState, useEffect, useCallback, useRef } from 'react';
import { useWeb3 } from './useWeb3';

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

interface EventHandlers {
  onNewBlock?: (block: Block) => void;
  onNewTransaction?: (transaction: Transaction) => void;
  onAuditRecord?: (record: AuditRecord) => void;
}

interface UseBlockchainReturn {
  latestBlocks: Block[];
  recentTransactions: Transaction[];
  auditRecords: AuditRecord[];
  networkStats: NetworkStats | null;
  isLoading: boolean;
  error: string | null;
  searchBlockchain: (query: string) => Promise<any>;
  subscribeToEvents: (handlers: EventHandlers) => () => void;
  getBlock: (blockNumber: number | 'latest') => Promise<Block | null>;
  getTransaction: (txHash: string) => Promise<Transaction | null>;
  getAuditRecord: (documentHash: string) => Promise<AuditRecord | null>;
  verifyDocument: (documentHash: string) => Promise<boolean>;
}

export const useBlockchain = (
  networkId: string,
  contractAddress?: string
): UseBlockchainReturn => {
  const { web3, isConnected } = useWeb3();
  const [latestBlocks, setLatestBlocks] = useState<Block[]>([]);
  const [recentTransactions, setRecentTransactions] = useState<Transaction[]>([]);
  const [auditRecords, setAuditRecords] = useState<AuditRecord[]>([]);
  const [networkStats, setNetworkStats] = useState<NetworkStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const eventHandlersRef = useRef<EventHandlers>({});
  const contractRef = useRef<any>(null);

  // Contract ABI (simplified)
  const auditContractABI = [
    {
      "inputs": [{"type": "bytes32", "name": "documentHash"}],
      "name": "getAuditRecord",
      "outputs": [
        {"type": "string", "name": "analysisResult"},
        {"type": "string", "name": "decisionType"},
        {"type": "string", "name": "complianceStatus"},
        {"type": "string", "name": "metadata"},
        {"type": "uint256", "name": "timestamp"},
        {"type": "address", "name": "submitter"}
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [{"type": "bytes32", "name": "documentHash"}],
      "name": "verifyDocument",
      "outputs": [{"type": "bool", "name": "exists"}],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "anonymous": false,
      "inputs": [
        {"indexed": true, "type": "bytes32", "name": "documentHash"},
        {"indexed": true, "type": "address", "name": "submitter"},
        {"indexed": false, "type": "string", "name": "analysisResult"},
        {"indexed": false, "type": "uint256", "name": "timestamp"}
      ],
      "name": "AuditRecordStored",
      "type": "event"
    }
  ];

  // Initialize contract
  useEffect(() => {
    if (web3 && contractAddress) {
      try {
        contractRef.current = new web3.eth.Contract(auditContractABI, contractAddress);
      } catch (err) {
        console.error('Error initializing contract:', err);
        setError('Failed to initialize smart contract');
      }
    }
  }, [web3, contractAddress]);

  // Fetch latest blocks
  const fetchLatestBlocks = useCallback(async () => {
    if (!web3) return;

    try {
      const latestBlockNumber = await web3.eth.getBlockNumber();
      const blocks: Block[] = [];

      for (let i = 0; i < 10; i++) {
        const blockNumber = latestBlockNumber - i;
        if (blockNumber < 0) break;

        const block = await web3.eth.getBlock(blockNumber, true);
        if (block) {
          blocks.push({
            number: block.number,
            hash: block.hash,
            timestamp: Number(block.timestamp),
            transactions: block.transactions.slice(0, 5).map((tx: any) => ({
              hash: tx.hash,
              from: tx.from,
              to: tx.to || '',
              value: web3.utils.fromWei(tx.value, 'ether'),
              gas: tx.gas,
              gasPrice: tx.gasPrice,
              blockNumber: tx.blockNumber,
              status: 'confirmed' as const,
              timestamp: Number(block.timestamp)
            })),
            gasUsed: block.gasUsed,
            gasLimit: block.gasLimit,
            miner: block.miner,
            size: block.size
          });
        }
      }

      setLatestBlocks(blocks);
    } catch (err) {
      console.error('Error fetching blocks:', err);
      setError('Failed to fetch blockchain data');
    }
  }, [web3]);

  // Fetch recent transactions
  const fetchRecentTransactions = useCallback(async () => {
    if (!web3) return;

    try {
      const latestBlock = await web3.eth.getBlock('latest', true);
      if (latestBlock && latestBlock.transactions) {
        const transactions = latestBlock.transactions.slice(0, 20).map((tx: any) => ({
          hash: tx.hash,
          from: tx.from,
          to: tx.to || '',
          value: web3.utils.fromWei(tx.value, 'ether'),
          gas: tx.gas,
          gasPrice: tx.gasPrice,
          blockNumber: tx.blockNumber,
          status: 'confirmed' as const,
          timestamp: Number(latestBlock.timestamp)
        }));

        setRecentTransactions(transactions);
      }
    } catch (err) {
      console.error('Error fetching transactions:', err);
    }
  }, [web3]);

  // Fetch audit records
  const fetchAuditRecords = useCallback(async () => {
    if (!contractRef.current) return;

    try {
      // Get past events
      const events = await contractRef.current.getPastEvents('AuditRecordStored', {
        fromBlock: 'earliest',
        toBlock: 'latest'
      });

      const records: AuditRecord[] = await Promise.all(
        events.slice(-50).map(async (event: any) => {
          const { documentHash, submitter, analysisResult, timestamp } = event.returnValues;
          
          try {
            // Get full record details
            const recordDetails = await contractRef.current.methods
              .getAuditRecord(documentHash)
              .call();

            return {
              documentHash,
              analysisResult: JSON.parse(analysisResult),
              timestamp: Number(timestamp),
              submitter,
              decisionType: recordDetails.decisionType,
              complianceStatus: recordDetails.complianceStatus,
              metadata: JSON.parse(recordDetails.metadata || '{}'),
              blockNumber: event.blockNumber,
              transactionHash: event.transactionHash,
              isVerified: true
            };
          } catch (err) {
            console.error('Error parsing audit record:', err);
            return {
              documentHash,
              analysisResult: { error: 'Failed to parse' },
              timestamp: Number(timestamp),
              submitter,
              decisionType: 'unknown',
              complianceStatus: 'error',
              metadata: {},
              blockNumber: event.blockNumber,
              transactionHash: event.transactionHash,
              isVerified: false
            };
          }
        })
      );

      setAuditRecords(records.reverse()); // Show newest first
    } catch (err) {
      console.error('Error fetching audit records:', err);
    }
  }, []);

  // Fetch network statistics
  const fetchNetworkStats = useCallback(async () => {
    if (!web3) return;

    try {
      const [
        blockNumber,
        gasPrice,
        netPeerCount
      ] = await Promise.all([
        web3.eth.getBlockNumber(),
        web3.eth.getGasPrice(),
        web3.eth.net.getPeerCount().catch(() => 0)
      ]);

      // Get recent blocks to calculate average block time
      const recentBlocks = await Promise.all([
        web3.eth.getBlock(blockNumber),
        web3.eth.getBlock(blockNumber - 1),
        web3.eth.getBlock(blockNumber - 2)
      ]);

      const blockTimes = [];
      for (let i = 0; i < recentBlocks.length - 1; i++) {
        const timeDiff = Number(recentBlocks[i].timestamp) - Number(recentBlocks[i + 1].timestamp);
        blockTimes.push(timeDiff);
      }

      const avgBlockTime = blockTimes.reduce((a, b) => a + b, 0) / blockTimes.length;

      setNetworkStats({
        blockHeight: blockNumber,
        blockTime: avgBlockTime,
        gasPrice: Number(web3.utils.fromWei(gasPrice, 'gwei')),
        peerCount: netPeerCount,
        hashRate: 0, // Would need external API
        difficulty: '0', // Would need external API
        totalSupply: '0', // Would need external API
        marketCap: '0' // Would need external API
      });
    } catch (err) {
      console.error('Error fetching network stats:', err);
    }
  }, [web3]);

  // Search blockchain
  const searchBlockchain = useCallback(async (query: string) => {
    if (!web3) throw new Error('Web3 not initialized');

    setIsLoading(true);
    try {
      // Determine query type
      if (query.length === 66 && query.startsWith('0x')) {
        // Transaction hash
        const tx = await web3.eth.getTransaction(query);
        if (tx) {
          return {
            type: 'transaction',
            data: {
              hash: tx.hash,
              from: tx.from,
              to: tx.to,
              value: web3.utils.fromWei(tx.value, 'ether'),
              gas: tx.gas,
              gasPrice: tx.gasPrice,
              blockNumber: tx.blockNumber,
              status: 'confirmed'
            }
          };
        }
      } else if (query.length === 42 && query.startsWith('0x')) {
        // Address
        const balance = await web3.eth.getBalance(query);
        const transactionCount = await web3.eth.getTransactionCount(query);
        
        return {
          type: 'address',
          data: {
            address: query,
            balance: web3.utils.fromWei(balance, 'ether'),
            transactionCount
          }
        };
      } else if (/^\d+$/.test(query)) {
        // Block number
        const block = await web3.eth.getBlock(parseInt(query), true);
        if (block) {
          return {
            type: 'block',
            data: {
              number: block.number,
              hash: block.hash,
              timestamp: Number(block.timestamp),
              transactions: block.transactions.map((tx: any) => ({
                hash: tx.hash,
                from: tx.from,
                to: tx.to,
                value: web3.utils.fromWei(tx.value, 'ether')
              })),
              gasUsed: block.gasUsed,
              gasLimit: block.gasLimit,
              miner: block.miner
            }
          };
        }
      }

      throw new Error('No results found');
    } finally {
      setIsLoading(false);
    }
  }, [web3]);

  // Get specific block
  const getBlock = useCallback(async (blockNumber: number | 'latest'): Promise<Block | null> => {
    if (!web3) return null;

    try {
      const block = await web3.eth.getBlock(blockNumber, true);
      if (!block) return null;

      return {
        number: block.number,
        hash: block.hash,
        timestamp: Number(block.timestamp),
        transactions: block.transactions.map((tx: any) => ({
          hash: tx.hash,
          from: tx.from,
          to: tx.to || '',
          value: web3.utils.fromWei(tx.value, 'ether'),
          gas: tx.gas,
          gasPrice: tx.gasPrice,
          blockNumber: tx.blockNumber,
          status: 'confirmed' as const,
          timestamp: Number(block.timestamp)
        })),
        gasUsed: block.gasUsed,
        gasLimit: block.gasLimit,
        miner: block.miner,
        size: block.size
      };
    } catch (err) {
      console.error('Error getting block:', err);
      return null;
    }
  }, [web3]);

  // Get specific transaction
  const getTransaction = useCallback(async (txHash: string): Promise<Transaction | null> => {
    if (!web3) return null;

    try {
      const tx = await web3.eth.getTransaction(txHash);
      if (!tx) return null;

      const receipt = await web3.eth.getTransactionReceipt(txHash);
      const block = await web3.eth.getBlock(tx.blockNumber);

      return {
        hash: tx.hash,
        from: tx.from,
        to: tx.to || '',
        value: web3.utils.fromWei(tx.value, 'ether'),
        gas: tx.gas,
        gasPrice: tx.gasPrice,
        blockNumber: tx.blockNumber,
        status: receipt?.status ? 'confirmed' : 'failed',
        timestamp: Number(block.timestamp)
      };
    } catch (err) {
      console.error('Error getting transaction:', err);
      return null;
    }
  }, [web3]);

  // Get audit record
  const getAuditRecord = useCallback(async (documentHash: string): Promise<AuditRecord | null> => {
    if (!contractRef.current) return null;

    try {
      const recordDetails = await contractRef.current.methods
        .getAuditRecord(documentHash)
        .call();

      return {
        documentHash,
        analysisResult: JSON.parse(recordDetails.analysisResult),
        timestamp: Number(recordDetails.timestamp),
        submitter: recordDetails.submitter,
        decisionType: recordDetails.decisionType,
        complianceStatus: recordDetails.complianceStatus,
        metadata: JSON.parse(recordDetails.metadata || '{}'),
        blockNumber: 0, // Would need to get from event
        transactionHash: '', // Would need to get from event
        isVerified: true
      };
    } catch (err) {
      console.error('Error getting audit record:', err);
      return null;
    }
  }, []);

  // Verify document
  const verifyDocument = useCallback(async (documentHash: string): Promise<boolean> => {
    if (!contractRef.current) return false;

    try {
      return await contractRef.current.methods
        .verifyDocument(documentHash)
        .call();
    } catch (err) {
      console.error('Error verifying document:', err);
      return false;
    }
  }, []);

  // Subscribe to real-time events
  const subscribeToEvents = useCallback((handlers: EventHandlers) => {
    eventHandlersRef.current = handlers;

    // WebSocket connection for real-time updates
    const wsUrl = getWebSocketUrl(networkId);
    if (wsUrl) {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        // Subscribe to new blocks
        wsRef.current?.send(JSON.stringify({
          jsonrpc: '2.0',
          method: 'eth_subscribe',
          params: ['newHeads'],
          id: 1
        }));
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.method === 'eth_subscription') {
            const blockHeader = data.params.result;
            if (handlers.onNewBlock) {
              // Convert to Block format
              const block: Block = {
                number: parseInt(blockHeader.number, 16),
                hash: blockHeader.hash,
                timestamp: parseInt(blockHeader.timestamp, 16),
                transactions: [],
                gasUsed: parseInt(blockHeader.gasUsed, 16),
                gasLimit: parseInt(blockHeader.gasLimit, 16),
                miner: blockHeader.miner,
                size: parseInt(blockHeader.size, 16)
              };
              handlers.onNewBlock(block);
            }
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    }

    // Contract event subscription
    if (contractRef.current && handlers.onAuditRecord) {
      const subscription = contractRef.current.events.AuditRecordStored()
        .on('data', (event: any) => {
          const { documentHash, submitter, analysisResult, timestamp } = event.returnValues;
          const record: AuditRecord = {
            documentHash,
            analysisResult: JSON.parse(analysisResult),
            timestamp: Number(timestamp),
            submitter,
            decisionType: '',
            complianceStatus: '',
            metadata: {},
            blockNumber: event.blockNumber,
            transactionHash: event.transactionHash,
            isVerified: true
          };
          handlers.onAuditRecord?.(record);
        });

      return () => {
        if (wsRef.current) {
          wsRef.current.close();
        }
        subscription.unsubscribe();
      };
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [networkId]);

  // Helper function to get WebSocket URL
  const getWebSocketUrl = (networkId: string): string | null => {
    const urls: Record<string, string> = {
      'ethereum': 'wss://mainnet.infura.io/ws/v3/YOUR_PROJECT_ID',
      'polygon': 'wss://polygon-mainnet.infura.io/ws/v3/YOUR_PROJECT_ID',
      'bsc': 'wss://bsc-dataseed1.binance.org:443/ws'
    };
    return urls[networkId] || null;
  };

  // Fetch data on mount and when web3 changes
  useEffect(() => {
    if (web3 && isConnected) {
      fetchLatestBlocks();
      fetchRecentTransactions();
      fetchNetworkStats();
      
      if (contractAddress) {
        fetchAuditRecords();
      }

      // Set up periodic updates
      const interval = setInterval(() => {
        fetchLatestBlocks();
        fetchRecentTransactions();
        fetchNetworkStats();
        if (contractAddress) {
          fetchAuditRecords();
        }
      }, 30000); // Update every 30 seconds

      return () => clearInterval(interval);
    }
  }, [web3, isConnected, contractAddress, fetchLatestBlocks, fetchRecentTransactions, fetchNetworkStats, fetchAuditRecords]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    latestBlocks,
    recentTransactions,
    auditRecords,
    networkStats,
    isLoading,
    error,
    searchBlockchain,
    subscribeToEvents,
    getBlock,
    getTransaction,
    getAuditRecord,
    verifyDocument
  };
};