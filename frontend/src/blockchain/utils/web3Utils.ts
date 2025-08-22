/**
 * Web3 utility functions for blockchain interactions
 */

// Network configurations
export const NETWORK_CONFIGS = {
  ethereum: {
    chainId: 1,
    name: 'Ethereum Mainnet',
    symbol: 'ETH',
    decimals: 18,
    rpcUrls: ['https://mainnet.infura.io/v3/YOUR_PROJECT_ID'],
    blockExplorerUrls: ['https://etherscan.io'],
    iconUrls: ['https://cryptologos.cc/logos/ethereum-eth-logo.svg']
  },
  goerli: {
    chainId: 5,
    name: 'Goerli Testnet',
    symbol: 'ETH',
    decimals: 18,
    rpcUrls: ['https://goerli.infura.io/v3/YOUR_PROJECT_ID'],
    blockExplorerUrls: ['https://goerli.etherscan.io'],
    iconUrls: ['https://cryptologos.cc/logos/ethereum-eth-logo.svg']
  },
  polygon: {
    chainId: 137,
    name: 'Polygon Mainnet',
    symbol: 'MATIC',
    decimals: 18,
    rpcUrls: ['https://polygon-rpc.com'],
    blockExplorerUrls: ['https://polygonscan.com'],
    iconUrls: ['https://cryptologos.cc/logos/polygon-matic-logo.svg']
  },
  bsc: {
    chainId: 56,
    name: 'Binance Smart Chain',
    symbol: 'BNB',
    decimals: 18,
    rpcUrls: ['https://bsc-dataseed1.binance.org'],
    blockExplorerUrls: ['https://bscscan.com'],
    iconUrls: ['https://cryptologos.cc/logos/binance-coin-bnb-logo.svg']
  }
};

// MetaMask connection utilities
export class MetaMaskUtils {
  static async isInstalled(): Promise<boolean> {
    return typeof window !== 'undefined' && typeof window.ethereum !== 'undefined';
  }

  static async isConnected(): Promise<boolean> {
    if (!await this.isInstalled()) return false;
    
    try {
      const accounts = await window.ethereum.request({ method: 'eth_accounts' });
      return accounts.length > 0;
    } catch {
      return false;
    }
  }

  static async connect(): Promise<string[]> {
    if (!await this.isInstalled()) {
      throw new Error('MetaMask is not installed');
    }

    return await window.ethereum.request({ method: 'eth_requestAccounts' });
  }

  static async getCurrentAccount(): Promise<string | null> {
    if (!await this.isInstalled()) return null;

    try {
      const accounts = await window.ethereum.request({ method: 'eth_accounts' });
      return accounts[0] || null;
    } catch {
      return null;
    }
  }

  static async getCurrentChainId(): Promise<number | null> {
    if (!await this.isInstalled()) return null;

    try {
      const chainId = await window.ethereum.request({ method: 'eth_chainId' });
      return parseInt(chainId, 16);
    } catch {
      return null;
    }
  }

  static async switchNetwork(chainId: number): Promise<void> {
    if (!await this.isInstalled()) {
      throw new Error('MetaMask is not installed');
    }

    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: `0x${chainId.toString(16)}` }],
      });
    } catch (switchError: any) {
      // This error code indicates that the chain has not been added to MetaMask
      if (switchError.code === 4902) {
        throw new Error('Network not added to MetaMask');
      }
      throw switchError;
    }
  }

  static async addNetwork(networkConfig: any): Promise<void> {
    if (!await this.isInstalled()) {
      throw new Error('MetaMask is not installed');
    }

    await window.ethereum.request({
      method: 'wallet_addEthereumChain',
      params: [{
        chainId: `0x${networkConfig.chainId.toString(16)}`,
        chainName: networkConfig.name,
        nativeCurrency: {
          name: networkConfig.symbol,
          symbol: networkConfig.symbol,
          decimals: networkConfig.decimals,
        },
        rpcUrls: networkConfig.rpcUrls,
        blockExplorerUrls: networkConfig.blockExplorerUrls,
        iconUrls: networkConfig.iconUrls,
      }],
    });
  }

  static async addToken(tokenConfig: {
    address: string;
    symbol: string;
    decimals: number;
    image?: string;
  }): Promise<void> {
    if (!await this.isInstalled()) {
      throw new Error('MetaMask is not installed');
    }

    await window.ethereum.request({
      method: 'wallet_watchAsset',
      params: {
        type: 'ERC20',
        options: {
          address: tokenConfig.address,
          symbol: tokenConfig.symbol,
          decimals: tokenConfig.decimals,
          image: tokenConfig.image,
        },
      },
    });
  }

  static formatAddress(address: string, length: number = 6): string {
    if (!address) return '';
    return `${address.slice(0, length)}...${address.slice(-4)}`;
  }

  static isValidAddress(address: string): boolean {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
  }

  static isValidTxHash(hash: string): boolean {
    return /^0x[a-fA-F0-9]{64}$/.test(hash);
  }
}

// Contract interaction utilities
export class ContractUtils {
  static async callMethod(
    web3: any,
    contractAddress: string,
    abi: any[],
    methodName: string,
    params: any[] = [],
    fromAddress?: string
  ): Promise<any> {
    const contract = new web3.eth.Contract(abi, contractAddress);
    
    if (fromAddress) {
      return await contract.methods[methodName](...params).call({ from: fromAddress });
    } else {
      return await contract.methods[methodName](...params).call();
    }
  }

  static async sendTransaction(
    web3: any,
    contractAddress: string,
    abi: any[],
    methodName: string,
    params: any[] = [],
    fromAddress: string,
    options: {
      gas?: number;
      gasPrice?: string;
      value?: string;
    } = {}
  ): Promise<any> {
    const contract = new web3.eth.Contract(abi, contractAddress);
    
    const gasEstimate = await contract.methods[methodName](...params)
      .estimateGas({ from: fromAddress, value: options.value || '0' });
    
    const gasPrice = options.gasPrice || await web3.eth.getGasPrice();
    
    return await contract.methods[methodName](...params).send({
      from: fromAddress,
      gas: Math.floor(gasEstimate * 1.2), // Add 20% buffer
      gasPrice,
      value: options.value || '0',
      ...options
    });
  }

  static async getPastEvents(
    web3: any,
    contractAddress: string,
    abi: any[],
    eventName: string,
    fromBlock: number | string = 'earliest',
    toBlock: number | string = 'latest',
    filter: any = {}
  ): Promise<any[]> {
    const contract = new web3.eth.Contract(abi, contractAddress);
    
    return await contract.getPastEvents(eventName, {
      fromBlock,
      toBlock,
      filter
    });
  }

  static subscribeToEvents(
    web3: any,
    contractAddress: string,
    abi: any[],
    eventName: string,
    callback: (event: any) => void,
    filter: any = {}
  ): any {
    const contract = new web3.eth.Contract(abi, contractAddress);
    
    return contract.events[eventName]({ filter })
      .on('data', callback)
      .on('error', console.error);
  }
}

// Transaction utilities
export class TransactionUtils {
  static async waitForConfirmation(
    web3: any,
    txHash: string,
    confirmations: number = 1,
    timeout: number = 300000 // 5 minutes
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      
      const checkConfirmation = async () => {
        try {
          const receipt = await web3.eth.getTransactionReceipt(txHash);
          
          if (!receipt) {
            // Transaction not mined yet
            if (Date.now() - startTime > timeout) {
              reject(new Error('Transaction timeout'));
              return;
            }
            setTimeout(checkConfirmation, 2000);
            return;
          }
          
          const currentBlock = await web3.eth.getBlockNumber();
          const confirmationCount = currentBlock - receipt.blockNumber + 1;
          
          if (confirmationCount >= confirmations) {
            resolve(receipt);
          } else {
            if (Date.now() - startTime > timeout) {
              reject(new Error('Transaction timeout'));
              return;
            }
            setTimeout(checkConfirmation, 2000);
          }
        } catch (error) {
          reject(error);
        }
      };
      
      checkConfirmation();
    });
  }

  static calculateGasCost(gasUsed: number, gasPrice: string, decimals: number = 18): string {
    const gasCostWei = BigInt(gasUsed) * BigInt(gasPrice);
    const gasCostEth = Number(gasCostWei) / Math.pow(10, decimals);
    return gasCostEth.toFixed(6);
  }

  static formatGasPrice(gasPriceWei: string): string {
    const gwei = Number(gasPriceWei) / 1e9;
    return gwei.toFixed(2);
  }
}

// Encoding/Decoding utilities
export class EncodingUtils {
  static encodeParameters(web3: any, types: string[], values: any[]): string {
    return web3.eth.abi.encodeParameters(types, values);
  }

  static decodeParameters(web3: any, types: string[], data: string): any {
    return web3.eth.abi.decodeParameters(types, data);
  }

  static encodeFunctionCall(web3: any, jsonInterface: any, parameters: any[]): string {
    return web3.eth.abi.encodeFunctionCall(jsonInterface, parameters);
  }

  static decodeFunctionCall(web3: any, jsonInterface: any, data: string): any {
    return web3.eth.abi.decodeFunctionCall(jsonInterface, data);
  }

  static keccak256(web3: any, value: string): string {
    return web3.utils.keccak256(value);
  }

  static sha3(web3: any, value: string): string {
    return web3.utils.sha3(value);
  }
}

// Unit conversion utilities
export class UnitUtils {
  static toWei(web3: any, value: string, unit: string = 'ether'): string {
    return web3.utils.toWei(value, unit);
  }

  static fromWei(web3: any, value: string, unit: string = 'ether'): string {
    return web3.utils.fromWei(value, unit);
  }

  static toBN(web3: any, value: string | number): any {
    return web3.utils.toBN(value);
  }

  static toHex(web3: any, value: string | number): string {
    return web3.utils.toHex(value);
  }

  static formatBalance(balance: string, decimals: number = 18, precision: number = 4): string {
    const balanceNumber = Number(balance) / Math.pow(10, decimals);
    return balanceNumber.toFixed(precision);
  }
}

// Error handling utilities
export class ErrorUtils {
  static parseError(error: any): { code: number; message: string; data?: any } {
    if (error.code && error.message) {
      return {
        code: error.code,
        message: error.message,
        data: error.data
      };
    }

    // MetaMask errors
    if (error.code === 4001) {
      return {
        code: 4001,
        message: 'User rejected the request'
      };
    }

    if (error.code === -32002) {
      return {
        code: -32002,
        message: 'Request already pending'
      };
    }

    if (error.code === -32603) {
      return {
        code: -32603,
        message: 'Internal error'
      };
    }

    // Generic error
    return {
      code: -1,
      message: error.message || 'Unknown error occurred'
    };
  }

  static isUserRejection(error: any): boolean {
    return error.code === 4001;
  }

  static isNetworkError(error: any): boolean {
    return error.code === -32603 || error.message?.includes('network');
  }

  static isInsufficientFunds(error: any): boolean {
    return error.message?.toLowerCase().includes('insufficient funds');
  }
}

// Local storage utilities for Web3
export class StorageUtils {
  static saveWalletConnection(address: string, chainId: number): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('walletConnection', JSON.stringify({
        address,
        chainId,
        timestamp: Date.now()
      }));
    }
  }

  static getWalletConnection(): { address: string; chainId: number; timestamp: number } | null {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('walletConnection');
      return stored ? JSON.parse(stored) : null;
    }
    return null;
  }

  static clearWalletConnection(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('walletConnection');
    }
  }

  static saveTransactionHistory(txHash: string, details: any): void {
    if (typeof window !== 'undefined') {
      const history = this.getTransactionHistory();
      history.unshift({ txHash, ...details, timestamp: Date.now() });
      
      // Keep only last 50 transactions
      const trimmed = history.slice(0, 50);
      localStorage.setItem('transactionHistory', JSON.stringify(trimmed));
    }
  }

  static getTransactionHistory(): any[] {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('transactionHistory');
      return stored ? JSON.parse(stored) : [];
    }
    return [];
  }

  static clearTransactionHistory(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('transactionHistory');
    }
  }
}

// Validation utilities
export class ValidationUtils {
  static isValidEthereumAddress(address: string): boolean {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
  }

  static isValidTransactionHash(hash: string): boolean {
    return /^0x[a-fA-F0-9]{64}$/.test(hash);
  }

  static isValidBlockNumber(blockNumber: string): boolean {
    return /^\d+$/.test(blockNumber) || blockNumber === 'latest' || blockNumber === 'earliest';
  }

  static isValidAmount(amount: string): boolean {
    return /^\d*\.?\d+$/.test(amount) && parseFloat(amount) > 0;
  }

  static isValidGasPrice(gasPrice: string): boolean {
    return /^\d*\.?\d+$/.test(gasPrice) && parseFloat(gasPrice) > 0;
  }

  static isValidChainId(chainId: number): boolean {
    return Number.isInteger(chainId) && chainId > 0;
  }
}

// Blockchain data formatters
export class FormatterUtils {
  static formatBlockNumber(blockNumber: number): string {
    return blockNumber.toLocaleString();
  }

  static formatTimestamp(timestamp: number): string {
    return new Date(timestamp * 1000).toLocaleString();
  }

  static formatGasUsage(gasUsed: number, gasLimit: number): {
    used: string;
    limit: string;
    percentage: number;
  } {
    return {
      used: gasUsed.toLocaleString(),
      limit: gasLimit.toLocaleString(),
      percentage: Math.round((gasUsed / gasLimit) * 100)
    };
  }

  static formatFileSize(bytes: number): string {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  static formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
  }
}

export default {
  MetaMaskUtils,
  ContractUtils,
  TransactionUtils,
  EncodingUtils,
  UnitUtils,
  ErrorUtils,
  StorageUtils,
  ValidationUtils,
  FormatterUtils,
  NETWORK_CONFIGS
};