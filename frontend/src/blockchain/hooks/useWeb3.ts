import { useState, useEffect, useCallback } from 'react';

declare global {
  interface Window {
    ethereum?: any;
  }
}

interface Web3State {
  web3: any | null;
  account: string | null;
  chainId: number | null;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
}

interface UseWeb3Return extends Web3State {
  connectWallet: () => Promise<void>;
  disconnectWallet: () => void;
  switchNetwork: (chainId: number) => Promise<void>;
  addNetwork: (networkConfig: any) => Promise<void>;
}

export const useWeb3 = (): UseWeb3Return => {
  const [state, setState] = useState<Web3State>({
    web3: null,
    account: null,
    chainId: null,
    isConnected: false,
    isLoading: false,
    error: null
  });

  // Check if MetaMask is installed
  const isMetaMaskInstalled = useCallback(() => {
    return typeof window !== 'undefined' && typeof window.ethereum !== 'undefined';
  }, []);

  // Initialize Web3 connection
  const initializeWeb3 = useCallback(async () => {
    if (!isMetaMaskInstalled()) {
      setState(prev => ({ 
        ...prev, 
        error: 'MetaMask is not installed. Please install MetaMask to continue.' 
      }));
      return;
    }

    try {
      // Import Web3 dynamically
      const Web3 = (await import('web3')).default;
      const web3Instance = new Web3(window.ethereum);
      
      setState(prev => ({ ...prev, web3: web3Instance }));

      // Check if already connected
      const accounts = await window.ethereum.request({ method: 'eth_accounts' });
      if (accounts.length > 0) {
        const chainId = await window.ethereum.request({ method: 'eth_chainId' });
        setState(prev => ({
          ...prev,
          account: accounts[0],
          chainId: parseInt(chainId, 16),
          isConnected: true
        }));
      }
    } catch (error) {
      console.error('Error initializing Web3:', error);
      setState(prev => ({ 
        ...prev, 
        error: 'Failed to initialize Web3 connection' 
      }));
    }
  }, [isMetaMaskInstalled]);

  // Connect wallet
  const connectWallet = useCallback(async () => {
    if (!isMetaMaskInstalled()) {
      setState(prev => ({ 
        ...prev, 
        error: 'MetaMask is not installed. Please install MetaMask to continue.' 
      }));
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // Request account access
      const accounts = await window.ethereum.request({ 
        method: 'eth_requestAccounts' 
      });
      
      const chainId = await window.ethereum.request({ 
        method: 'eth_chainId' 
      });

      if (accounts.length > 0) {
        setState(prev => ({
          ...prev,
          account: accounts[0],
          chainId: parseInt(chainId, 16),
          isConnected: true,
          isLoading: false
        }));
      }
    } catch (error: any) {
      console.error('Error connecting wallet:', error);
      let errorMessage = 'Failed to connect wallet';
      
      if (error.code === 4001) {
        errorMessage = 'Connection rejected by user';
      } else if (error.code === -32002) {
        errorMessage = 'Connection request already pending';
      }

      setState(prev => ({ 
        ...prev, 
        error: errorMessage,
        isLoading: false 
      }));
    }
  }, [isMetaMaskInstalled]);

  // Disconnect wallet
  const disconnectWallet = useCallback(() => {
    setState(prev => ({
      ...prev,
      account: null,
      chainId: null,
      isConnected: false,
      error: null
    }));
  }, []);

  // Switch network
  const switchNetwork = useCallback(async (targetChainId: number) => {
    if (!window.ethereum) {
      throw new Error('MetaMask is not installed');
    }

    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: `0x${targetChainId.toString(16)}` }],
      });
    } catch (error: any) {
      // This error code indicates that the chain has not been added to MetaMask
      if (error.code === 4902) {
        throw new Error('Network not added to MetaMask');
      }
      throw error;
    }
  }, []);

  // Add network to MetaMask
  const addNetwork = useCallback(async (networkConfig: any) => {
    if (!window.ethereum) {
      throw new Error('MetaMask is not installed');
    }

    try {
      await window.ethereum.request({
        method: 'wallet_addEthereumChain',
        params: [networkConfig],
      });
    } catch (error) {
      console.error('Error adding network:', error);
      throw error;
    }
  }, []);

  // Handle account changes
  const handleAccountsChanged = useCallback((accounts: string[]) => {
    if (accounts.length === 0) {
      // User disconnected
      setState(prev => ({
        ...prev,
        account: null,
        isConnected: false
      }));
    } else {
      // User switched accounts
      setState(prev => ({
        ...prev,
        account: accounts[0],
        isConnected: true
      }));
    }
  }, []);

  // Handle chain changes
  const handleChainChanged = useCallback((chainId: string) => {
    setState(prev => ({
      ...prev,
      chainId: parseInt(chainId, 16)
    }));
    // Reload the page to avoid any issues with state
    window.location.reload();
  }, []);

  // Handle disconnect
  const handleDisconnect = useCallback(() => {
    setState(prev => ({
      ...prev,
      account: null,
      chainId: null,
      isConnected: false
    }));
  }, []);

  // Set up event listeners
  useEffect(() => {
    if (window.ethereum) {
      window.ethereum.on('accountsChanged', handleAccountsChanged);
      window.ethereum.on('chainChanged', handleChainChanged);
      window.ethereum.on('disconnect', handleDisconnect);

      // Cleanup
      return () => {
        if (window.ethereum.removeListener) {
          window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
          window.ethereum.removeListener('chainChanged', handleChainChanged);
          window.ethereum.removeListener('disconnect', handleDisconnect);
        }
      };
    }
  }, [handleAccountsChanged, handleChainChanged, handleDisconnect]);

  // Initialize on mount
  useEffect(() => {
    initializeWeb3();
  }, [initializeWeb3]);

  return {
    ...state,
    connectWallet,
    disconnectWallet,
    switchNetwork,
    addNetwork
  };
};