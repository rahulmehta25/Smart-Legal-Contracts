/**
 * Speech Recognition Module
 * Advanced Web Speech API integration with comprehensive error handling,
 * browser compatibility, and real-time transcription capabilities
 */

import {
  ISpeechRecognitionConfig,
  ISpeechRecognitionCallbacks,
  ISpeechRecognitionResult,
  IVoiceError,
  VoiceErrorCode
} from '@/types/voice';

// Extend the Window interface to include webkitSpeechRecognition
declare global {
  interface Window {
    SpeechRecognition: typeof SpeechRecognition;
    webkitSpeechRecognition: typeof SpeechRecognition;
  }
}

export class SpeechRecognitionService {
  private recognition: SpeechRecognition | null = null;
  private isSupported: boolean = false;
  private isListening: boolean = false;
  private config: ISpeechRecognitionConfig;
  private callbacks: Partial<ISpeechRecognitionCallbacks>;
  private retryCount: number = 0;
  private maxRetries: number = 3;
  private restartTimeout: NodeJS.Timeout | null = null;
  private silenceTimer: NodeJS.Timeout | null = null;
  private lastResultTimestamp: number = 0;

  // Default configuration
  private defaultConfig: ISpeechRecognitionConfig = {
    continuous: true,
    interimResults: true,
    maxAlternatives: 3,
    language: 'en-US'
  };

  constructor(
    config?: Partial<ISpeechRecognitionConfig>,
    callbacks?: Partial<ISpeechRecognitionCallbacks>
  ) {
    this.config = { ...this.defaultConfig, ...config };
    this.callbacks = callbacks || {};
    this.initialize();
  }

  /**
   * Initialize speech recognition with browser compatibility checks
   */
  private initialize(): void {
    try {
      // Check for browser support
      const SpeechRecognitionConstructor = 
        window.SpeechRecognition || window.webkitSpeechRecognition;

      if (!SpeechRecognitionConstructor) {
        this.handleError('SPEECH_RECOGNITION_NOT_SUPPORTED', 
          'Speech recognition is not supported in this browser');
        return;
      }

      this.recognition = new SpeechRecognitionConstructor();
      this.isSupported = true;
      this.setupRecognition();

    } catch (error) {
      this.handleError('SPEECH_RECOGNITION_NOT_SUPPORTED', 
        `Failed to initialize speech recognition: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Configure speech recognition with advanced settings
   */
  private setupRecognition(): void {
    if (!this.recognition) return;

    // Apply configuration
    this.recognition.continuous = this.config.continuous;
    this.recognition.interimResults = this.config.interimResults;
    this.recognition.maxAlternatives = this.config.maxAlternatives;
    this.recognition.lang = this.config.language;

    // Apply grammar list if provided
    if (this.config.grammars) {
      this.recognition.grammars = this.config.grammars;
    }

    // Setup event handlers
    this.setupEventHandlers();
  }

  /**
   * Setup comprehensive event handlers for speech recognition
   */
  private setupEventHandlers(): void {
    if (!this.recognition) return;

    this.recognition.onstart = () => {
      this.isListening = true;
      this.retryCount = 0;
      this.lastResultTimestamp = Date.now();
      this.callbacks.onStart?.();
    };

    this.recognition.onend = () => {
      this.isListening = false;
      this.clearTimers();
      this.callbacks.onEnd?.();
      
      // Auto-restart if continuous mode is enabled and not manually stopped
      if (this.config.continuous && this.retryCount < this.maxRetries) {
        this.scheduleRestart();
      }
    };

    this.recognition.onerror = (event) => {
      this.handleSpeechError(event as SpeechRecognitionErrorEvent);
    };

    this.recognition.onresult = (event) => {
      this.handleResults(event as SpeechRecognitionEvent);
    };

    this.recognition.onspeechstart = () => {
      this.resetSilenceTimer();
      this.callbacks.onSpeechStart?.();
    };

    this.recognition.onspeechend = () => {
      this.callbacks.onSpeechEnd?.();
    };

    this.recognition.onsoundstart = () => {
      this.callbacks.onSoundStart?.();
    };

    this.recognition.onsoundend = () => {
      this.callbacks.onSoundEnd?.();
    };

    this.recognition.onnomatch = () => {
      this.callbacks.onNoMatch?.();
    };
  }

  /**
   * Handle speech recognition results with confidence scoring
   */
  private handleResults(event: SpeechRecognitionEvent): void {
    const results: ISpeechRecognitionResult[] = [];
    
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      const alternatives: Array<{transcript: string; confidence: number}> = [];

      // Extract alternatives with confidence scores
      for (let j = 0; j < result.length; j++) {
        alternatives.push({
          transcript: result[j].transcript,
          confidence: result[j].confidence || 0.5
        });
      }

      results.push({
        transcript: result[0].transcript,
        confidence: result[0].confidence || 0.5,
        isFinal: result.isFinal,
        alternatives
      });
    }

    this.lastResultTimestamp = Date.now();
    this.callbacks.onResult?.(results);
  }

  /**
   * Handle speech recognition errors with retry logic
   */
  private handleSpeechError(event: SpeechRecognitionErrorEvent): void {
    let errorCode: VoiceErrorCode;
    let recoverable = true;

    switch (event.error) {
      case 'no-speech':
        errorCode = 'TIMEOUT_ERROR';
        break;
      case 'audio-capture':
        errorCode = 'MICROPHONE_ACCESS_DENIED';
        recoverable = false;
        break;
      case 'not-allowed':
        errorCode = 'MICROPHONE_ACCESS_DENIED';
        recoverable = false;
        break;
      case 'network':
        errorCode = 'NETWORK_ERROR';
        break;
      case 'service-not-allowed':
        errorCode = 'SPEECH_RECOGNITION_NOT_SUPPORTED';
        recoverable = false;
        break;
      case 'aborted':
        errorCode = 'PROCESSING_ERROR';
        recoverable = false;
        break;
      default:
        errorCode = 'PROCESSING_ERROR';
    }

    this.handleError(errorCode, event.error, recoverable);
    this.callbacks.onError?.(event);

    // Implement retry logic for recoverable errors
    if (recoverable && this.retryCount < this.maxRetries) {
      this.retryCount++;
      this.scheduleRestart();
    }
  }

  /**
   * Schedule automatic restart with exponential backoff
   */
  private scheduleRestart(): void {
    if (this.restartTimeout) return;

    const delay = Math.min(1000 * Math.pow(2, this.retryCount), 10000);
    this.restartTimeout = setTimeout(() => {
      this.restartTimeout = null;
      if (this.config.continuous && !this.isListening) {
        this.start();
      }
    }, delay);
  }

  /**
   * Reset silence timer to detect speech inactivity
   */
  private resetSilenceTimer(): void {
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
    }

    this.silenceTimer = setTimeout(() => {
      if (Date.now() - this.lastResultTimestamp > 5000) {
        this.stop();
        this.callbacks.onEnd?.();
      }
    }, 5000);
  }

  /**
   * Clear all timers
   */
  private clearTimers(): void {
    if (this.restartTimeout) {
      clearTimeout(this.restartTimeout);
      this.restartTimeout = null;
    }
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
  }

  /**
   * Handle errors with comprehensive error information
   */
  private handleError(code: VoiceErrorCode, message: string, recoverable: boolean = true): void {
    const error: IVoiceError = {
      code,
      message,
      timestamp: new Date(),
      recoverable,
      details: {
        isSupported: this.isSupported,
        isListening: this.isListening,
        retryCount: this.retryCount,
        config: this.config
      }
    };

    console.error('Speech Recognition Error:', error);
  }

  /**
   * Start speech recognition
   */
  public start(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.isSupported) {
        reject(new Error('Speech recognition is not supported'));
        return;
      }

      if (this.isListening) {
        resolve();
        return;
      }

      if (!this.recognition) {
        reject(new Error('Speech recognition not initialized'));
        return;
      }

      try {
        this.recognition.start();
        resolve();
      } catch (error) {
        this.handleError('PROCESSING_ERROR', 
          `Failed to start recognition: ${error instanceof Error ? error.message : 'Unknown error'}`);
        reject(error);
      }
    });
  }

  /**
   * Stop speech recognition
   */
  public stop(): void {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.clearTimers();
    }
  }

  /**
   * Abort speech recognition immediately
   */
  public abort(): void {
    if (this.recognition && this.isListening) {
      this.recognition.abort();
      this.clearTimers();
    }
  }

  /**
   * Update configuration and restart if necessary
   */
  public updateConfig(newConfig: Partial<ISpeechRecognitionConfig>): void {
    const wasListening = this.isListening;
    
    if (wasListening) {
      this.stop();
    }

    this.config = { ...this.config, ...newConfig };
    
    if (this.recognition) {
      this.setupRecognition();
    }

    if (wasListening) {
      this.start();
    }
  }

  /**
   * Update callbacks
   */
  public updateCallbacks(newCallbacks: Partial<ISpeechRecognitionCallbacks>): void {
    this.callbacks = { ...this.callbacks, ...newCallbacks };
  }

  /**
   * Get current status
   */
  public getStatus(): {
    isSupported: boolean;
    isListening: boolean;
    config: ISpeechRecognitionConfig;
    retryCount: number;
  } {
    return {
      isSupported: this.isSupported,
      isListening: this.isListening,
      config: this.config,
      retryCount: this.retryCount
    };
  }

  /**
   * Get supported languages
   */
  public static getSupportedLanguages(): string[] {
    // Common languages supported by most browsers
    return [
      'en-US', 'en-GB', 'en-AU', 'en-CA', 'en-IN', 'en-NZ', 'en-ZA',
      'es-ES', 'es-MX', 'es-AR', 'es-CO', 'es-CL', 'es-PE', 'es-VE',
      'fr-FR', 'fr-CA', 'fr-CH', 'fr-BE',
      'de-DE', 'de-AT', 'de-CH',
      'it-IT', 'it-CH',
      'pt-PT', 'pt-BR',
      'ru-RU',
      'ja-JP',
      'ko-KR',
      'zh-CN', 'zh-TW', 'zh-HK',
      'ar-SA', 'ar-EG', 'ar-JO', 'ar-KW', 'ar-LB', 'ar-QA', 'ar-AE',
      'hi-IN',
      'th-TH',
      'tr-TR',
      'pl-PL',
      'cs-CZ',
      'sk-SK',
      'hu-HU',
      'ro-RO',
      'bg-BG',
      'hr-HR',
      'sl-SI',
      'et-EE',
      'lv-LV',
      'lt-LT',
      'fi-FI',
      'sv-SE',
      'no-NO',
      'da-DK',
      'is-IS',
      'nl-NL', 'nl-BE',
      'he-IL',
      'id-ID',
      'ms-MY',
      'vi-VN',
      'uk-UA',
      'el-GR',
      'mt-MT',
      'cy-GB',
      'ga-IE',
      'eu-ES',
      'ca-ES',
      'gl-ES'
    ];
  }

  /**
   * Test browser compatibility
   */
  public static testCompatibility(): {
    supported: boolean;
    features: {
      continuous: boolean;
      interimResults: boolean;
      maxAlternatives: boolean;
      grammars: boolean;
    };
    userAgent: string;
  } {
    const SpeechRecognitionConstructor = 
      window.SpeechRecognition || window.webkitSpeechRecognition;
    
    const supported = !!SpeechRecognitionConstructor;
    
    let features = {
      continuous: false,
      interimResults: false,
      maxAlternatives: false,
      grammars: false
    };

    if (supported) {
      try {
        const testRecognition = new SpeechRecognitionConstructor();
        features = {
          continuous: 'continuous' in testRecognition,
          interimResults: 'interimResults' in testRecognition,
          maxAlternatives: 'maxAlternatives' in testRecognition,
          grammars: 'grammars' in testRecognition
        };
      } catch (error) {
        console.warn('Error testing speech recognition features:', error);
      }
    }

    return {
      supported,
      features,
      userAgent: navigator.userAgent
    };
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.stop();
    this.clearTimers();
    this.recognition = null;
    this.callbacks = {};
  }
}