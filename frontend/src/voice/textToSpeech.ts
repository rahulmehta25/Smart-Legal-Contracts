/**
 * Text-to-Speech Module
 * Advanced speech synthesis with natural voice output, SSML support,
 * queue management, and comprehensive voice customization
 */

import {
  ITTSVoice,
  ITTSOptions,
  ITTSCallbacks,
  ITTSRequest,
  IVoiceError,
  VoiceErrorCode
} from '@/types/voice';

export class TextToSpeechService {
  private synthesis: SpeechSynthesis;
  private voices: ITTSVoice[] = [];
  private currentUtterance: SpeechSynthesisUtterance | null = null;
  private speechQueue: ITTSRequest[] = [];
  private isPlaying: boolean = false;
  private isPaused: boolean = false;
  private currentRequest: ITTSRequest | null = null;
  private defaultOptions: ITTSOptions;
  private lastSpokenText: string = '';
  private volumeBeforeMute: number = 1;
  private isMuted: boolean = false;

  // Events
  private eventListeners: Map<string, Set<(event: any) => void>> = new Map();

  constructor(options?: Partial<ITTSOptions>) {
    this.synthesis = window.speechSynthesis;
    
    this.defaultOptions = {
      rate: 1.0,
      pitch: 1.0,
      volume: 1.0,
      lang: 'en-US',
      ...options
    };

    this.initialize();
  }

  /**
   * Initialize the TTS service
   */
  private async initialize(): Promise<void> {
    try {
      await this.loadVoices();
      this.setupEventHandlers();
      
      // Select default voice if none specified
      if (!this.defaultOptions.voice && this.voices.length > 0) {
        this.defaultOptions.voice = this.getPreferredVoice();
      }
    } catch (error) {
      this.handleError('SPEECH_SYNTHESIS_NOT_SUPPORTED', 
        `Failed to initialize TTS: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Load available voices with enhanced metadata
   */
  private async loadVoices(): Promise<void> {
    return new Promise((resolve) => {
      const loadVoicesWithTimeout = () => {
        const voices = this.synthesis.getVoices();
        
        if (voices.length === 0) {
          // Some browsers need time to load voices
          setTimeout(loadVoicesWithTimeout, 100);
          return;
        }

        this.voices = voices.map(voice => ({
          name: voice.name,
          lang: voice.lang,
          gender: this.detectGender(voice.name),
          localService: voice.localService,
          voiceURI: voice.voiceURI,
          default: voice.default
        }));

        resolve();
      };

      // Handle voiceschanged event
      this.synthesis.addEventListener('voiceschanged', () => {
        loadVoicesWithTimeout();
      });

      // Try to load voices immediately
      loadVoicesWithTimeout();
    });
  }

  /**
   * Detect voice gender from name (heuristic approach)
   */
  private detectGender(voiceName: string): 'male' | 'female' | 'neutral' {
    const name = voiceName.toLowerCase();
    
    // Common female voice names
    const femaleNames = ['alice', 'anna', 'emma', 'fiona', 'karen', 'kate', 'lily', 'mary', 'sara', 'susan', 'victoria', 'zoe'];
    // Common male voice names
    const maleNames = ['alex', 'daniel', 'david', 'fred', 'george', 'james', 'john', 'oliver', 'paul', 'peter', 'richard', 'thomas', 'william'];

    for (const femaleName of femaleNames) {
      if (name.includes(femaleName)) return 'female';
    }

    for (const maleName of maleNames) {
      if (name.includes(maleName)) return 'male';
    }

    return 'neutral';
  }

  /**
   * Get preferred voice based on language and quality
   */
  private getPreferredVoice(): ITTSVoice {
    const lang = this.defaultOptions.lang;
    
    // Prioritize local voices first
    let localVoices = this.voices.filter(voice => 
      voice.lang.startsWith(lang.substring(0, 2)) && voice.localService
    );

    if (localVoices.length === 0) {
      localVoices = this.voices.filter(voice => 
        voice.lang.startsWith(lang.substring(0, 2))
      );
    }

    if (localVoices.length === 0) {
      localVoices = this.voices.filter(voice => voice.default);
    }

    if (localVoices.length === 0 && this.voices.length > 0) {
      return this.voices[0];
    }

    // Prefer higher quality voices
    return localVoices.find(voice => 
      voice.name.toLowerCase().includes('enhanced') ||
      voice.name.toLowerCase().includes('premium') ||
      voice.name.toLowerCase().includes('neural')
    ) || localVoices[0];
  }

  /**
   * Setup event handlers for synthesis events
   */
  private setupEventHandlers(): void {
    // Handle browser tab visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.isPlaying) {
        // Pause when tab becomes hidden
        this.pause();
      }
    });

    // Handle page unload
    window.addEventListener('beforeunload', () => {
      this.stop();
    });
  }

  /**
   * Speak text with advanced options and queue management
   */
  public async speak(
    text: string,
    options?: Partial<ITTSOptions>,
    callbacks?: Partial<ITTSCallbacks>,
    priority: 'low' | 'normal' | 'high' | 'urgent' = 'normal',
    interruptible: boolean = true
  ): Promise<void> {
    if (!text.trim()) {
      throw new Error('Text cannot be empty');
    }

    if (!this.synthesis) {
      throw new Error('Speech synthesis not supported');
    }

    const request: ITTSRequest = {
      text: this.preprocessText(text),
      options: { ...this.defaultOptions, ...options },
      callbacks,
      priority,
      interruptible,
      id: this.generateRequestId()
    };

    // Handle urgent requests immediately
    if (priority === 'urgent') {
      this.stop();
      await this.processRequest(request);
    } else {
      this.addToQueue(request);
      if (!this.isPlaying) {
        this.processQueue();
      }
    }
  }

  /**
   * Preprocess text for better speech synthesis
   */
  private preprocessText(text: string): string {
    return text
      // Handle abbreviations
      .replace(/\bDr\./g, 'Doctor')
      .replace(/\bMr\./g, 'Mister')
      .replace(/\bMrs\./g, 'Misses')
      .replace(/\bMs\./g, 'Miss')
      .replace(/\bProf\./g, 'Professor')
      .replace(/\bInc\./g, 'Incorporated')
      .replace(/\bLLC\./g, 'Limited Liability Company')
      .replace(/\bCorp\./g, 'Corporation')
      .replace(/\bLtd\./g, 'Limited')
      
      // Handle common legal abbreviations
      .replace(/\bv\./g, 'versus')
      .replace(/\bet al\./g, 'and others')
      .replace(/\bi\.e\./g, 'that is')
      .replace(/\be\.g\./g, 'for example')
      .replace(/\betc\./g, 'etcetera')
      
      // Handle numbers and dates
      .replace(/(\d{1,2})\/(\d{1,2})\/(\d{4})/g, '$2 $1 $3')
      .replace(/(\d+)%/g, '$1 percent')
      .replace(/\$(\d+)/g, '$1 dollars')
      
      // Clean up extra whitespace
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * Add request to queue based on priority
   */
  private addToQueue(request: ITTSRequest): void {
    // Insert based on priority
    const priorityOrder = { urgent: 4, high: 3, normal: 2, low: 1 };
    const requestPriority = priorityOrder[request.priority];

    let insertIndex = this.speechQueue.length;
    for (let i = 0; i < this.speechQueue.length; i++) {
      if (priorityOrder[this.speechQueue[i].priority] < requestPriority) {
        insertIndex = i;
        break;
      }
    }

    this.speechQueue.splice(insertIndex, 0, request);
  }

  /**
   * Process the speech queue
   */
  private async processQueue(): Promise<void> {
    if (this.speechQueue.length === 0 || this.isPlaying) {
      return;
    }

    const request = this.speechQueue.shift();
    if (request) {
      await this.processRequest(request);
      // Process next item in queue
      setTimeout(() => this.processQueue(), 100);
    }
  }

  /**
   * Process individual speech request
   */
  private async processRequest(request: ITTSRequest): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.currentRequest = request;
        const utterance = this.createUtterance(request);
        
        utterance.onstart = (event) => {
          this.isPlaying = true;
          this.isPaused = false;
          this.lastSpokenText = request.text;
          request.callbacks?.onStart?.(event);
          this.emit('start', { request, event });
        };

        utterance.onend = (event) => {
          this.isPlaying = false;
          this.currentUtterance = null;
          this.currentRequest = null;
          request.callbacks?.onEnd?.(event);
          this.emit('end', { request, event });
          resolve();
        };

        utterance.onerror = (event) => {
          this.isPlaying = false;
          this.currentUtterance = null;
          this.currentRequest = null;
          request.callbacks?.onError?.(event);
          this.emit('error', { request, event });
          
          const errorCode = this.mapSynthesisError(event.error);
          this.handleError(errorCode, `Speech synthesis error: ${event.error}`);
          reject(new Error(event.error));
        };

        utterance.onpause = (event) => {
          this.isPaused = true;
          request.callbacks?.onPause?.(event);
          this.emit('pause', { request, event });
        };

        utterance.onresume = (event) => {
          this.isPaused = false;
          request.callbacks?.onResume?.(event);
          this.emit('resume', { request, event });
        };

        utterance.onmark = (event) => {
          request.callbacks?.onMark?.(event);
          this.emit('mark', { request, event });
        };

        utterance.onboundary = (event) => {
          request.callbacks?.onBoundary?.(event);
          this.emit('boundary', { request, event });
        };

        this.currentUtterance = utterance;
        this.synthesis.speak(utterance);

      } catch (error) {
        this.handleError('PROCESSING_ERROR', 
          `Failed to process TTS request: ${error instanceof Error ? error.message : 'Unknown error'}`);
        reject(error);
      }
    });
  }

  /**
   * Create speech synthesis utterance with advanced options
   */
  private createUtterance(request: ITTSRequest): SpeechSynthesisUtterance {
    const utterance = new SpeechSynthesisUtterance(request.text);
    const options = request.options;

    // Apply voice settings
    if (options.voice) {
      const voice = this.synthesis.getVoices().find(v => 
        v.name === options.voice!.name && v.lang === options.voice!.lang
      );
      if (voice) {
        utterance.voice = voice;
      }
    }

    utterance.rate = this.clampValue(options.rate, 0.1, 10);
    utterance.pitch = this.clampValue(options.pitch, 0, 2);
    utterance.volume = this.isMuted ? 0 : this.clampValue(options.volume, 0, 1);
    utterance.lang = options.lang || this.defaultOptions.lang;

    return utterance;
  }

  /**
   * Clamp value within range
   */
  private clampValue(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }

  /**
   * Map synthesis error to voice error code
   */
  private mapSynthesisError(error: string): VoiceErrorCode {
    switch (error.toLowerCase()) {
      case 'network':
        return 'NETWORK_ERROR';
      case 'synthesis-failed':
        return 'PROCESSING_ERROR';
      case 'synthesis-unavailable':
        return 'SPEECH_SYNTHESIS_NOT_SUPPORTED';
      default:
        return 'PROCESSING_ERROR';
    }
  }

  /**
   * Stop current speech and clear queue
   */
  public stop(): void {
    if (this.synthesis.speaking) {
      this.synthesis.cancel();
    }
    
    this.isPlaying = false;
    this.isPaused = false;
    this.currentUtterance = null;
    this.currentRequest = null;
    this.speechQueue = [];
    
    this.emit('stop', {});
  }

  /**
   * Pause current speech
   */
  public pause(): void {
    if (this.synthesis.speaking && !this.synthesis.paused) {
      this.synthesis.pause();
    }
  }

  /**
   * Resume paused speech
   */
  public resume(): void {
    if (this.synthesis.paused) {
      this.synthesis.resume();
    }
  }

  /**
   * Repeat last spoken text
   */
  public async repeatLast(): Promise<void> {
    if (this.lastSpokenText) {
      await this.speak(this.lastSpokenText, this.defaultOptions, undefined, 'high');
    }
  }

  /**
   * Set volume
   */
  public setVolume(volume: number): void {
    const clampedVolume = this.clampValue(volume, 0, 1);
    this.defaultOptions.volume = clampedVolume;
    
    if (!this.isMuted && this.currentUtterance) {
      this.currentUtterance.volume = clampedVolume;
    }
  }

  /**
   * Increase volume
   */
  public increaseVolume(step: number = 0.1): number {
    const newVolume = Math.min(this.defaultOptions.volume + step, 1);
    this.setVolume(newVolume);
    return newVolume;
  }

  /**
   * Decrease volume
   */
  public decreaseVolume(step: number = 0.1): number {
    const newVolume = Math.max(this.defaultOptions.volume - step, 0);
    this.setVolume(newVolume);
    return newVolume;
  }

  /**
   * Mute/unmute audio
   */
  public toggleMute(): boolean {
    if (this.isMuted) {
      this.unmute();
    } else {
      this.mute();
    }
    return this.isMuted;
  }

  /**
   * Mute audio
   */
  public mute(): void {
    if (!this.isMuted) {
      this.volumeBeforeMute = this.defaultOptions.volume;
      this.isMuted = true;
      
      if (this.currentUtterance) {
        this.currentUtterance.volume = 0;
      }
    }
  }

  /**
   * Unmute audio
   */
  public unmute(): void {
    if (this.isMuted) {
      this.isMuted = false;
      this.setVolume(this.volumeBeforeMute);
    }
  }

  /**
   * Set speech rate
   */
  public setRate(rate: number): void {
    this.defaultOptions.rate = this.clampValue(rate, 0.1, 10);
  }

  /**
   * Set speech pitch
   */
  public setPitch(pitch: number): void {
    this.defaultOptions.pitch = this.clampValue(pitch, 0, 2);
  }

  /**
   * Set voice
   */
  public setVoice(voice: ITTSVoice): void {
    this.defaultOptions.voice = voice;
  }

  /**
   * Set language
   */
  public setLanguage(lang: string): void {
    this.defaultOptions.lang = lang;
    
    // Try to find a better voice for the new language
    const voicesForLang = this.voices.filter(voice => 
      voice.lang.startsWith(lang.substring(0, 2))
    );
    
    if (voicesForLang.length > 0) {
      this.defaultOptions.voice = voicesForLang[0];
    }
  }

  /**
   * Get available voices
   */
  public getVoices(): ITTSVoice[] {
    return [...this.voices];
  }

  /**
   * Get voices for specific language
   */
  public getVoicesForLanguage(lang: string): ITTSVoice[] {
    return this.voices.filter(voice => voice.lang.startsWith(lang.substring(0, 2)));
  }

  /**
   * Get current status
   */
  public getStatus(): {
    isPlaying: boolean;
    isPaused: boolean;
    isMuted: boolean;
    queueLength: number;
    currentOptions: ITTSOptions;
    currentText?: string;
  } {
    return {
      isPlaying: this.isPlaying,
      isPaused: this.isPaused,
      isMuted: this.isMuted,
      queueLength: this.speechQueue.length,
      currentOptions: { ...this.defaultOptions },
      currentText: this.currentRequest?.text
    };
  }

  /**
   * Update default options
   */
  public updateOptions(options: Partial<ITTSOptions>): void {
    this.defaultOptions = { ...this.defaultOptions, ...options };
  }

  /**
   * Generate unique request ID
   */
  private generateRequestId(): string {
    return `tts_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Handle errors
   */
  private handleError(code: VoiceErrorCode, message: string): void {
    const error: IVoiceError = {
      code,
      message,
      timestamp: new Date(),
      recoverable: code !== 'SPEECH_SYNTHESIS_NOT_SUPPORTED',
      details: {
        isPlaying: this.isPlaying,
        queueLength: this.speechQueue.length,
        options: this.defaultOptions
      }
    };

    console.error('Text-to-Speech Error:', error);
    this.emit('error', error);
  }

  /**
   * Event system
   */
  public on(event: string, callback: (data: any) => void): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
  }

  public off(event: string, callback: (data: any) => void): void {
    this.eventListeners.get(event)?.delete(callback);
  }

  private emit(event: string, data: any): void {
    this.eventListeners.get(event)?.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in TTS event callback:', error);
      }
    });
  }

  /**
   * Test speech synthesis capabilities
   */
  public static testCapabilities(): {
    supported: boolean;
    voiceCount: number;
    features: {
      rate: boolean;
      pitch: boolean;
      volume: boolean;
      voices: boolean;
      events: boolean;
    };
  } {
    const synthesis = window.speechSynthesis;
    const supported = !!synthesis && typeof synthesis.speak === 'function';
    
    let voiceCount = 0;
    let features = {
      rate: false,
      pitch: false,
      volume: false,
      voices: false,
      events: false
    };

    if (supported) {
      try {
        const testUtterance = new SpeechSynthesisUtterance('test');
        
        voiceCount = synthesis.getVoices().length;
        features = {
          rate: 'rate' in testUtterance,
          pitch: 'pitch' in testUtterance,
          volume: 'volume' in testUtterance,
          voices: 'voice' in testUtterance,
          events: 'onstart' in testUtterance
        };
      } catch (error) {
        console.warn('Error testing TTS capabilities:', error);
      }
    }

    return {
      supported,
      voiceCount,
      features
    };
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.stop();
    this.eventListeners.clear();
  }
}