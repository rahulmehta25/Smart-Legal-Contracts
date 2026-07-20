/**
 * Audio Processor Module
 * Real-time audio processing with voice activity detection, noise suppression,
 * audio analysis, and biometric voice features extraction
 */

import {
  IAudioConfig,
  IAudioMetrics,
  IAudioProcessorCallbacks,
  IVoiceFeatures,
  IVADResult,
  IVADConfig,
  IEmotionResult
} from '@/types/voice';

export class AudioProcessor {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private analyzer: AnalyserNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private config: IAudioConfig;
  private callbacks: Partial<IAudioProcessorCallbacks>;
  private vadConfig: IVADConfig;
  
  // Audio analysis data
  private frequencyData: Uint8Array = new Uint8Array();
  private timeData: Uint8Array = new Uint8Array();
  private audioBuffer: Float32Array = new Float32Array();
  
  // Voice Activity Detection
  private vadHistory: boolean[] = [];
  private energyHistory: number[] = [];
  private vadThreshold: number = 0.01;
  private silenceFrames: number = 0;
  private speechFrames: number = 0;
  private isSpeechActive: boolean = false;
  
  // Real-time metrics
  private currentMetrics: IAudioMetrics = {
    volume: 0,
    frequency: 0,
    clarity: 0,
    backgroundNoise: 0,
    speechDetected: false
  };

  // Voice features for biometrics
  private voiceFeatures: IVoiceFeatures = {
    pitch: [],
    formants: [],
    mfcc: [],
    spectralCentroid: 0,
    spectralRolloff: 0,
    zeroCrossingRate: 0
  };

  // Default configurations
  private defaultConfig: IAudioConfig = {
    sampleRate: 44100,
    bufferSize: 4096,
    channels: 1,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true
  };

  private defaultVADConfig: IVADConfig = {
    sensitivity: 0.5,
    minSpeechDuration: 300, // ms
    minSilenceDuration: 500, // ms
    energyThreshold: 0.01
  };

  constructor(
    config?: Partial<IAudioConfig>,
    vadConfig?: Partial<IVADConfig>,
    callbacks?: Partial<IAudioProcessorCallbacks>
  ) {
    this.config = { ...this.defaultConfig, ...config };
    this.vadConfig = { ...this.defaultVADConfig, ...vadConfig };
    this.callbacks = callbacks || {};
  }

  /**
   * Initialize audio processing
   */
  public async initialize(): Promise<void> {
    try {
      // Create audio context
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: this.config.sampleRate
      });

      // Request microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
          echoCancellation: this.config.echoCancellation,
          noiseSuppression: this.config.noiseSuppression,
          autoGainControl: this.config.autoGainControl
        }
      });

      await this.setupAudioNodes();
      this.startProcessing();

    } catch (error) {
      throw new Error(`Failed to initialize audio processor: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Setup audio processing nodes
   */
  private async setupAudioNodes(): Promise<void> {
    if (!this.audioContext || !this.mediaStream) {
      throw new Error('Audio context or media stream not initialized');
    }

    // Create audio source
    this.source = this.audioContext.createMediaStreamSource(this.mediaStream);

    // Create analyzer node
    this.analyzer = this.audioContext.createAnalyser();
    this.analyzer.fftSize = 2048;
    this.analyzer.smoothingTimeConstant = 0.8;
    this.analyzer.minDecibels = -90;
    this.analyzer.maxDecibels = -10;

    // Create script processor for real-time analysis
    this.processor = this.audioContext.createScriptProcessor(
      this.config.bufferSize,
      this.config.channels,
      this.config.channels
    );

    // Setup processing chain
    this.source.connect(this.analyzer);
    this.analyzer.connect(this.processor);
    this.processor.connect(this.audioContext.destination);

    // Initialize data arrays
    this.frequencyData = new Uint8Array(this.analyzer.frequencyBinCount);
    this.timeData = new Uint8Array(this.analyzer.frequencyBinCount);
    this.audioBuffer = new Float32Array(this.config.bufferSize);

    // Setup audio processing callback
    this.processor.onaudioprocess = this.processAudioData.bind(this);
  }

  /**
   * Start audio processing
   */
  private startProcessing(): void {
    if (this.audioContext && this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
  }

  /**
   * Process real-time audio data
   */
  private processAudioData(event: AudioProcessingEvent): void {
    if (!this.analyzer) return;

    const inputBuffer = event.inputBuffer;
    const inputData = inputBuffer.getChannelData(0);
    
    // Copy input data to our buffer
    this.audioBuffer.set(inputData);

    // Get frequency and time domain data
    this.analyzer.getByteFrequencyData(this.frequencyData);
    this.analyzer.getByteTimeDomainData(this.timeData);

    // Analyze audio
    this.analyzeAudio(inputData);

    // Voice Activity Detection
    const vadResult = this.performVAD(inputData);
    this.updateVADState(vadResult);

    // Extract voice features for biometrics
    this.extractVoiceFeatures(inputData);

    // Update metrics
    this.updateMetrics();

    // Emit callbacks
    this.callbacks.onAudioData?.(this.audioBuffer);
    this.callbacks.onVolumeChange?.(this.currentMetrics.volume);

    if (vadResult.isSpeech && !this.isSpeechActive) {
      this.callbacks.onSpeechDetected?.();
    } else if (!vadResult.isSpeech && this.isSpeechActive) {
      this.callbacks.onSpeechEnded?.();
    }

    if (this.currentMetrics.backgroundNoise > 0.7) {
      this.callbacks.onNoiseDetected?.(this.currentMetrics.backgroundNoise);
    }
  }

  /**
   * Analyze audio characteristics
   */
  private analyzeAudio(inputData: Float32Array): void {
    // Calculate RMS volume
    let sumSquares = 0;
    for (let i = 0; i < inputData.length; i++) {
      sumSquares += inputData[i] * inputData[i];
    }
    const rms = Math.sqrt(sumSquares / inputData.length);
    this.currentMetrics.volume = rms;

    // Calculate zero crossing rate
    let crossings = 0;
    for (let i = 1; i < inputData.length; i++) {
      if ((inputData[i] > 0) !== (inputData[i - 1] > 0)) {
        crossings++;
      }
    }
    this.voiceFeatures.zeroCrossingRate = crossings / inputData.length;

    // Calculate spectral centroid
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < this.frequencyData.length; i++) {
      const magnitude = this.frequencyData[i] / 255.0;
      const frequency = (i * this.config.sampleRate) / (2 * this.frequencyData.length);
      numerator += frequency * magnitude;
      denominator += magnitude;
    }
    this.voiceFeatures.spectralCentroid = denominator > 0 ? numerator / denominator : 0;

    // Calculate spectral rolloff (85% of energy)
    let energySum = 0;
    let totalEnergy = 0;
    for (let i = 0; i < this.frequencyData.length; i++) {
      totalEnergy += this.frequencyData[i];
    }
    const targetEnergy = totalEnergy * 0.85;
    
    for (let i = 0; i < this.frequencyData.length; i++) {
      energySum += this.frequencyData[i];
      if (energySum >= targetEnergy) {
        this.voiceFeatures.spectralRolloff = (i * this.config.sampleRate) / (2 * this.frequencyData.length);
        break;
      }
    }

    // Estimate fundamental frequency (pitch)
    const pitch = this.estimatePitch(inputData);
    if (pitch > 0) {
      this.voiceFeatures.pitch.push(pitch);
      if (this.voiceFeatures.pitch.length > 100) {
        this.voiceFeatures.pitch.shift(); // Keep only recent values
      }
    }

    // Calculate clarity (signal-to-noise ratio approximation)
    const peakFrequency = this.findPeakFrequency();
    const noiseLevel = this.estimateNoiseLevel();
    this.currentMetrics.clarity = peakFrequency > 0 ? Math.min(peakFrequency / Math.max(noiseLevel, 0.001), 1) : 0;
    this.currentMetrics.backgroundNoise = noiseLevel;
    this.currentMetrics.frequency = peakFrequency;
  }

  /**
   * Estimate pitch using autocorrelation
   */
  private estimatePitch(buffer: Float32Array): number {
    const sampleRate = this.config.sampleRate;
    const minPeriod = Math.floor(sampleRate / 800); // 800 Hz max
    const maxPeriod = Math.floor(sampleRate / 50);  // 50 Hz min

    let bestCorrelation = 0;
    let bestPeriod = 0;

    for (let period = minPeriod; period < maxPeriod; period++) {
      let correlation = 0;
      for (let i = 0; i < buffer.length - period; i++) {
        correlation += buffer[i] * buffer[i + period];
      }
      
      if (correlation > bestCorrelation) {
        bestCorrelation = correlation;
        bestPeriod = period;
      }
    }

    return bestPeriod > 0 ? sampleRate / bestPeriod : 0;
  }

  /**
   * Find peak frequency in the spectrum
   */
  private findPeakFrequency(): number {
    let maxMagnitude = 0;
    let peakIndex = 0;

    for (let i = 1; i < this.frequencyData.length; i++) {
      if (this.frequencyData[i] > maxMagnitude) {
        maxMagnitude = this.frequencyData[i];
        peakIndex = i;
      }
    }

    return (peakIndex * this.config.sampleRate) / (2 * this.frequencyData.length);
  }

  /**
   * Estimate background noise level
   */
  private estimateNoiseLevel(): number {
    // Use the lower frequencies as noise estimation
    let noiseSum = 0;
    const noiseRange = Math.min(10, this.frequencyData.length);
    
    for (let i = 0; i < noiseRange; i++) {
      noiseSum += this.frequencyData[i];
    }
    
    return (noiseSum / noiseRange) / 255.0;
  }

  /**
   * Perform Voice Activity Detection
   */
  private performVAD(buffer: Float32Array): IVADResult {
    // Calculate energy
    let energy = 0;
    for (let i = 0; i < buffer.length; i++) {
      energy += buffer[i] * buffer[i];
    }
    energy = energy / buffer.length;

    // Store energy history for adaptive thresholding
    this.energyHistory.push(energy);
    if (this.energyHistory.length > 50) {
      this.energyHistory.shift();
    }

    // Calculate adaptive threshold
    const avgEnergy = this.energyHistory.reduce((a, b) => a + b, 0) / this.energyHistory.length;
    const dynamicThreshold = Math.max(avgEnergy * (1 + this.vadConfig.sensitivity), this.vadConfig.energyThreshold);

    // Determine if speech is present
    const isSpeech = energy > dynamicThreshold;
    const confidence = Math.min(energy / dynamicThreshold, 1);

    // Store VAD history
    this.vadHistory.push(isSpeech);
    if (this.vadHistory.length > 20) {
      this.vadHistory.shift();
    }

    return {
      isSpeech,
      confidence,
      energy,
      timestamp: Date.now()
    };
  }

  /**
   * Update Voice Activity Detection state
   */
  private updateVADState(vadResult: IVADResult): void {
    if (vadResult.isSpeech) {
      this.speechFrames++;
      this.silenceFrames = 0;
    } else {
      this.silenceFrames++;
      this.speechFrames = 0;
    }

    // Convert frames to milliseconds (approximate)
    const frameTimeMs = (this.config.bufferSize / this.config.sampleRate) * 1000;
    const speechTimeMs = this.speechFrames * frameTimeMs;
    const silenceTimeMs = this.silenceFrames * frameTimeMs;

    // Determine speech state based on duration thresholds
    if (!this.isSpeechActive && speechTimeMs >= this.vadConfig.minSpeechDuration) {
      this.isSpeechActive = true;
    } else if (this.isSpeechActive && silenceTimeMs >= this.vadConfig.minSilenceDuration) {
      this.isSpeechActive = false;
    }

    this.currentMetrics.speechDetected = this.isSpeechActive;
  }

  /**
   * Extract voice features for biometric analysis
   */
  private extractVoiceFeatures(buffer: Float32Array): void {
    if (!this.isSpeechActive) return;

    // Extract formants using LPC analysis (simplified)
    const formants = this.extractFormants(buffer);
    this.voiceFeatures.formants = formants;

    // Extract MFCC features (simplified implementation)
    const mfcc = this.extractMFCC(this.frequencyData);
    this.voiceFeatures.mfcc = mfcc;
  }

  /**
   * Extract formant frequencies (simplified LPC analysis)
   */
  private extractFormants(buffer: Float32Array): number[] {
    // This is a simplified formant extraction
    // In production, use more sophisticated LPC analysis
    const formants: number[] = [];
    
    // Find peaks in the frequency spectrum
    const peaks: number[] = [];
    for (let i = 1; i < this.frequencyData.length - 1; i++) {
      if (this.frequencyData[i] > this.frequencyData[i-1] && 
          this.frequencyData[i] > this.frequencyData[i+1] &&
          this.frequencyData[i] > 50) { // Minimum threshold
        const frequency = (i * this.config.sampleRate) / (2 * this.frequencyData.length);
        peaks.push(frequency);
      }
    }

    // Return the first few formants (typically F1, F2, F3)
    return peaks.slice(0, 3);
  }

  /**
   * Extract MFCC features (simplified)
   */
  private extractMFCC(frequencyData: Uint8Array): number[] {
    // This is a very simplified MFCC implementation
    // In production, use a proper MFCC library
    const mfcc: number[] = [];
    const numCoefficients = 13;
    const melFilters = this.createMelFilterBank(26);

    // Apply mel filter bank
    const melEnergy: number[] = [];
    for (let i = 0; i < melFilters.length; i++) {
      let energy = 0;
      for (let j = 0; j < frequencyData.length; j++) {
        energy += frequencyData[j] * melFilters[i][j] || 0;
      }
      melEnergy.push(Math.log(Math.max(energy, 1)));
    }

    // Apply DCT to get MFCC coefficients
    for (let i = 0; i < numCoefficients; i++) {
      let coefficient = 0;
      for (let j = 0; j < melEnergy.length; j++) {
        coefficient += melEnergy[j] * Math.cos(i * (j + 0.5) * Math.PI / melEnergy.length);
      }
      mfcc.push(coefficient);
    }

    return mfcc;
  }

  /**
   * Create mel filter bank (simplified)
   */
  private createMelFilterBank(numFilters: number): number[][] {
    const filters: number[][] = [];
    const nyquist = this.config.sampleRate / 2;
    const melMax = this.hzToMel(nyquist);
    
    for (let i = 0; i < numFilters; i++) {
      const filter = new Array(this.frequencyData.length).fill(0);
      filters.push(filter);
    }

    return filters;
  }

  /**
   * Convert Hz to Mel scale
   */
  private hzToMel(hz: number): number {
    return 2595 * Math.log10(1 + hz / 700);
  }

  /**
   * Convert Mel to Hz scale
   */
  private melToHz(mel: number): number {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }

  /**
   * Update current metrics
   */
  private updateMetrics(): void {
    // Metrics are updated in real-time during audio analysis
    // This method can be extended for additional metric calculations
  }

  /**
   * Detect emotion from voice characteristics (basic implementation)
   */
  public detectEmotionFromVoice(): IEmotionResult {
    const avgPitch = this.voiceFeatures.pitch.length > 0 
      ? this.voiceFeatures.pitch.reduce((a, b) => a + b, 0) / this.voiceFeatures.pitch.length 
      : 0;

    const pitchVariability = this.calculatePitchVariability();
    const energy = this.currentMetrics.volume;
    const spectralCentroid = this.voiceFeatures.spectralCentroid;

    // Simple emotion classification based on acoustic features
    let emotion: string = 'neutral';
    let valence = 0;
    let arousal = energy;

    if (avgPitch > 200 && energy > 0.5) {
      emotion = 'excited';
      valence = 0.7;
      arousal = 0.8;
    } else if (avgPitch > 180 && pitchVariability > 50) {
      emotion = 'happy';
      valence = 0.6;
      arousal = 0.6;
    } else if (avgPitch < 120 && energy < 0.3) {
      emotion = 'sad';
      valence = -0.5;
      arousal = 0.2;
    } else if (energy > 0.7 && spectralCentroid > 2000) {
      emotion = 'angry';
      valence = -0.3;
      arousal = 0.9;
    } else if (pitchVariability > 80 && this.voiceFeatures.zeroCrossingRate > 0.1) {
      emotion = 'frustrated';
      valence = -0.4;
      arousal = 0.7;
    }

    return {
      emotion: emotion as any,
      confidence: 0.6,
      valence,
      arousal
    };
  }

  /**
   * Calculate pitch variability
   */
  private calculatePitchVariability(): number {
    if (this.voiceFeatures.pitch.length < 2) return 0;

    const mean = this.voiceFeatures.pitch.reduce((a, b) => a + b, 0) / this.voiceFeatures.pitch.length;
    const variance = this.voiceFeatures.pitch.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / this.voiceFeatures.pitch.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Get current audio metrics
   */
  public getMetrics(): IAudioMetrics {
    return { ...this.currentMetrics };
  }

  /**
   * Get voice features for biometric analysis
   */
  public getVoiceFeatures(): IVoiceFeatures {
    return {
      pitch: [...this.voiceFeatures.pitch],
      formants: [...this.voiceFeatures.formants],
      mfcc: [...this.voiceFeatures.mfcc],
      spectralCentroid: this.voiceFeatures.spectralCentroid,
      spectralRolloff: this.voiceFeatures.spectralRolloff,
      zeroCrossingRate: this.voiceFeatures.zeroCrossingRate
    };
  }

  /**
   * Update configuration
   */
  public updateConfig(config: Partial<IAudioConfig>): void {
    this.config = { ...this.config, ...config };
    // Note: Some config changes might require reinitialization
  }

  /**
   * Update VAD configuration
   */
  public updateVADConfig(vadConfig: Partial<IVADConfig>): void {
    this.vadConfig = { ...this.vadConfig, ...vadConfig };
  }

  /**
   * Update callbacks
   */
  public updateCallbacks(callbacks: Partial<IAudioProcessorCallbacks>): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  /**
   * Get current status
   */
  public getStatus(): {
    isInitialized: boolean;
    isProcessing: boolean;
    sampleRate: number;
    bufferSize: number;
    vadActive: boolean;
    speechDetected: boolean;
  } {
    return {
      isInitialized: !!this.audioContext,
      isProcessing: !!this.processor,
      sampleRate: this.config.sampleRate,
      bufferSize: this.config.bufferSize,
      vadActive: this.isSpeechActive,
      speechDetected: this.currentMetrics.speechDetected
    };
  }

  /**
   * Stop audio processing and cleanup
   */
  public stop(): void {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.analyzer) {
      this.analyzer.disconnect();
      this.analyzer = null;
    }

    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  /**
   * Test audio processing capabilities
   */
  public static testCapabilities(): {
    audioContext: boolean;
    mediaDevices: boolean;
    scriptProcessor: boolean;
    analyser: boolean;
    webAudio: boolean;
  } {
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    
    return {
      audioContext: !!AudioContext,
      mediaDevices: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
      scriptProcessor: !!(AudioContext && AudioContext.prototype.createScriptProcessor),
      analyser: !!(AudioContext && AudioContext.prototype.createAnalyser),
      webAudio: !!(AudioContext && navigator.mediaDevices)
    };
  }
}