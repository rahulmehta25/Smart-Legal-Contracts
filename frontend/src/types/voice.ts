/**
 * Voice Interface Types
 * Comprehensive type definitions for voice recognition, synthesis, and accessibility features
 */

// Speech Recognition Types
export interface ISpeechRecognitionResult {
  transcript: string;
  confidence: number;
  isFinal: boolean;
  alternatives: ISpeechRecognitionAlternative[];
}

export interface ISpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

export interface ISpeechRecognitionConfig {
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  language: string;
  grammars?: SpeechGrammarList;
}

export interface ISpeechRecognitionCallbacks {
  onResult: (results: ISpeechRecognitionResult[]) => void;
  onError: (error: SpeechRecognitionErrorEvent) => void;
  onStart: () => void;
  onEnd: () => void;
  onSpeechStart: () => void;
  onSpeechEnd: () => void;
  onSoundStart: () => void;
  onSoundEnd: () => void;
  onNoMatch: () => void;
}

// Voice Command Types
export interface IVoiceCommand {
  id: string;
  phrases: string[];
  action: VoiceCommandAction;
  description: string;
  category: VoiceCommandCategory;
  parameters?: Record<string, unknown>;
  confidence: number;
  enabled: boolean;
}

export type VoiceCommandAction = 
  | 'ANALYZE_DOCUMENT'
  | 'SHOW_ARBITRATION_CLAUSES'
  | 'READ_SUMMARY'
  | 'COMPARE_VERSIONS'
  | 'EXPORT_PDF'
  | 'NAVIGATE'
  | 'SEARCH'
  | 'HELP'
  | 'REPEAT'
  | 'STOP'
  | 'PAUSE'
  | 'RESUME'
  | 'INCREASE_VOLUME'
  | 'DECREASE_VOLUME'
  | 'MUTE'
  | 'UNMUTE'
  | 'TOGGLE_DARK_MODE'
  | 'INCREASE_FONT_SIZE'
  | 'DECREASE_FONT_SIZE'
  | 'HIGH_CONTRAST'
  | 'FOCUS_NEXT'
  | 'FOCUS_PREVIOUS';

export type VoiceCommandCategory =
  | 'ANALYSIS'
  | 'NAVIGATION'
  | 'ACCESSIBILITY'
  | 'PLAYBACK'
  | 'SYSTEM'
  | 'HELP';

export interface IVoiceCommandResult {
  command: IVoiceCommand;
  confidence: number;
  parameters: Record<string, unknown>;
  transcript: string;
  timestamp: Date;
}

export interface IVoiceCommandContext {
  currentPage: string;
  selectedElement?: string;
  documentLoaded: boolean;
  analysisInProgress: boolean;
  resultsAvailable: boolean;
  readingInProgress: boolean;
}

// Natural Language Understanding Types
export interface INLUIntent {
  name: string;
  confidence: number;
  entities: INLUEntity[];
}

export interface INLUEntity {
  entity: string;
  value: string;
  confidence: number;
  start: number;
  end: number;
}

export interface INLUResult {
  intent: INLUIntent;
  text: string;
  confidence: number;
}

// Text-to-Speech Types
export interface ITTSVoice {
  name: string;
  lang: string;
  gender: 'male' | 'female' | 'neutral';
  localService: boolean;
  voiceURI: string;
  default: boolean;
}

export interface ITTSOptions {
  voice?: ITTSVoice;
  rate: number;
  pitch: number;
  volume: number;
  lang: string;
}

export interface ITTSCallbacks {
  onStart: (event: SpeechSynthesisEvent) => void;
  onEnd: (event: SpeechSynthesisEvent) => void;
  onError: (event: SpeechSynthesisErrorEvent) => void;
  onPause: (event: SpeechSynthesisEvent) => void;
  onResume: (event: SpeechSynthesisEvent) => void;
  onMark: (event: SpeechSynthesisEvent) => void;
  onBoundary: (event: SpeechSynthesisEvent) => void;
}

export interface ITTSRequest {
  text: string;
  options: Partial<ITTSOptions>;
  callbacks?: Partial<ITTSCallbacks>;
  priority: 'low' | 'normal' | 'high' | 'urgent';
  interruptible: boolean;
  id?: string;
}

// Voice Assistant Types
export interface IVoiceAssistantConfig {
  wakeWord?: string;
  confidenceThreshold: number;
  speechSynthesis: ITTSOptions;
  speechRecognition: ISpeechRecognitionConfig;
  nlpProvider: 'local' | 'openai' | 'azure';
  emotionDetection: boolean;
  voiceBiometrics: boolean;
}

export interface IVoiceAssistantState {
  isListening: boolean;
  isSpeaking: boolean;
  isProcessing: boolean;
  currentCommand?: IVoiceCommand;
  error?: string;
  confidence: number;
}

export interface IVoiceAssistantResponse {
  text: string;
  action?: VoiceCommandAction;
  parameters?: Record<string, unknown>;
  suggestions?: string[];
  followUp?: string;
}

// Audio Processing Types
export interface IAudioConfig {
  sampleRate: number;
  bufferSize: number;
  channels: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
}

export interface IAudioMetrics {
  volume: number;
  frequency: number;
  clarity: number;
  backgroundNoise: number;
  speechDetected: boolean;
}

export interface IAudioProcessorCallbacks {
  onAudioData: (audioBuffer: Float32Array) => void;
  onVolumeChange: (volume: number) => void;
  onSpeechDetected: () => void;
  onSpeechEnded: () => void;
  onNoiseDetected: (level: number) => void;
}

// Voice Biometrics Types
export interface IVoiceBiometricProfile {
  id: string;
  userId: string;
  voicePrint: Float32Array;
  features: IVoiceFeatures;
  confidence: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface IVoiceFeatures {
  pitch: number[];
  formants: number[];
  mfcc: number[];
  spectralCentroid: number;
  spectralRolloff: number;
  zeroCrossingRate: number;
}

export interface IVoiceVerificationResult {
  verified: boolean;
  confidence: number;
  userId?: string;
  similarity: number;
}

// Emotion Detection Types
export interface IEmotionResult {
  emotion: EmotionType;
  confidence: number;
  valence: number; // -1 to 1 (negative to positive)
  arousal: number; // 0 to 1 (calm to excited)
}

export type EmotionType = 
  | 'neutral'
  | 'happy'
  | 'sad'
  | 'angry'
  | 'fearful'
  | 'disgusted'
  | 'surprised'
  | 'excited'
  | 'frustrated'
  | 'confused';

// Meeting Transcription Types
export interface ISpeakerSegment {
  speakerId: string;
  speakerName?: string;
  startTime: number;
  endTime: number;
  text: string;
  confidence: number;
  emotion?: IEmotionResult;
}

export interface IMeetingTranscription {
  id: string;
  title: string;
  startTime: Date;
  endTime?: Date;
  speakers: string[];
  segments: ISpeakerSegment[];
  summary?: string;
  keyPoints: string[];
  actionItems: string[];
}

export interface ISpeakerDiarizationConfig {
  minSpeakers: number;
  maxSpeakers: number;
  segmentLength: number;
  overlapDuration: number;
}

// Translation Types
export interface IVoiceTranslationConfig {
  sourceLanguage: string;
  targetLanguage: string;
  realTime: boolean;
  preserveEmotion: boolean;
}

export interface ITranslationResult {
  originalText: string;
  translatedText: string;
  sourceLanguage: string;
  targetLanguage: string;
  confidence: number;
  detectedLanguage?: string;
}

// Voice Activity Detection Types
export interface IVADResult {
  isSpeech: boolean;
  confidence: number;
  energy: number;
  timestamp: number;
}

export interface IVADConfig {
  sensitivity: number;
  minSpeechDuration: number;
  minSilenceDuration: number;
  energyThreshold: number;
}

// Offline Voice Types
export interface IOfflineModel {
  name: string;
  language: string;
  size: number;
  version: string;
  downloaded: boolean;
  accuracy: number;
  type: 'recognition' | 'synthesis' | 'nlu';
}

export interface IOfflineVoiceConfig {
  enableOfflineRecognition: boolean;
  enableOfflineSynthesis: boolean;
  modelPath: string;
  fallbackToOnline: boolean;
  autoDownloadModels: boolean;
}

// Voice UI Component Types
export interface IVoiceButtonProps {
  isListening: boolean;
  isProcessing: boolean;
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
  variant?: 'primary' | 'secondary' | 'danger';
  onStartListening: () => void;
  onStopListening: () => void;
  className?: string;
  id?: string;
}

export interface IWaveformVisualizerProps {
  audioData: Float32Array;
  width: number;
  height: number;
  color?: string;
  backgroundColor?: string;
  animated?: boolean;
  className?: string;
  id?: string;
}

export interface IVoiceCommandHelperProps {
  visible: boolean;
  commands: IVoiceCommand[];
  currentCategory?: VoiceCommandCategory;
  onCommandSelect: (command: IVoiceCommand) => void;
  onCategoryChange: (category: VoiceCommandCategory) => void;
  className?: string;
  id?: string;
}

export interface ITranscriptionDisplayProps {
  transcript: string;
  isInterim: boolean;
  confidence: number;
  showConfidence?: boolean;
  maxLines?: number;
  className?: string;
  id?: string;
}

export interface IVoiceSettingsPanelProps {
  config: IVoiceAssistantConfig;
  onConfigChange: (config: Partial<IVoiceAssistantConfig>) => void;
  availableVoices: ITTSVoice[];
  className?: string;
  id?: string;
}

// Error Types
export interface IVoiceError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: Date;
  recoverable: boolean;
}

export type VoiceErrorCode =
  | 'SPEECH_RECOGNITION_NOT_SUPPORTED'
  | 'MICROPHONE_ACCESS_DENIED'
  | 'SPEECH_SYNTHESIS_NOT_SUPPORTED'
  | 'NETWORK_ERROR'
  | 'PROCESSING_ERROR'
  | 'COMMAND_NOT_RECOGNIZED'
  | 'CONFIDENCE_TOO_LOW'
  | 'TIMEOUT_ERROR'
  | 'AUDIO_CONTEXT_ERROR'
  | 'MODEL_LOAD_ERROR';

// Event Types
export interface IVoiceEvent {
  type: string;
  payload: Record<string, unknown>;
  timestamp: Date;
}

export type VoiceEventType =
  | 'listening_started'
  | 'listening_stopped'
  | 'speech_detected'
  | 'speech_ended'
  | 'command_recognized'
  | 'command_executed'
  | 'synthesis_started'
  | 'synthesis_completed'
  | 'error_occurred'
  | 'config_changed';