/**
 * Voice Assistant Module
 * AI-powered voice assistant with natural language understanding,
 * context-aware responses, and emotion detection capabilities
 */

import {
  IVoiceAssistantConfig,
  IVoiceAssistantState,
  IVoiceAssistantResponse,
  IVoiceCommandResult,
  IEmotionResult,
  IVoiceCommandContext,
  VoiceCommandAction
} from '@/types/voice';

import { SpeechRecognitionService } from './speechRecognition';
import { TextToSpeechService } from './textToSpeech';
import { VoiceCommandProcessor } from './voiceCommands';

export class VoiceAssistant {
  private speechRecognition: SpeechRecognitionService;
  private textToSpeech: TextToSpeechService;
  private commandProcessor: VoiceCommandProcessor;
  private config: IVoiceAssistantConfig;
  private state: IVoiceAssistantState;
  private context: IVoiceCommandContext;
  private conversationHistory: Array<{
    timestamp: Date;
    userInput: string;
    response: IVoiceAssistantResponse;
    emotion?: IEmotionResult;
  }> = [];

  // Event handlers
  private eventHandlers: Map<string, Set<(data: any) => void>> = new Map();

  // Response templates
  private responseTemplates: Map<VoiceCommandAction, string[]> = new Map();

  // Default configuration
  private defaultConfig: IVoiceAssistantConfig = {
    confidenceThreshold: 0.7,
    speechSynthesis: {
      rate: 1.0,
      pitch: 1.0,
      volume: 1.0,
      lang: 'en-US'
    },
    speechRecognition: {
      continuous: true,
      interimResults: true,
      maxAlternatives: 3,
      language: 'en-US'
    },
    nlpProvider: 'local',
    emotionDetection: true,
    voiceBiometrics: false
  };

  // Initial state
  private initialState: IVoiceAssistantState = {
    isListening: false,
    isSpeaking: false,
    isProcessing: false,
    confidence: 0,
    error: undefined,
    currentCommand: undefined
  };

  constructor(
    context: IVoiceCommandContext,
    config?: Partial<IVoiceAssistantConfig>
  ) {
    this.config = { ...this.defaultConfig, ...config };
    this.state = { ...this.initialState };
    this.context = context;

    this.initializeServices();
    this.loadResponseTemplates();
    this.setupEventHandlers();
  }

  /**
   * Initialize voice services
   */
  private async initializeServices(): Promise<void> {
    try {
      // Initialize speech recognition
      this.speechRecognition = new SpeechRecognitionService(
        this.config.speechRecognition,
        {
          onResult: this.handleSpeechResult.bind(this),
          onError: this.handleSpeechError.bind(this),
          onStart: this.handleSpeechStart.bind(this),
          onEnd: this.handleSpeechEnd.bind(this)
        }
      );

      // Initialize text-to-speech
      this.textToSpeech = new TextToSpeechService(this.config.speechSynthesis);

      // Initialize command processor
      this.commandProcessor = new VoiceCommandProcessor(
        this.context,
        this.config.confidenceThreshold
      );

      this.emit('initialized', { success: true });
    } catch (error) {
      this.handleError(`Failed to initialize voice assistant: ${error instanceof Error ? error.message : 'Unknown error'}`);
      this.emit('initialized', { success: false, error });
    }
  }

  /**
   * Load response templates for different commands
   */
  private loadResponseTemplates(): void {
    this.responseTemplates.set('ANALYZE_DOCUMENT', [
      'Starting document analysis for arbitration clauses.',
      'I\'ll analyze this document for arbitration provisions.',
      'Beginning arbitration clause detection.',
      'Scanning document for dispute resolution clauses.'
    ]);

    this.responseTemplates.set('SHOW_ARBITRATION_CLAUSES', [
      'Here are the arbitration clauses I found.',
      'I\'ve identified the following arbitration provisions.',
      'These are the dispute resolution clauses in the document.',
      'I found these arbitration-related sections.'
    ]);

    this.responseTemplates.set('READ_SUMMARY', [
      'Here\'s a summary of the analysis results.',
      'Let me read you the key findings.',
      'Here\'s what I discovered in the document.',
      'This is the analysis summary.'
    ]);

    this.responseTemplates.set('COMPARE_VERSIONS', [
      'I\'ll compare this version with the previous one.',
      'Starting version comparison analysis.',
      'Checking for changes between document versions.',
      'Comparing the current and previous versions.'
    ]);

    this.responseTemplates.set('EXPORT_PDF', [
      'Generating PDF report with the analysis results.',
      'Creating a downloadable PDF of the findings.',
      'Exporting analysis results to PDF format.',
      'Preparing your PDF report.'
    ]);

    this.responseTemplates.set('NAVIGATE', [
      'Navigating to the requested page.',
      'Taking you to that section.',
      'Opening the requested page.',
      'Switching to that view.'
    ]);

    this.responseTemplates.set('HELP', [
      'I can help you analyze documents, read results, and navigate the interface. What would you like to do?',
      'Here are some things you can say: "Analyze document", "Read summary", "Show results", or "Help with navigation".',
      'I understand commands for document analysis, navigation, and accessibility. Try saying "What can you do?" for more options.',
      'You can ask me to analyze documents, read results aloud, or help with navigation. What would you like to do?'
    ]);

    this.responseTemplates.set('TOGGLE_DARK_MODE', [
      'Switching to dark mode.',
      'Dark theme enabled.',
      'Changed to dark mode for better visibility.',
      'Dark mode is now active.'
    ]);

    this.responseTemplates.set('INCREASE_FONT_SIZE', [
      'Font size increased.',
      'Text is now larger.',
      'Made the text bigger for better readability.',
      'Font size has been increased.'
    ]);

    this.responseTemplates.set('HIGH_CONTRAST', [
      'High contrast mode enabled.',
      'Switched to high contrast for better visibility.',
      'High contrast theme is now active.',
      'Enhanced contrast mode enabled.'
    ]);

    this.responseTemplates.set('STOP', [
      'Stopping playback.',
      'Audio stopped.',
      'Playback has been stopped.',
      'I\'ve stopped speaking.'
    ]);

    this.responseTemplates.set('PAUSE', [
      'Pausing playback.',
      'Audio paused.',
      'Playback has been paused.',
      'I\'ve paused the audio.'
    ]);

    this.responseTemplates.set('RESUME', [
      'Resuming playback.',
      'Audio resumed.',
      'Continuing from where we left off.',
      'Playback has been resumed.'
    ]);
  }

  /**
   * Setup event handlers for voice services
   */
  private setupEventHandlers(): void {
    // TTS event handlers
    this.textToSpeech.on('start', () => {
      this.updateState({ isSpeaking: true });
      this.emit('speaking_started', {});
    });

    this.textToSpeech.on('end', () => {
      this.updateState({ isSpeaking: false });
      this.emit('speaking_ended', {});
    });

    this.textToSpeech.on('error', (error: any) => {
      this.handleError(`TTS Error: ${error.message || 'Unknown error'}`);
    });
  }

  /**
   * Handle speech recognition results
   */
  private async handleSpeechResult(results: Array<{transcript: string; confidence: number; isFinal: boolean}>): Promise<void> {
    if (results.length === 0) return;

    const result = results[0];
    
    this.updateState({ 
      isProcessing: true,
      confidence: result.confidence 
    });

    this.emit('transcript_received', {
      transcript: result.transcript,
      confidence: result.confidence,
      isFinal: result.isFinal
    });

    if (result.isFinal && result.confidence >= this.config.confidenceThreshold) {
      await this.processUserInput(result.transcript, result.confidence);
    }
  }

  /**
   * Process user input and generate response
   */
  private async processUserInput(transcript: string, confidence: number): Promise<void> {
    try {
      // Detect emotion if enabled
      let emotion: IEmotionResult | undefined;
      if (this.config.emotionDetection) {
        emotion = this.detectEmotion(transcript);
      }

      // Process command
      const commandResult = this.commandProcessor.processCommand(transcript, confidence);
      
      if (commandResult) {
        this.updateState({ 
          currentCommand: commandResult.command,
          isProcessing: true 
        });

        // Generate response
        const response = await this.generateResponse(commandResult, emotion);
        
        // Add to conversation history
        this.conversationHistory.push({
          timestamp: new Date(),
          userInput: transcript,
          response,
          emotion
        });

        // Execute command and provide feedback
        await this.executeCommand(commandResult);
        await this.speak(response.text);

        this.emit('command_executed', {
          command: commandResult,
          response,
          emotion
        });

      } else {
        // Handle unrecognized command
        const response = this.generateFallbackResponse(transcript, emotion);
        
        this.conversationHistory.push({
          timestamp: new Date(),
          userInput: transcript,
          response,
          emotion
        });

        await this.speak(response.text);

        this.emit('command_not_recognized', {
          transcript,
          confidence,
          response,
          emotion
        });
      }

    } catch (error) {
      this.handleError(`Error processing user input: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      this.updateState({ 
        isProcessing: false,
        currentCommand: undefined 
      });
    }
  }

  /**
   * Generate contextual response for recognized commands
   */
  private async generateResponse(
    commandResult: IVoiceCommandResult,
    emotion?: IEmotionResult
  ): Promise<IVoiceAssistantResponse> {
    const { command } = commandResult;
    const templates = this.responseTemplates.get(command.action) || ['I\'ll help you with that.'];
    
    // Select response based on context and emotion
    let responseText = this.selectResponseTemplate(templates, emotion);
    
    // Enhance response based on command parameters
    responseText = this.enhanceResponse(responseText, commandResult);

    // Generate suggestions
    const suggestions = this.generateSuggestions(command.action);

    // Generate follow-up if needed
    const followUp = this.generateFollowUp(command.action);

    return {
      text: responseText,
      action: command.action,
      parameters: commandResult.parameters,
      suggestions,
      followUp
    };
  }

  /**
   * Select appropriate response template
   */
  private selectResponseTemplate(templates: string[], emotion?: IEmotionResult): string {
    if (!emotion) {
      return templates[Math.floor(Math.random() * templates.length)];
    }

    // Adjust response based on detected emotion
    if (emotion.emotion === 'frustrated' || emotion.emotion === 'angry') {
      return templates[0]; // Use more direct response
    } else if (emotion.emotion === 'confused') {
      return `I understand you might be confused. ${templates[0]}`;
    } else {
      return templates[Math.floor(Math.random() * templates.length)];
    }
  }

  /**
   * Enhance response with command-specific information
   */
  private enhanceResponse(baseResponse: string, commandResult: IVoiceCommandResult): string {
    const { command, parameters } = commandResult;

    switch (command.action) {
      case 'NAVIGATE':
        if (parameters.target) {
          return `${baseResponse} Going to ${parameters.target}.`;
        }
        break;

      case 'SEARCH':
        if (parameters.query) {
          return `${baseResponse} Searching for "${parameters.query}".`;
        }
        break;

      case 'INCREASE_FONT_SIZE':
      case 'DECREASE_FONT_SIZE':
        if (parameters.amount) {
          return `${baseResponse} Adjusting by ${parameters.amount} steps.`;
        }
        break;

      default:
        break;
    }

    return baseResponse;
  }

  /**
   * Generate suggestions based on current action
   */
  private generateSuggestions(action: VoiceCommandAction): string[] {
    const suggestionMap: Record<VoiceCommandAction, string[]> = {
      'ANALYZE_DOCUMENT': ['Show results', 'Read summary', 'Export to PDF'],
      'SHOW_ARBITRATION_CLAUSES': ['Read summary', 'Compare versions', 'Export report'],
      'READ_SUMMARY': ['Show details', 'Export to PDF', 'Compare versions'],
      'NAVIGATE': ['Go to dashboard', 'Show history', 'Open settings'],
      'HELP': ['Analyze document', 'Show results', 'Change settings'],
      'SEARCH': ['Show results', 'Refine search', 'Clear search'],
      'COMPARE_VERSIONS': ['Show differences', 'Export comparison', 'Go back'],
      'EXPORT_PDF': ['Download report', 'Share results', 'Go to dashboard'],
      'TOGGLE_DARK_MODE': ['Increase font size', 'High contrast', 'Accessibility settings'],
      'INCREASE_FONT_SIZE': ['Decrease font size', 'High contrast', 'Voice commands'],
      'DECREASE_FONT_SIZE': ['Increase font size', 'High contrast', 'Reset settings'],
      'HIGH_CONTRAST': ['Change font size', 'Dark mode', 'Accessibility help'],
      'STOP': ['Resume', 'Repeat', 'Help'],
      'PAUSE': ['Resume', 'Stop', 'Help'],
      'RESUME': ['Pause', 'Stop', 'Repeat'],
      'REPEAT': ['Continue', 'Stop', 'Help'],
      'INCREASE_VOLUME': ['Decrease volume', 'Mute', 'Stop'],
      'DECREASE_VOLUME': ['Increase volume', 'Mute', 'Stop'],
      'MUTE': ['Unmute', 'Increase volume', 'Help'],
      'UNMUTE': ['Mute', 'Adjust volume', 'Help'],
      'FOCUS_NEXT': ['Focus previous', 'Navigate', 'Help'],
      'FOCUS_PREVIOUS': ['Focus next', 'Navigate', 'Help']
    };

    return suggestionMap[action] || ['Help', 'Stop', 'Repeat'];
  }

  /**
   * Generate follow-up question or prompt
   */
  private generateFollowUp(action: VoiceCommandAction): string | undefined {
    const followUpMap: Record<VoiceCommandAction, string> = {
      'ANALYZE_DOCUMENT': 'Would you like me to read the results or show them on screen?',
      'SHOW_ARBITRATION_CLAUSES': 'Would you like me to read the details or export a report?',
      'HELP': 'What specific task would you like help with?',
      'SEARCH': 'Would you like to refine your search or see more results?'
    };

    return followUpMap[action];
  }

  /**
   * Generate fallback response for unrecognized commands
   */
  private generateFallbackResponse(transcript: string, emotion?: IEmotionResult): IVoiceAssistantResponse {
    let responseText = 'I\'m sorry, I didn\'t understand that command.';

    if (emotion?.emotion === 'frustrated') {
      responseText = 'I can see you might be frustrated. Let me help you with what you need.';
    } else if (emotion?.emotion === 'confused') {
      responseText = 'It seems you might be unsure. Let me explain what I can do.';
    }

    const suggestions = [
      'Try saying "Help" to see available commands',
      'Say "Analyze document" to start',
      'Ask "What can you do?" for more options'
    ];

    return {
      text: responseText,
      suggestions,
      followUp: 'You can say things like "Analyze document", "Show results", or "Help me navigate".'
    };
  }

  /**
   * Execute the recognized command
   */
  private async executeCommand(commandResult: IVoiceCommandResult): Promise<void> {
    const { command, parameters } = commandResult;

    this.emit('command_executing', { command, parameters });

    try {
      switch (command.action) {
        case 'ANALYZE_DOCUMENT':
          await this.handleAnalyzeDocument(parameters);
          break;

        case 'SHOW_ARBITRATION_CLAUSES':
          await this.handleShowResults(parameters);
          break;

        case 'READ_SUMMARY':
          await this.handleReadSummary(parameters);
          break;

        case 'COMPARE_VERSIONS':
          await this.handleCompareVersions(parameters);
          break;

        case 'EXPORT_PDF':
          await this.handleExportPDF(parameters);
          break;

        case 'NAVIGATE':
          await this.handleNavigate(parameters);
          break;

        case 'SEARCH':
          await this.handleSearch(parameters);
          break;

        case 'HELP':
          await this.handleHelp(parameters);
          break;

        case 'TOGGLE_DARK_MODE':
          await this.handleToggleDarkMode();
          break;

        case 'INCREASE_FONT_SIZE':
          await this.handleChangeFontSize('increase', parameters);
          break;

        case 'DECREASE_FONT_SIZE':
          await this.handleChangeFontSize('decrease', parameters);
          break;

        case 'HIGH_CONTRAST':
          await this.handleHighContrast();
          break;

        case 'STOP':
          this.textToSpeech.stop();
          break;

        case 'PAUSE':
          this.textToSpeech.pause();
          break;

        case 'RESUME':
          this.textToSpeech.resume();
          break;

        case 'REPEAT':
          await this.textToSpeech.repeatLast();
          break;

        case 'INCREASE_VOLUME':
          this.textToSpeech.increaseVolume();
          break;

        case 'DECREASE_VOLUME':
          this.textToSpeech.decreaseVolume();
          break;

        case 'MUTE':
          this.textToSpeech.mute();
          break;

        case 'UNMUTE':
          this.textToSpeech.unmute();
          break;

        default:
          console.warn(`Unhandled command action: ${command.action}`);
      }

    } catch (error) {
      this.handleError(`Error executing command: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * Command handlers
   */
  private async handleAnalyzeDocument(parameters: Record<string, unknown>): Promise<void> {
    this.emit('analyze_document_requested', parameters);
  }

  private async handleShowResults(parameters: Record<string, unknown>): Promise<void> {
    this.emit('show_results_requested', parameters);
  }

  private async handleReadSummary(parameters: Record<string, unknown>): Promise<void> {
    this.emit('read_summary_requested', parameters);
  }

  private async handleCompareVersions(parameters: Record<string, unknown>): Promise<void> {
    this.emit('compare_versions_requested', parameters);
  }

  private async handleExportPDF(parameters: Record<string, unknown>): Promise<void> {
    this.emit('export_pdf_requested', parameters);
  }

  private async handleNavigate(parameters: Record<string, unknown>): Promise<void> {
    this.emit('navigate_requested', parameters);
  }

  private async handleSearch(parameters: Record<string, unknown>): Promise<void> {
    this.emit('search_requested', parameters);
  }

  private async handleHelp(parameters: Record<string, unknown>): Promise<void> {
    this.emit('help_requested', parameters);
  }

  private async handleToggleDarkMode(): Promise<void> {
    this.emit('toggle_dark_mode_requested', {});
  }

  private async handleChangeFontSize(action: 'increase' | 'decrease', parameters: Record<string, unknown>): Promise<void> {
    this.emit('change_font_size_requested', { action, amount: parameters.amount || 1 });
  }

  private async handleHighContrast(): Promise<void> {
    this.emit('high_contrast_requested', {});
  }

  /**
   * Basic emotion detection from text
   */
  private detectEmotion(text: string): IEmotionResult {
    const lowerText = text.toLowerCase();
    
    // Simple keyword-based emotion detection
    const emotionKeywords = {
      frustrated: ['frustrated', 'annoying', 'stupid', 'broken', 'wrong', 'bad'],
      angry: ['angry', 'mad', 'furious', 'hate', 'terrible'],
      confused: ['confused', 'don\'t understand', 'unclear', 'what', 'how'],
      happy: ['good', 'great', 'excellent', 'perfect', 'love', 'awesome'],
      sad: ['sad', 'sorry', 'disappointed', 'upset'],
      excited: ['excited', 'amazing', 'fantastic', 'wonderful']
    };

    for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
      for (const keyword of keywords) {
        if (lowerText.includes(keyword)) {
          return {
            emotion: emotion as any,
            confidence: 0.7,
            valence: emotion === 'happy' || emotion === 'excited' ? 0.8 : -0.5,
            arousal: emotion === 'angry' || emotion === 'frustrated' || emotion === 'excited' ? 0.8 : 0.3
          };
        }
      }
    }

    return {
      emotion: 'neutral',
      confidence: 0.5,
      valence: 0,
      arousal: 0.2
    };
  }

  /**
   * Speech recognition event handlers
   */
  private handleSpeechStart(): void {
    this.updateState({ isListening: true });
    this.emit('listening_started', {});
  }

  private handleSpeechEnd(): void {
    this.updateState({ isListening: false });
    this.emit('listening_stopped', {});
  }

  private handleSpeechError(error: any): void {
    this.handleError(`Speech recognition error: ${error.error || 'Unknown error'}`);
  }

  /**
   * Speak text using TTS
   */
  private async speak(text: string): Promise<void> {
    if (!text.trim()) return;

    try {
      await this.textToSpeech.speak(text, this.config.speechSynthesis);
    } catch (error) {
      this.handleError(`Error speaking text: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Public methods
   */

  /**
   * Start listening for voice commands
   */
  public async startListening(): Promise<void> {
    try {
      await this.speechRecognition.start();
    } catch (error) {
      this.handleError(`Failed to start listening: ${error instanceof Error ? error.message : 'Unknown error'}`);
      throw error;
    }
  }

  /**
   * Stop listening
   */
  public stopListening(): void {
    this.speechRecognition.stop();
  }

  /**
   * Update configuration
   */
  public updateConfig(config: Partial<IVoiceAssistantConfig>): void {
    this.config = { ...this.config, ...config };
    
    // Update services with new config
    if (config.speechRecognition) {
      this.speechRecognition.updateConfig(config.speechRecognition);
    }
    
    if (config.speechSynthesis) {
      this.textToSpeech.updateOptions(config.speechSynthesis);
    }
    
    if (config.confidenceThreshold !== undefined) {
      this.commandProcessor.updateConfidenceThreshold(config.confidenceThreshold);
    }

    this.emit('config_updated', { config: this.config });
  }

  /**
   * Update context
   */
  public updateContext(context: Partial<IVoiceCommandContext>): void {
    this.context = { ...this.context, ...context };
    this.commandProcessor.updateContext(context);
    this.emit('context_updated', { context: this.context });
  }

  /**
   * Get current state
   */
  public getState(): IVoiceAssistantState {
    return { ...this.state };
  }

  /**
   * Get conversation history
   */
  public getConversationHistory(): typeof this.conversationHistory {
    return [...this.conversationHistory];
  }

  /**
   * Clear conversation history
   */
  public clearConversationHistory(): void {
    this.conversationHistory = [];
    this.emit('conversation_history_cleared', {});
  }

  /**
   * Update internal state
   */
  private updateState(updates: Partial<IVoiceAssistantState>): void {
    this.state = { ...this.state, ...updates };
    this.emit('state_changed', { state: this.state });
  }

  /**
   * Handle errors
   */
  private handleError(message: string): void {
    this.updateState({ error: message, isProcessing: false });
    this.emit('error', { message });
    console.error('Voice Assistant Error:', message);
  }

  /**
   * Event system
   */
  public on(event: string, handler: (data: any) => void): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  public off(event: string, handler: (data: any) => void): void {
    this.eventHandlers.get(event)?.delete(handler);
  }

  private emit(event: string, data: any): void {
    this.eventHandlers.get(event)?.forEach(handler => {
      try {
        handler(data);
      } catch (error) {
        console.error('Error in voice assistant event handler:', error);
      }
    });
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.speechRecognition.destroy();
    this.textToSpeech.destroy();
    this.eventHandlers.clear();
    this.conversationHistory = [];
  }
}