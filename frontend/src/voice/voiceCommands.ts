/**
 * Voice Command Processing System
 * Advanced natural language understanding for voice commands with intent recognition,
 * parameter extraction, and context-aware command execution
 */

import {
  IVoiceCommand,
  IVoiceCommandResult,
  IVoiceCommandContext,
  INLUResult,
  INLUIntent,
  INLUEntity,
  VoiceCommandAction,
  VoiceCommandCategory
} from '@/types/voice';

export class VoiceCommandProcessor {
  private commands: Map<string, IVoiceCommand> = new Map();
  private context: IVoiceCommandContext;
  private confidenceThreshold: number = 0.7;
  private aliases: Map<string, string[]> = new Map();
  private stopWords: Set<string> = new Set();

  constructor(context: IVoiceCommandContext, confidenceThreshold: number = 0.7) {
    this.context = context;
    this.confidenceThreshold = confidenceThreshold;
    this.initializeStopWords();
    this.loadDefaultCommands();
    this.setupAliases();
  }

  /**
   * Initialize common stop words for better command parsing
   */
  private initializeStopWords(): void {
    this.stopWords = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
      'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
      'before', 'after', 'above', 'below', 'between', 'among', 'can', 'could',
      'would', 'should', 'will', 'shall', 'may', 'might', 'must', 'ought',
      'please', 'now', 'then', 'here', 'there', 'where', 'when', 'why', 'how'
    ]);
  }

  /**
   * Load default voice commands for arbitration analysis system
   */
  private loadDefaultCommands(): void {
    const defaultCommands: IVoiceCommand[] = [
      // Document Analysis Commands
      {
        id: 'analyze-document',
        phrases: [
          'analyze this document',
          'start analysis',
          'check for arbitration clauses',
          'scan document',
          'begin document review',
          'analyze the file',
          'check this contract'
        ],
        action: 'ANALYZE_DOCUMENT',
        description: 'Analyze the current document for arbitration clauses',
        category: 'ANALYSIS',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'show-arbitration-clauses',
        phrases: [
          'show arbitration clauses',
          'display results',
          'show findings',
          'what did you find',
          'show analysis results',
          'view arbitration sections',
          'highlight clauses'
        ],
        action: 'SHOW_ARBITRATION_CLAUSES',
        description: 'Display found arbitration clauses and analysis results',
        category: 'ANALYSIS',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'read-summary',
        phrases: [
          'read the summary',
          'read results',
          'tell me what you found',
          'speak the analysis',
          'read analysis aloud',
          'voice the results',
          'summarize findings'
        ],
        action: 'READ_SUMMARY',
        description: 'Read the analysis summary using text-to-speech',
        category: 'ANALYSIS',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'compare-versions',
        phrases: [
          'compare with previous version',
          'show version differences',
          'compare documents',
          'what changed',
          'version comparison',
          'diff analysis',
          'compare revisions'
        ],
        action: 'COMPARE_VERSIONS',
        description: 'Compare current document with previous versions',
        category: 'ANALYSIS',
        confidence: 0.85,
        enabled: true
      },
      {
        id: 'export-pdf',
        phrases: [
          'export to PDF',
          'generate report',
          'create PDF report',
          'download results',
          'save analysis',
          'export findings',
          'generate document'
        ],
        action: 'EXPORT_PDF',
        description: 'Export analysis results to PDF format',
        category: 'ANALYSIS',
        confidence: 0.8,
        enabled: true
      },

      // Navigation Commands
      {
        id: 'go-to-dashboard',
        phrases: [
          'go to dashboard',
          'open dashboard',
          'show main page',
          'home page',
          'go home',
          'main menu'
        ],
        action: 'NAVIGATE',
        description: 'Navigate to the main dashboard',
        category: 'NAVIGATION',
        parameters: { target: 'dashboard' },
        confidence: 0.85,
        enabled: true
      },
      {
        id: 'go-to-history',
        phrases: [
          'show history',
          'view past analyses',
          'open history',
          'previous documents',
          'analysis history',
          'past results'
        ],
        action: 'NAVIGATE',
        description: 'Navigate to analysis history',
        category: 'NAVIGATION',
        parameters: { target: 'history' },
        confidence: 0.85,
        enabled: true
      },
      {
        id: 'search',
        phrases: [
          'search for',
          'find',
          'look for',
          'locate',
          'search documents'
        ],
        action: 'SEARCH',
        description: 'Search documents or results',
        category: 'NAVIGATION',
        confidence: 0.8,
        enabled: true
      },

      // Accessibility Commands
      {
        id: 'toggle-dark-mode',
        phrases: [
          'toggle dark mode',
          'switch to dark theme',
          'enable dark mode',
          'turn on dark theme',
          'change to dark mode'
        ],
        action: 'TOGGLE_DARK_MODE',
        description: 'Toggle between light and dark themes',
        category: 'ACCESSIBILITY',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'increase-font-size',
        phrases: [
          'increase font size',
          'make text larger',
          'bigger font',
          'zoom in text',
          'larger text'
        ],
        action: 'INCREASE_FONT_SIZE',
        description: 'Increase the font size for better readability',
        category: 'ACCESSIBILITY',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'decrease-font-size',
        phrases: [
          'decrease font size',
          'make text smaller',
          'smaller font',
          'zoom out text',
          'reduce text size'
        ],
        action: 'DECREASE_FONT_SIZE',
        description: 'Decrease the font size',
        category: 'ACCESSIBILITY',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'high-contrast',
        phrases: [
          'enable high contrast',
          'high contrast mode',
          'better contrast',
          'increase contrast',
          'accessibility mode'
        ],
        action: 'HIGH_CONTRAST',
        description: 'Enable high contrast mode for better visibility',
        category: 'ACCESSIBILITY',
        confidence: 0.9,
        enabled: true
      },

      // Playback Commands
      {
        id: 'stop',
        phrases: [
          'stop',
          'stop reading',
          'quiet',
          'silence',
          'stop talking',
          'pause reading'
        ],
        action: 'STOP',
        description: 'Stop current text-to-speech playback',
        category: 'PLAYBACK',
        confidence: 0.95,
        enabled: true
      },
      {
        id: 'pause',
        phrases: [
          'pause',
          'pause reading',
          'hold on',
          'wait'
        ],
        action: 'PAUSE',
        description: 'Pause current text-to-speech playback',
        category: 'PLAYBACK',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'resume',
        phrases: [
          'resume',
          'continue',
          'keep reading',
          'go on',
          'continue reading'
        ],
        action: 'RESUME',
        description: 'Resume paused text-to-speech playback',
        category: 'PLAYBACK',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'repeat',
        phrases: [
          'repeat',
          'say that again',
          'repeat last',
          'what did you say',
          'read again'
        ],
        action: 'REPEAT',
        description: 'Repeat the last spoken content',
        category: 'PLAYBACK',
        confidence: 0.9,
        enabled: true
      },

      // Volume Commands
      {
        id: 'increase-volume',
        phrases: [
          'louder',
          'increase volume',
          'turn up volume',
          'speak louder',
          'volume up'
        ],
        action: 'INCREASE_VOLUME',
        description: 'Increase text-to-speech volume',
        category: 'PLAYBACK',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'decrease-volume',
        phrases: [
          'quieter',
          'decrease volume',
          'turn down volume',
          'speak quieter',
          'volume down'
        ],
        action: 'DECREASE_VOLUME',
        description: 'Decrease text-to-speech volume',
        category: 'PLAYBACK',
        confidence: 0.9,
        enabled: true
      },
      {
        id: 'mute',
        phrases: [
          'mute',
          'turn off sound',
          'silence audio',
          'mute volume'
        ],
        action: 'MUTE',
        description: 'Mute all audio output',
        category: 'PLAYBACK',
        confidence: 0.95,
        enabled: true
      },

      // Help Commands
      {
        id: 'help',
        phrases: [
          'help',
          'what can you do',
          'show commands',
          'voice commands',
          'available commands',
          'how do I',
          'what can I say'
        ],
        action: 'HELP',
        description: 'Show available voice commands and help information',
        category: 'HELP',
        confidence: 0.9,
        enabled: true
      }
    ];

    // Add commands to the map
    defaultCommands.forEach(command => {
      this.commands.set(command.id, command);
    });
  }

  /**
   * Setup command aliases for better recognition
   */
  private setupAliases(): void {
    this.aliases.set('analyze', ['check', 'scan', 'review', 'examine']);
    this.aliases.set('show', ['display', 'view', 'reveal', 'present']);
    this.aliases.set('read', ['speak', 'voice', 'say', 'tell']);
    this.aliases.set('document', ['file', 'contract', 'paper', 'text']);
    this.aliases.set('arbitration', ['dispute resolution', 'mediation', 'settlement']);
    this.aliases.set('clause', ['section', 'provision', 'term', 'paragraph']);
    this.aliases.set('summary', ['overview', 'results', 'findings']);
    this.aliases.set('export', ['save', 'download', 'generate']);
    this.aliases.set('compare', ['diff', 'difference', 'contrast']);
    this.aliases.set('navigate', ['go to', 'open', 'switch to']);
    this.aliases.set('increase', ['bigger', 'larger', 'more', 'up']);
    this.aliases.set('decrease', ['smaller', 'less', 'down', 'reduce']);
    this.aliases.set('enable', ['turn on', 'activate', 'start']);
    this.aliases.set('disable', ['turn off', 'deactivate', 'stop']);
  }

  /**
   * Process voice input and extract commands
   */
  public processCommand(transcript: string, confidence: number): IVoiceCommandResult | null {
    if (confidence < this.confidenceThreshold) {
      return null;
    }

    // Preprocess the transcript
    const processedText = this.preprocessText(transcript);
    
    // Try to match against known commands
    const matchedCommand = this.matchCommand(processedText);
    
    if (!matchedCommand) {
      // Try NLU approach for complex commands
      return this.processWithNLU(processedText, confidence);
    }

    // Extract parameters if any
    const parameters = this.extractParameters(processedText, matchedCommand);

    return {
      command: matchedCommand,
      confidence: confidence * matchedCommand.confidence,
      parameters,
      transcript: processedText,
      timestamp: new Date()
    };
  }

  /**
   * Preprocess text for better matching
   */
  private preprocessText(text: string): string {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s]/g, ' ') // Remove punctuation
      .replace(/\s+/g, ' ') // Normalize whitespace
      .trim();
  }

  /**
   * Match transcript against known commands
   */
  private matchCommand(text: string): IVoiceCommand | null {
    let bestMatch: IVoiceCommand | null = null;
    let bestScore = 0;

    for (const command of this.commands.values()) {
      if (!command.enabled) continue;

      for (const phrase of command.phrases) {
        const score = this.calculateSimilarity(text, phrase.toLowerCase());
        if (score > bestScore && score >= this.confidenceThreshold) {
          bestScore = score;
          bestMatch = command;
        }
      }
    }

    return bestMatch;
  }

  /**
   * Calculate similarity between two strings using various techniques
   */
  private calculateSimilarity(text1: string, text2: string): number {
    // Exact match
    if (text1 === text2) return 1.0;

    // Contains check
    if (text1.includes(text2) || text2.includes(text1)) return 0.9;

    // Word-based similarity
    const words1 = this.getSignificantWords(text1);
    const words2 = this.getSignificantWords(text2);

    if (words1.length === 0 || words2.length === 0) return 0;

    const intersection = words1.filter(word => words2.includes(word));
    const union = [...new Set([...words1, ...words2])];

    const jaccardSimilarity = intersection.length / union.length;

    // Check for aliases
    const aliasScore = this.calculateAliasScore(words1, words2);

    // Combine scores
    return Math.max(jaccardSimilarity, aliasScore);
  }

  /**
   * Get significant words (excluding stop words)
   */
  private getSignificantWords(text: string): string[] {
    return text.split(' ')
      .filter(word => word.length > 0 && !this.stopWords.has(word));
  }

  /**
   * Calculate similarity score based on aliases
   */
  private calculateAliasScore(words1: string[], words2: string[]): number {
    let matches = 0;
    let total = Math.max(words1.length, words2.length);

    for (const word1 of words1) {
      for (const word2 of words2) {
        if (word1 === word2) {
          matches++;
          continue;
        }

        // Check aliases
        const aliases1 = this.aliases.get(word1) || [];
        const aliases2 = this.aliases.get(word2) || [];

        if (aliases1.includes(word2) || aliases2.includes(word1)) {
          matches += 0.8; // Alias matches are weighted lower
        }
      }
    }

    return Math.min(matches / total, 1.0);
  }

  /**
   * Extract parameters from the command text
   */
  private extractParameters(text: string, command: IVoiceCommand): Record<string, unknown> {
    const parameters: Record<string, unknown> = { ...command.parameters };

    // Extract common parameters based on command action
    switch (command.action) {
      case 'SEARCH':
        const searchQuery = this.extractSearchQuery(text);
        if (searchQuery) {
          parameters.query = searchQuery;
        }
        break;

      case 'NAVIGATE':
        const target = this.extractNavigationTarget(text);
        if (target) {
          parameters.target = target;
        }
        break;

      case 'INCREASE_FONT_SIZE':
      case 'DECREASE_FONT_SIZE':
        const amount = this.extractAmount(text);
        if (amount) {
          parameters.amount = amount;
        }
        break;

      default:
        break;
    }

    return parameters;
  }

  /**
   * Extract search query from text
   */
  private extractSearchQuery(text: string): string | null {
    const searchPhrases = ['search for', 'find', 'look for', 'locate'];
    
    for (const phrase of searchPhrases) {
      const index = text.indexOf(phrase);
      if (index !== -1) {
        const query = text.substring(index + phrase.length).trim();
        return query.length > 0 ? query : null;
      }
    }

    return null;
  }

  /**
   * Extract navigation target from text
   */
  private extractNavigationTarget(text: string): string | null {
    const targets = ['dashboard', 'history', 'settings', 'help', 'profile'];
    
    for (const target of targets) {
      if (text.includes(target)) {
        return target;
      }
    }

    return null;
  }

  /**
   * Extract amount/quantity from text
   */
  private extractAmount(text: string): number | null {
    const numbers = text.match(/\b(\d+)\b/);
    if (numbers) {
      return parseInt(numbers[1], 10);
    }

    // Check for relative amounts
    if (text.includes('little') || text.includes('bit')) {
      return 1;
    }
    if (text.includes('lot') || text.includes('much')) {
      return 5;
    }

    return null;
  }

  /**
   * Process command using Natural Language Understanding
   */
  private processWithNLU(text: string, confidence: number): IVoiceCommandResult | null {
    const nluResult = this.performNLU(text);
    
    if (!nluResult || nluResult.confidence < this.confidenceThreshold) {
      return null;
    }

    // Try to find a matching command based on intent
    const command = this.findCommandByIntent(nluResult.intent);
    
    if (!command) {
      return null;
    }

    // Extract parameters from entities
    const parameters = this.extractParametersFromEntities(nluResult.intent.entities);

    return {
      command,
      confidence: confidence * nluResult.confidence,
      parameters,
      transcript: text,
      timestamp: new Date()
    };
  }

  /**
   * Perform basic Natural Language Understanding
   */
  private performNLU(text: string): INLUResult | null {
    // This is a simplified NLU implementation
    // In production, you would use a service like DialogFlow, LUIS, or Rasa
    
    const intents = this.classifyIntent(text);
    const entities = this.extractEntities(text);

    if (intents.length === 0) {
      return null;
    }

    const topIntent = intents[0];
    topIntent.entities = entities;

    return {
      intent: topIntent,
      text,
      confidence: topIntent.confidence
    };
  }

  /**
   * Classify intent from text
   */
  private classifyIntent(text: string): INLUIntent[] {
    const intents: INLUIntent[] = [];

    // Analysis intents
    if (this.containsKeywords(text, ['analyze', 'check', 'scan', 'review'])) {
      intents.push({
        name: 'analyze_document',
        confidence: 0.8,
        entities: []
      });
    }

    // Navigation intents
    if (this.containsKeywords(text, ['go', 'navigate', 'open', 'show'])) {
      intents.push({
        name: 'navigate',
        confidence: 0.7,
        entities: []
      });
    }

    // Playback intents
    if (this.containsKeywords(text, ['read', 'speak', 'say', 'tell'])) {
      intents.push({
        name: 'text_to_speech',
        confidence: 0.8,
        entities: []
      });
    }

    // Control intents
    if (this.containsKeywords(text, ['stop', 'pause', 'resume', 'mute'])) {
      intents.push({
        name: 'playback_control',
        confidence: 0.9,
        entities: []
      });
    }

    return intents.sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Check if text contains any of the keywords
   */
  private containsKeywords(text: string, keywords: string[]): boolean {
    return keywords.some(keyword => text.includes(keyword));
  }

  /**
   * Extract entities from text
   */
  private extractEntities(text: string): INLUEntity[] {
    const entities: INLUEntity[] = [];

    // Document types
    const documentTypes = ['contract', 'agreement', 'document', 'file', 'paper'];
    for (const type of documentTypes) {
      const index = text.indexOf(type);
      if (index !== -1) {
        entities.push({
          entity: 'document_type',
          value: type,
          confidence: 0.9,
          start: index,
          end: index + type.length
        });
      }
    }

    // Actions
    const actions = ['analyze', 'read', 'show', 'export', 'compare'];
    for (const action of actions) {
      const index = text.indexOf(action);
      if (index !== -1) {
        entities.push({
          entity: 'action',
          value: action,
          confidence: 0.8,
          start: index,
          end: index + action.length
        });
      }
    }

    return entities;
  }

  /**
   * Find command by intent name
   */
  private findCommandByIntent(intent: INLUIntent): IVoiceCommand | null {
    const intentToActionMap: Record<string, VoiceCommandAction> = {
      'analyze_document': 'ANALYZE_DOCUMENT',
      'navigate': 'NAVIGATE',
      'text_to_speech': 'READ_SUMMARY',
      'playback_control': 'STOP'
    };

    const action = intentToActionMap[intent.name];
    if (!action) return null;

    // Find the first enabled command with this action
    for (const command of this.commands.values()) {
      if (command.enabled && command.action === action) {
        return command;
      }
    }

    return null;
  }

  /**
   * Extract parameters from NLU entities
   */
  private extractParametersFromEntities(entities: INLUEntity[]): Record<string, unknown> {
    const parameters: Record<string, unknown> = {};

    for (const entity of entities) {
      parameters[entity.entity] = entity.value;
    }

    return parameters;
  }

  /**
   * Update context
   */
  public updateContext(context: Partial<IVoiceCommandContext>): void {
    this.context = { ...this.context, ...context };
  }

  /**
   * Add custom command
   */
  public addCommand(command: IVoiceCommand): void {
    this.commands.set(command.id, command);
  }

  /**
   * Remove command
   */
  public removeCommand(commandId: string): void {
    this.commands.delete(commandId);
  }

  /**
   * Enable/disable command
   */
  public toggleCommand(commandId: string, enabled: boolean): void {
    const command = this.commands.get(commandId);
    if (command) {
      command.enabled = enabled;
    }
  }

  /**
   * Get all commands by category
   */
  public getCommandsByCategory(category: VoiceCommandCategory): IVoiceCommand[] {
    return Array.from(this.commands.values())
      .filter(command => command.category === category && command.enabled);
  }

  /**
   * Get all available commands
   */
  public getAllCommands(): IVoiceCommand[] {
    return Array.from(this.commands.values())
      .filter(command => command.enabled);
  }

  /**
   * Update confidence threshold
   */
  public updateConfidenceThreshold(threshold: number): void {
    this.confidenceThreshold = Math.max(0.1, Math.min(1.0, threshold));
  }

  /**
   * Get command statistics
   */
  public getStatistics(): {
    totalCommands: number;
    enabledCommands: number;
    commandsByCategory: Record<VoiceCommandCategory, number>;
  } {
    const totalCommands = this.commands.size;
    const enabledCommands = Array.from(this.commands.values())
      .filter(command => command.enabled).length;

    const commandsByCategory: Record<VoiceCommandCategory, number> = {
      'ANALYSIS': 0,
      'NAVIGATION': 0,
      'ACCESSIBILITY': 0,
      'PLAYBACK': 0,
      'SYSTEM': 0,
      'HELP': 0
    };

    for (const command of this.commands.values()) {
      if (command.enabled) {
        commandsByCategory[command.category]++;
      }
    }

    return {
      totalCommands,
      enabledCommands,
      commandsByCategory
    };
  }
}