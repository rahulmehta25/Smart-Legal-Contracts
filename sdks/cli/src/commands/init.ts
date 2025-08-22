/**
 * Initialize command for creating new projects with Arbitration SDK
 */

import { join, resolve } from 'path';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import chalk from 'chalk';
import inquirer from 'inquirer';
import ora from 'ora';
import { exec } from 'child_process';
import { promisify } from 'util';

import { 
  ProjectTemplate, 
  Platform, 
  validateProjectName,
  generateProjectFiles,
  installDependencies,
  initializeGitRepository,
  displaySuccessMessage 
} from '../utils';

const execAsync = promisify(exec);

interface InitOptions {
  platform: Platform;
  template?: string;
  git: boolean;
  install: boolean;
}

export async function initCommand(
  projectName?: string,
  options: InitOptions = {
    platform: 'react-native',
    git: false,
    install: false
  }
): Promise<void> {
  console.log(chalk.cyan.bold('\n🚀 Initializing Arbitration SDK Project\n'));

  try {
    // Get project name if not provided
    if (!projectName) {
      const { name } = await inquirer.prompt([
        {
          type: 'input',
          name: 'name',
          message: 'What is your project name?',
          validate: (input: string) => {
            const validation = validateProjectName(input);
            return validation.isValid || validation.errors[0];
          },
        },
      ]);
      projectName = name;
    }

    // Validate project name
    const nameValidation = validateProjectName(projectName);
    if (!nameValidation.isValid) {
      console.error(chalk.red('❌ Invalid project name:'));
      nameValidation.errors.forEach(error => console.error(chalk.red(`   ${error}`)));
      process.exit(1);
    }

    // Check if directory already exists
    const projectPath = resolve(projectName);
    if (existsSync(projectPath)) {
      const { overwrite } = await inquirer.prompt([
        {
          type: 'confirm',
          name: 'overwrite',
          message: `Directory ${projectName} already exists. Overwrite?`,
          default: false,
        },
      ]);

      if (!overwrite) {
        console.log(chalk.yellow('Operation cancelled.'));
        return;
      }
    }

    // Get platform if not specified
    let { platform } = options;
    if (!platform) {
      const { selectedPlatform } = await inquirer.prompt([
        {
          type: 'list',
          name: 'selectedPlatform',
          message: 'Which platform would you like to target?',
          choices: [
            { name: '📱 React Native (Cross-platform)', value: 'react-native' },
            { name: '🐦 Flutter (Cross-platform)', value: 'flutter' },
            { name: '🍎 iOS (Native)', value: 'ios' },
            { name: '🤖 Android (Native)', value: 'android' },
          ],
        },
      ]);
      platform = selectedPlatform;
    }

    // Get template
    const templates = getAvailableTemplates(platform);
    let template = options.template;
    
    if (!template && templates.length > 1) {
      const { selectedTemplate } = await inquirer.prompt([
        {
          type: 'list',
          name: 'selectedTemplate',
          message: 'Choose a project template:',
          choices: templates.map(t => ({
            name: `${t.name} - ${t.description}`,
            value: t.id,
          })),
        },
      ]);
      template = selectedTemplate;
    } else if (!template) {
      template = templates[0]?.id || 'basic';
    }

    // Additional project configuration
    const config = await getProjectConfiguration(platform);

    // Create project directory
    const spinner = ora('Creating project directory...').start();
    
    if (!existsSync(projectPath)) {
      mkdirSync(projectPath, { recursive: true });
    }

    spinner.succeed('Project directory created');

    // Generate project files
    spinner.start('Generating project files...');
    await generateProjectFiles({
      projectName,
      projectPath,
      platform,
      template: template || 'basic',
      config,
    });
    spinner.succeed('Project files generated');

    // Initialize git repository
    if (options.git || config.initGit) {
      spinner.start('Initializing git repository...');
      try {
        await initializeGitRepository(projectPath);
        spinner.succeed('Git repository initialized');
      } catch (error) {
        spinner.warn('Failed to initialize git repository');
      }
    }

    // Install dependencies
    if (options.install || config.installDependencies) {
      spinner.start('Installing dependencies...');
      try {
        await installDependencies(projectPath, platform);
        spinner.succeed('Dependencies installed');
      } catch (error) {
        spinner.warn('Failed to install dependencies');
        console.log(chalk.yellow('You can install them manually later with:'));
        console.log(chalk.cyan(getInstallCommand(platform)));
      }
    }

    // Platform-specific post-setup
    await performPlatformSpecificSetup(projectPath, platform, spinner);

    // Success message
    displaySuccessMessage({
      projectName,
      projectPath,
      platform,
      nextSteps: getNextSteps(projectName, platform, !options.install && !config.installDependencies),
    });

  } catch (error) {
    console.error(chalk.red('❌ Failed to initialize project:'));
    console.error(chalk.red(error instanceof Error ? error.message : String(error)));
    process.exit(1);
  }
}

function getAvailableTemplates(platform: Platform): ProjectTemplate[] {
  const templates: Record<Platform, ProjectTemplate[]> = {
    'react-native': [
      {
        id: 'basic',
        name: 'Basic App',
        description: 'Simple document analyzer app',
        features: ['Document upload', 'Text analysis', 'Results display'],
      },
      {
        id: 'advanced',
        name: 'Advanced App',
        description: 'Full-featured app with camera, offline mode',
        features: ['Camera scanning', 'Offline mode', 'Analytics', 'Biometric auth'],
      },
      {
        id: 'minimal',
        name: 'Minimal Integration',
        description: 'Just the SDK without UI components',
        features: ['Core SDK', 'TypeScript support'],
      },
    ],
    flutter: [
      {
        id: 'basic',
        name: 'Basic App',
        description: 'Material Design document analyzer',
        features: ['Material UI', 'Document analysis', 'State management'],
      },
      {
        id: 'cupertino',
        name: 'Cupertino App',
        description: 'iOS-style document analyzer',
        features: ['Cupertino UI', 'iOS styling', 'Adaptive design'],
      },
      {
        id: 'package',
        name: 'Flutter Package',
        description: 'Reusable Flutter package',
        features: ['Package structure', 'Example app', 'Documentation'],
      },
    ],
    ios: [
      {
        id: 'swiftui',
        name: 'SwiftUI App',
        description: 'Modern SwiftUI interface',
        features: ['SwiftUI', 'Combine', 'iOS 15+'],
      },
      {
        id: 'uikit',
        name: 'UIKit App',
        description: 'Traditional UIKit interface',
        features: ['UIKit', 'Storyboards', 'iOS 13+'],
      },
    ],
    android: [
      {
        id: 'compose',
        name: 'Jetpack Compose App',
        description: 'Modern Compose UI',
        features: ['Jetpack Compose', 'Material 3', 'Android 6+'],
      },
      {
        id: 'views',
        name: 'Traditional Views',
        description: 'XML layouts and fragments',
        features: ['XML layouts', 'Fragments', 'Android 6+'],
      },
    ],
  };

  return templates[platform] || [];
}

async function getProjectConfiguration(platform: Platform) {
  const questions = [
    {
      type: 'input',
      name: 'bundleId',
      message: 'Bundle identifier (e.g., com.company.app):',
      default: 'com.example.arbitrationapp',
      validate: (input: string) => {
        const bundleIdRegex = /^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/;
        return bundleIdRegex.test(input) || 'Please enter a valid bundle identifier';
      },
      when: () => platform === 'ios' || platform === 'android' || platform === 'react-native' || platform === 'flutter',
    },
    {
      type: 'input',
      name: 'displayName',
      message: 'Display name for the app:',
      default: 'Arbitration Analyzer',
    },
    {
      type: 'confirm',
      name: 'initGit',
      message: 'Initialize git repository?',
      default: true,
    },
    {
      type: 'confirm',
      name: 'installDependencies',
      message: 'Install dependencies now?',
      default: true,
    },
    {
      type: 'confirm',
      name: 'enableAnalytics',
      message: 'Enable analytics in the SDK?',
      default: true,
    },
    {
      type: 'confirm',
      name: 'enableOfflineMode',
      message: 'Enable offline mode?',
      default: true,
    },
  ];

  // Platform-specific questions
  if (platform === 'react-native') {
    questions.push(
      {
        type: 'list',
        name: 'navigation',
        message: 'Navigation library:',
        choices: [
          { name: 'React Navigation 6', value: 'react-navigation' },
          { name: 'None', value: 'none' },
        ],
        default: 'react-navigation',
      },
      {
        type: 'list',
        name: 'stateManagement',
        message: 'State management:',
        choices: [
          { name: 'Redux Toolkit', value: 'redux-toolkit' },
          { name: 'Context API', value: 'context' },
          { name: 'None', value: 'none' },
        ],
        default: 'redux-toolkit',
      }
    );
  }

  if (platform === 'flutter') {
    questions.push(
      {
        type: 'list',
        name: 'stateManagement',
        message: 'State management:',
        choices: [
          { name: 'Riverpod', value: 'riverpod' },
          { name: 'Provider', value: 'provider' },
          { name: 'Bloc', value: 'bloc' },
          { name: 'None', value: 'none' },
        ],
        default: 'riverpod',
      }
    );
  }

  return inquirer.prompt(questions);
}

async function performPlatformSpecificSetup(
  projectPath: string,
  platform: Platform,
  spinner: ora.Ora
) {
  try {
    switch (platform) {
      case 'ios':
        spinner.start('Setting up iOS project...');
        await setupIOSProject(projectPath);
        spinner.succeed('iOS project setup complete');
        break;

      case 'android':
        spinner.start('Setting up Android project...');
        await setupAndroidProject(projectPath);
        spinner.succeed('Android project setup complete');
        break;

      case 'react-native':
        spinner.start('Setting up React Native project...');
        await setupReactNativeProject(projectPath);
        spinner.succeed('React Native project setup complete');
        break;

      case 'flutter':
        spinner.start('Setting up Flutter project...');
        await setupFlutterProject(projectPath);
        spinner.succeed('Flutter project setup complete');
        break;
    }
  } catch (error) {
    spinner.warn(`Platform setup completed with warnings`);
    console.log(chalk.yellow('Some setup steps may need to be completed manually.'));
  }
}

async function setupIOSProject(projectPath: string) {
  // Check for Xcode
  try {
    await execAsync('xcode-select -p');
    // Setup CocoaPods if needed
    await execAsync('cd ' + projectPath + ' && pod install');
  } catch (error) {
    console.log(chalk.yellow('Xcode or CocoaPods not found. Manual setup may be required.'));
  }
}

async function setupAndroidProject(projectPath: string) {
  // Check for Android SDK
  if (!process.env.ANDROID_HOME && !process.env.ANDROID_SDK_ROOT) {
    console.log(chalk.yellow('Android SDK not found. Set ANDROID_HOME environment variable.'));
  }
}

async function setupReactNativeProject(projectPath: string) {
  // Run React Native specific setup
  try {
    await execAsync('cd ' + projectPath + ' && npx react-native doctor');
  } catch (error) {
    // Doctor command might fail, but it's not critical
  }
}

async function setupFlutterProject(projectPath: string) {
  // Run Flutter specific setup
  try {
    await execAsync('cd ' + projectPath + ' && flutter doctor');
    await execAsync('cd ' + projectPath + ' && flutter pub get');
  } catch (error) {
    console.log(chalk.yellow('Flutter setup incomplete. Run flutter doctor for details.'));
  }
}

function getInstallCommand(platform: Platform): string {
  switch (platform) {
    case 'react-native':
      return 'npm install';
    case 'flutter':
      return 'flutter pub get';
    case 'ios':
      return 'pod install';
    case 'android':
      return './gradlew build';
    default:
      return 'npm install';
  }
}

function getNextSteps(projectName: string, platform: Platform, needsInstall: boolean): string[] {
  const steps = [
    `cd ${projectName}`,
  ];

  if (needsInstall) {
    steps.push(getInstallCommand(platform));
  }

  switch (platform) {
    case 'react-native':
      steps.push(
        'npx react-native start',
        'npx react-native run-ios  # or run-android'
      );
      break;
    case 'flutter':
      steps.push('flutter run');
      break;
    case 'ios':
      steps.push('open ios/YourApp.xcworkspace');
      break;
    case 'android':
      steps.push('open android/ in Android Studio');
      break;
  }

  steps.push('arbitration doctor  # Validate setup');

  return steps;
}