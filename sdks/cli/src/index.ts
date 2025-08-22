#!/usr/bin/env node

/**
 * Arbitration Platform CLI
 * 
 * Command-line interface for managing Arbitration Platform SDKs across
 * iOS, Android, React Native, and Flutter platforms.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import figlet from 'figlet';
import boxen from 'boxen';
import updateNotifier from 'update-notifier';
import { readFileSync } from 'fs';
import { join } from 'path';

// Commands
import { initCommand } from './commands/init';
import { installCommand } from './commands/install';
import { updateCommand } from './commands/update';
import { configCommand } from './commands/config';
import { analyzeCommand } from './commands/analyze';
import { testCommand } from './commands/test';
import { docsCommand } from './commands/docs';
import { migrateCommand } from './commands/migrate';
import { validateCommand } from './commands/validate';
import { buildCommand } from './commands/build';

// Utils
import { checkForUpdates, displayBanner, setupErrorHandling } from './utils';

const packageJson = JSON.parse(
  readFileSync(join(__dirname, '..', 'package.json'), 'utf8')
);

// Setup update notifications
const notifier = updateNotifier({
  pkg: packageJson,
  updateCheckInterval: 1000 * 60 * 60 * 24, // 24 hours
});

if (notifier.update) {
  console.log(
    boxen(
      `Update available: ${chalk.dim(notifier.update.current)} → ${chalk.green(
        notifier.update.latest
      )}\nRun ${chalk.cyan('npm i -g @arbitration-platform/cli')} to update`,
      {
        padding: 1,
        margin: 1,
        align: 'center',
        borderColor: 'yellow',
        borderStyle: 'round',
      }
    )
  );
}

// Setup error handling
setupErrorHandling();

// Create the main program
const program = new Command();

program
  .name('arbitration')
  .description('CLI tool for managing Arbitration Platform SDKs')
  .version(packageJson.version)
  .option('-v, --verbose', 'enable verbose logging')
  .option('--no-banner', 'disable banner display')
  .hook('preAction', async (thisCommand) => {
    const opts = thisCommand.opts();
    
    if (opts.banner !== false) {
      displayBanner();
    }

    if (opts.verbose) {
      process.env.VERBOSE = 'true';
    }
  });

// SDK Management Commands
program
  .command('init')
  .alias('create')
  .description('Initialize a new project with Arbitration SDK')
  .argument('[project-name]', 'name of the project')
  .option('-p, --platform <platform>', 'target platform (ios|android|react-native|flutter)', 'react-native')
  .option('-t, --template <template>', 'project template to use')
  .option('--git', 'initialize git repository')
  .option('--install', 'install dependencies after creation')
  .action(initCommand);

program
  .command('install')
  .alias('add')
  .description('Install or add Arbitration SDK to existing project')
  .argument('[platform]', 'platform to install (ios|android|react-native|flutter)')
  .option('-v, --version <version>', 'specific SDK version to install')
  .option('--dev', 'install as development dependency')
  .option('--save-exact', 'save exact version in package.json')
  .action(installCommand);

program
  .command('update')
  .alias('upgrade')
  .description('Update Arbitration SDK to latest version')
  .argument('[platform]', 'platform to update')
  .option('--check', 'check for available updates without installing')
  .option('--pre-release', 'include pre-release versions')
  .option('--force', 'force update even if breaking changes')
  .action(updateCommand);

// Configuration Commands
program
  .command('config')
  .description('Manage SDK configuration')
  .argument('[action]', 'action to perform (get|set|list|reset)')
  .argument('[key]', 'configuration key')
  .argument('[value]', 'configuration value')
  .option('-g, --global', 'operate on global configuration')
  .option('-l, --local', 'operate on local project configuration')
  .option('--json', 'output in JSON format')
  .action(configCommand);

// Analysis Commands
program
  .command('analyze')
  .description('Analyze documents for arbitration clauses')
  .argument('<files...>', 'files to analyze')
  .option('-o, --output <format>', 'output format (json|csv|html)', 'json')
  .option('-t, --threshold <threshold>', 'confidence threshold (0.0-1.0)', '0.5')
  .option('-r, --recursive', 'recursively analyze directories')
  .option('--batch', 'process files in batch mode')
  .option('--watch', 'watch files for changes')
  .action(analyzeCommand);

// Testing Commands
program
  .command('test')
  .description('Run SDK tests and validation')
  .argument('[platform]', 'platform to test')
  .option('--unit', 'run unit tests only')
  .option('--integration', 'run integration tests only')
  .option('--performance', 'run performance tests')
  .option('--coverage', 'generate coverage report')
  .option('--watch', 'watch mode for tests')
  .action(testCommand);

// Documentation Commands
program
  .command('docs')
  .description('Generate or serve documentation')
  .argument('[action]', 'action to perform (generate|serve|deploy)')
  .option('-p, --port <port>', 'port for documentation server', '8080')
  .option('-o, --output <dir>', 'output directory for generated docs')
  .option('--open', 'open documentation in browser')
  .action(docsCommand);

// Migration Commands
program
  .command('migrate')
  .description('Migrate from other arbitration detection libraries')
  .argument('<source>', 'source library (clauses-detector|legal-parser|contract-ai)')
  .option('--dry-run', 'preview migration without making changes')
  .option('--backup', 'create backup before migration')
  .option('--force', 'force migration even with conflicts')
  .action(migrateCommand);

// Validation Commands
program
  .command('validate')
  .description('Validate SDK setup and configuration')
  .argument('[platform]', 'platform to validate')
  .option('--fix', 'automatically fix common issues')
  .option('--report', 'generate detailed validation report')
  .action(validateCommand);

// Build Commands
program
  .command('build')
  .description('Build and package SDK for distribution')
  .argument('[platform]', 'platform to build')
  .option('--release', 'build for release')
  .option('--debug', 'build with debug symbols')
  .option('--clean', 'clean build artifacts first')
  .option('--archive', 'create distribution archive')
  .action(buildCommand);

// Global Options and Utilities
program
  .command('doctor')
  .description('Check SDK installation and environment')
  .option('--fix', 'automatically fix detected issues')
  .action(async (options) => {
    const { doctorCommand } = await import('./commands/doctor');
    await doctorCommand(options);
  });

program
  .command('clean')
  .description('Clean SDK cache and temporary files')
  .option('--cache', 'clean SDK cache only')
  .option('--temp', 'clean temporary files only')
  .option('--all', 'clean everything')
  .action(async (options) => {
    const { cleanCommand } = await import('./commands/clean');
    await cleanCommand(options);
  });

program
  .command('info')
  .description('Display SDK and environment information')
  .option('--json', 'output in JSON format')
  .action(async (options) => {
    const { infoCommand } = await import('./commands/info');
    await infoCommand(options);
  });

// Interactive mode
program
  .command('interactive')
  .alias('i')
  .description('Start interactive mode')
  .action(async () => {
    const { interactiveMode } = await import('./commands/interactive');
    await interactiveMode();
  });

// Help customization
program.configureHelp({
  sortSubcommands: true,
  subcommandTerm: (cmd) => cmd.name(),
});

// Custom help for common workflows
program
  .command('help-workflows')
  .description('Show common workflow examples')
  .action(() => {
    console.log(chalk.cyan.bold('\n📋 Common Workflows\n'));
    
    console.log(chalk.yellow('1. Start a new React Native project:'));
    console.log('   arbitration init my-app --platform react-native --install\n');
    
    console.log(chalk.yellow('2. Add SDK to existing Flutter project:'));
    console.log('   cd my-flutter-app');
    console.log('   arbitration install flutter\n');
    
    console.log(chalk.yellow('3. Analyze documents:'));
    console.log('   arbitration analyze contract.pdf terms.txt --output html\n');
    
    console.log(chalk.yellow('4. Update all SDKs:'));
    console.log('   arbitration update --check');
    console.log('   arbitration update\n');
    
    console.log(chalk.yellow('5. Test SDK integration:'));
    console.log('   arbitration test --integration --coverage\n');
    
    console.log(chalk.yellow('6. Validate setup:'));
    console.log('   arbitration doctor --fix\n');
    
    console.log(chalk.gray('For more help: arbitration <command> --help'));
  });

// Error handling for unknown commands
program.on('command:*', (operands) => {
  console.error(chalk.red(`Unknown command: ${operands[0]}`));
  console.log(chalk.gray('See --help for available commands'));
  process.exit(1);
});

// Parse command line arguments
if (process.argv.length === 2) {
  // No arguments provided, show help
  program.help();
} else {
  program.parse();
}

export { program };