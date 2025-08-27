#!/usr/bin/env node

const { AutomatedDemo } = require('./automated-demo');

async function runQuickDemo() {
  console.log('🚀 Starting 2-minute quick demo...');
  
  const demo = new AutomatedDemo();
  
  try {
    await demo.initialize({ 
      headless: false,
      recordVideo: process.argv.includes('--record')
    });
    
    await demo.runQuickDemo();
    
    console.log('✅ Quick demo completed successfully!');
    console.log('📁 Screenshots saved to: screenshots/');
    
    if (process.argv.includes('--record')) {
      console.log('🎥 Video saved to: demo-videos/');
    }
    
  } catch (error) {
    console.error('❌ Quick demo failed:', error);
    process.exit(1);
  } finally {
    await demo.cleanup();
  }
}

if (require.main === module) {
  runQuickDemo();
}

module.exports = { runQuickDemo };