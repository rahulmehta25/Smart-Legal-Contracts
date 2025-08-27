#!/usr/bin/env node

const { AutomatedDemo } = require('./automated-demo');

async function runFullDemo() {
  console.log('🚀 Starting 10-minute comprehensive demo...');
  
  const demo = new AutomatedDemo();
  
  try {
    await demo.initialize({ 
      headless: false,
      recordVideo: process.argv.includes('--record')
    });
    
    await demo.runFullDemo();
    
    console.log('✅ Full demo completed successfully!');
    console.log('📁 Screenshots saved to: screenshots/');
    
    if (process.argv.includes('--record')) {
      console.log('🎥 Video saved to: demo-videos/');
    }
    
    console.log('\n📊 Demo Summary:');
    console.log('- Homepage tour with feature highlights');
    console.log('- Multiple document upload demonstration');
    console.log('- Document type comparison (with/without arbitration)');
    console.log('- Analytics dashboard showcase');
    console.log('- Export functionality demo');
    console.log('- Mobile responsive experience');
    console.log('- API integration preview');
    
  } catch (error) {
    console.error('❌ Full demo failed:', error);
    process.exit(1);
  } finally {
    await demo.cleanup();
  }
}

if (require.main === module) {
  runFullDemo();
}

module.exports = { runFullDemo };