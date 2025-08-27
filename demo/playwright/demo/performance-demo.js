#!/usr/bin/env node

const { InteractionDemoGenerator } = require('./interaction-demo');

async function runPerformanceDemo() {
  console.log('⚡ Starting performance showcase demo...');
  
  const demo = new InteractionDemoGenerator();
  
  try {
    await demo.initialize();
    
    console.log('📊 Measuring Core Web Vitals...');
    await demo.runPerformanceDemo();
    
    console.log('✅ Performance demo completed!');
    console.log('\n⚡ Performance Highlights:');
    console.log('- First Contentful Paint (FCP)');
    console.log('- Largest Contentful Paint (LCP)');
    console.log('- Cumulative Layout Shift (CLS)');
    console.log('- Time to Interactive (TTI)');
    console.log('- Document analysis speed');
    console.log('- Batch processing capabilities');
    
  } catch (error) {
    console.error('❌ Performance demo failed:', error);
    process.exit(1);
  } finally {
    await demo.cleanup();
  }
}

if (require.main === module) {
  runPerformanceDemo();
}

module.exports = { runPerformanceDemo };