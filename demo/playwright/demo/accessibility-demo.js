#!/usr/bin/env node

const { InteractionDemoGenerator } = require('./interaction-demo');

async function runAccessibilityDemo() {
  console.log('♿ Starting accessibility features demo...');
  
  const demo = new InteractionDemoGenerator();
  
  try {
    await demo.initialize();
    
    console.log('🎯 Demonstrating accessibility features...');
    await demo.runAccessibilityDemo();
    
    console.log('✅ Accessibility demo completed!');
    console.log('\n♿ Accessibility Features Demonstrated:');
    console.log('- Keyboard navigation support');
    console.log('- Screen reader compatibility');
    console.log('- High contrast mode');
    console.log('- Zoom level testing (up to 200%)');
    console.log('- Focus management');
    console.log('- ARIA attributes and landmarks');
    console.log('- Color contrast compliance');
    console.log('- Touch target sizing (44px minimum)');
    
  } catch (error) {
    console.error('❌ Accessibility demo failed:', error);
    process.exit(1);
  } finally {
    await demo.cleanup();
  }
}

if (require.main === module) {
  runAccessibilityDemo();
}

module.exports = { runAccessibilityDemo };