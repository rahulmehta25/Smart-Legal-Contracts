import { AppRegistry } from 'react-native';
import App from './App';
import { name as appName } from './package.json';

// Register the app
AppRegistry.registerComponent(appName, () => App);