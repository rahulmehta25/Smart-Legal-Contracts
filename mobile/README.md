# Arbitration Detector Mobile App

A React Native mobile application for detecting arbitration clauses in documents using AI-powered OCR and text analysis.

## Features

### 🔍 Document Scanning
- AR-powered document scanner with real-time text detection
- High-quality OCR using native Vision frameworks (iOS) and ML Kit (Android)
- Automatic perspective correction and image enhancement
- Support for multiple document formats

### 🤖 AI Analysis
- Advanced arbitration clause detection using pattern matching and AI
- Risk level assessment (Low, Medium, High, Critical)
- Detailed explanations for detected clauses
- Confidence scoring for analysis results

### 📱 Mobile-First Experience
- Native performance with React Native
- Biometric authentication (Face ID, Touch ID, Fingerprint)
- Dark mode and system theme support
- Offline-first architecture with sync capabilities

### 🔒 Security & Privacy
- Local document processing with offline capabilities
- Encrypted local storage using SQLite
- Biometric authentication for sensitive actions
- Optional cloud sync with end-to-end encryption

### 📊 Analytics & History
- Comprehensive document history with search and filtering
- Analysis statistics and risk reports
- Export capabilities for legal review
- Weekly progress reports

## Architecture

### Frontend
- **React Native 0.73.2** - Cross-platform mobile framework
- **TypeScript** - Type safety and developer experience
- **React Navigation 6** - Navigation and routing
- **React Query** - Data fetching and caching
- **Zustand** - State management
- **React Native Paper** - Material Design components

### Native Modules
- **iOS Vision Framework** - High-performance OCR and document detection
- **Android ML Kit** - Google's machine learning for text recognition
- **Custom image processing** - Perspective correction and enhancement

### Data & Storage
- **SQLite** - Local database for documents and analyses
- **React Native Keychain** - Secure credential storage
- **AsyncStorage** - User preferences and app state
- **File system** - Document and image caching

### Services
- **Offline Service** - SQLite database management and sync queue
- **OCR Service** - Text recognition and document processing
- **Analysis Service** - Arbitration clause detection and risk assessment
- **Notification Service** - Push notifications and alerts
- **Biometric Service** - Authentication and security
- **User Service** - User management and preferences

## Installation

### Prerequisites
- Node.js 16+
- React Native CLI
- Xcode 14+ (for iOS development)
- Android Studio (for Android development)
- CocoaPods (for iOS dependencies)

### Setup
```bash
# Clone the repository
cd mobile

# Install dependencies
npm install

# iOS setup
cd ios
pod install
cd ..

# Android setup (if needed)
npx react-native link
```

### Development
```bash
# Start Metro bundler
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android
```

### Build for Production
```bash
# iOS
npm run build:ios

# Android
npm run build:android
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
API_BASE_URL=https://api.arbitrationdetector.com
FIREBASE_API_KEY=your_firebase_key
SENTRY_DSN=your_sentry_dsn
```

### Permissions

#### iOS (Info.plist)
```xml
<key>NSCameraUsageDescription</key>
<string>Camera access is required to scan documents</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Photo library access is required to select documents</string>
<key>NSFaceIDUsageDescription</key>
<string>Face ID is used for secure authentication</string>
```

#### Android (AndroidManifest.xml)
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.USE_FINGERPRINT" />
<uses-permission android:name="android.permission.USE_BIOMETRIC" />
```

## Usage

### Document Scanning
1. Open the app and navigate to the Scanner tab
2. Position your document within the camera frame
3. Tap the capture button or let auto-capture detect the document
4. Review the extracted text and save the document
5. Wait for analysis results or view them in the Analysis tab

### Analysis Results
- View detected arbitration clauses with explanations
- Check risk levels and recommendations
- Export results for legal review
- Share findings with team members

### Settings & Preferences
- Enable biometric authentication for enhanced security
- Configure notification preferences
- Toggle dark mode and system theme
- Manage offline mode and auto-sync settings

## Contributing

### Code Style
- Use TypeScript for all new code
- Follow ESLint and Prettier configurations
- Write unit tests for services and utilities
- Document complex functions and components

### Testing
```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run E2E tests (if configured)
npm run test:e2e
```

### Debugging
```bash
# Enable debugging
npm run debug

# View logs
npx react-native log-ios
npx react-native log-android
```

## Performance Optimization

### Image Processing
- Native modules handle heavy OCR operations
- Images are compressed and cached locally
- Background processing prevents UI blocking

### Memory Management
- Automatic image cleanup after processing
- SQLite query optimization
- Lazy loading for document lists

### Battery Optimization
- Efficient camera usage with auto-pause
- Background sync throttling
- Smart notification scheduling

## Security

### Data Protection
- All documents processed locally by default
- Optional encrypted cloud backup
- Automatic data expiration for security

### Authentication
- Biometric authentication for app access
- Secure keychain storage for credentials
- Session timeout and auto-lock

### Privacy
- No tracking or analytics by default
- User consent for data sharing
- GDPR and CCPA compliance ready

## Troubleshooting

### Common Issues

**OCR not working on Android:**
- Ensure Google Play Services is updated
- Check device compatibility with ML Kit

**iOS build fails:**
- Clean build folder: `rm -rf ios/build`
- Reinstall pods: `cd ios && pod install`
- Update Xcode and command line tools

**Camera permissions denied:**
- Check app permissions in device settings
- Restart app after granting permissions

**Performance issues:**
- Close other camera apps
- Free up device storage
- Update to latest app version

### Support
For technical support or questions:
- Create an issue on GitHub
- Email: support@arbitrationdetector.com
- Documentation: [docs.arbitrationdetector.com](https://docs.arbitrationdetector.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- React Native community for excellent libraries
- Google ML Kit for powerful text recognition
- Apple Vision framework for iOS OCR capabilities
- Open source contributors and maintainers