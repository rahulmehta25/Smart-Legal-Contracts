import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Dimensions,
  ActivityIndicator,
  Modal,
  Animated,
} from 'react-native';
import { Camera, useCameraDevices, useFrameProcessor } from 'react-native-vision-camera';
import { runOnJS } from 'react-native-reanimated';
import { check, request, PERMISSIONS, RESULTS } from 'react-native-permissions';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useTheme } from '@hooks/useTheme';
import { useNavigation } from '@react-navigation/native';
import ARScanner from '@components/ARScanner';
import { documentService } from '@services/documentService';
import { ocrService } from '@services/ocrService';
import { ScanResult, CameraPermissions } from '@types/index';

const { width, height } = Dimensions.get('window');

const ScannerScreen: React.FC = () => {
  const { theme } = useTheme();
  const navigation = useNavigation();
  
  const [hasPermissions, setHasPermissions] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [flashEnabled, setFlashEnabled] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const camera = useRef<Camera>(null);
  const devices = useCameraDevices();
  const device = devices.back;
  
  const overlayOpacity = useRef(new Animated.Value(0)).current;
  const scanLinePosition = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    checkPermissions();
    startScanLineAnimation();
  }, []);

  const checkPermissions = async () => {
    try {
      const cameraPermission = await check(PERMISSIONS.IOS.CAMERA);
      const storagePermission = await check(PERMISSIONS.IOS.PHOTO_LIBRARY);
      
      if (cameraPermission === RESULTS.GRANTED && storagePermission === RESULTS.GRANTED) {
        setHasPermissions(true);
      } else {
        await requestPermissions();
      }
    } catch (error) {
      console.error('Error checking permissions:', error);
      Alert.alert('Error', 'Failed to check camera permissions');
    }
  };

  const requestPermissions = async () => {
    try {
      const cameraResult = await request(PERMISSIONS.IOS.CAMERA);
      const storageResult = await request(PERMISSIONS.IOS.PHOTO_LIBRARY);
      
      if (cameraResult === RESULTS.GRANTED && storageResult === RESULTS.GRANTED) {
        setHasPermissions(true);
      } else {
        Alert.alert(
          'Permissions Required',
          'Camera and storage permissions are required to scan documents.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Settings', onPress: () => {/* Open settings */} }
          ]
        );
      }
    } catch (error) {
      console.error('Error requesting permissions:', error);
    }
  };

  const startScanLineAnimation = () => {
    const scanAnimation = Animated.loop(
      Animated.sequence([
        Animated.timing(scanLinePosition, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(scanLinePosition, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: true,
        }),
      ])
    );
    scanAnimation.start();
  };

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    
    if (isScanning) {
      // Process frame for text detection
      runOnJS(processFrame)(frame);
    }
  }, [isScanning]);

  const processFrame = async (frame: any) => {
    if (isProcessing) return;
    
    try {
      setIsProcessing(true);
      
      // Extract text from frame using OCR service
      const result = await ocrService.extractTextFromFrame(frame);
      
      if (result && result.text.length > 50) { // Minimum text threshold
        setIsScanning(false);
        setScanResult(result);
        setShowPreview(true);
        
        // Show overlay animation
        Animated.timing(overlayOpacity, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        }).start();
      }
    } catch (error) {
      console.error('Error processing frame:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const takePicture = async () => {
    if (!camera.current) return;
    
    try {
      setIsScanning(true);
      
      const photo = await camera.current.takePhoto({
        quality: 90,
        enableAutoRedEyeReduction: true,
        enableAutoStabilization: true,
        enableShutterSound: true,
      });
      
      // Process the captured image
      const result = await ocrService.extractTextFromImage(photo.path);
      
      if (result) {
        setScanResult(result);
        setShowPreview(true);
        
        Animated.timing(overlayOpacity, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        }).start();
      } else {
        Alert.alert('No Text Detected', 'Please try again with better lighting or positioning.');
      }
    } catch (error) {
      console.error('Error taking picture:', error);
      Alert.alert('Error', 'Failed to capture image. Please try again.');
    } finally {
      setIsScanning(false);
    }
  };

  const retakePhoto = () => {
    setShowPreview(false);
    setScanResult(null);
    
    Animated.timing(overlayOpacity, {
      toValue: 0,
      duration: 300,
      useNativeDriver: true,
    }).start();
  };

  const saveDocument = async () => {
    if (!scanResult) return;
    
    try {
      setIsProcessing(true);
      
      // Create document from scan result
      const document = await documentService.createFromScan(scanResult);
      
      Alert.alert(
        'Document Saved',
        'Your document has been saved and queued for analysis.',
        [
          {
            text: 'View Document',
            onPress: () => {
              navigation.navigate('DocumentDetails', { documentId: document.id });
            },
          },
          { text: 'Scan Another', onPress: retakePhoto },
        ]
      );
    } catch (error) {
      console.error('Error saving document:', error);
      Alert.alert('Error', 'Failed to save document. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleFlash = () => {
    setFlashEnabled(!flashEnabled);
  };

  if (!hasPermissions) {
    return (
      <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <View style={styles.permissionContainer}>
          <Icon name="camera-alt" size={64} color={theme.colors.textSecondary} />
          <Text style={[styles.permissionTitle, { color: theme.colors.text }]}>
            Camera Permission Required
          </Text>
          <Text style={[styles.permissionText, { color: theme.colors.textSecondary }]}>
            Please grant camera and storage permissions to scan documents.
          </Text>
          <TouchableOpacity
            style={[styles.permissionButton, { backgroundColor: theme.colors.primary }]}
            onPress={requestPermissions}
          >
            <Text style={styles.permissionButtonText}>Grant Permissions</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <Text style={[styles.errorText, { color: theme.colors.text }]}>
          Camera not available
        </Text>
      </View>
    );
  }

  const styles = StyleSheet.create({
    container: {
      flex: 1,
    },
    camera: {
      flex: 1,
    },
    overlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      justifyContent: 'center',
      alignItems: 'center',
    },
    scanFrame: {
      width: width * 0.8,
      height: height * 0.6,
      borderWidth: 2,
      borderColor: theme.colors.primary,
      borderRadius: theme.borderRadius.lg,
      backgroundColor: 'transparent',
    },
    scanLine: {
      position: 'absolute',
      left: 0,
      right: 0,
      height: 2,
      backgroundColor: theme.colors.primary,
    },
    controlsContainer: {
      position: 'absolute',
      bottom: 0,
      left: 0,
      right: 0,
      backgroundColor: theme.colors.background + 'E6',
      paddingVertical: theme.spacing.lg,
      paddingHorizontal: theme.spacing.md,
    },
    controlsRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    controlButton: {
      width: 60,
      height: 60,
      borderRadius: 30,
      backgroundColor: theme.colors.surface,
      justifyContent: 'center',
      alignItems: 'center',
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.25,
      shadowRadius: 4,
      elevation: 5,
    },
    captureButton: {
      width: 80,
      height: 80,
      borderRadius: 40,
      backgroundColor: theme.colors.primary,
      justifyContent: 'center',
      alignItems: 'center',
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.3,
      shadowRadius: 6,
      elevation: 8,
    },
    instructionContainer: {
      position: 'absolute',
      top: 100,
      left: theme.spacing.md,
      right: theme.spacing.md,
      backgroundColor: theme.colors.background + 'E6',
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.lg,
    },
    instructionText: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.text,
      textAlign: 'center',
      fontWeight: theme.typography.fontWeight.medium,
    },
    previewModal: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    previewHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: theme.spacing.md,
      backgroundColor: theme.colors.surface,
    },
    previewTitle: {
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.bold,
      color: theme.colors.text,
    },
    previewContent: {
      flex: 1,
      padding: theme.spacing.md,
    },
    previewText: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.text,
      lineHeight: 24,
    },
    previewActions: {
      flexDirection: 'row',
      padding: theme.spacing.md,
      gap: theme.spacing.md,
    },
    actionButton: {
      flex: 1,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.borderRadius.lg,
      alignItems: 'center',
    },
    primaryButton: {
      backgroundColor: theme.colors.primary,
    },
    secondaryButton: {
      backgroundColor: theme.colors.surface,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    actionButtonText: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
    },
    primaryButtonText: {
      color: 'white',
    },
    secondaryButtonText: {
      color: theme.colors.text,
    },
    permissionContainer: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      padding: theme.spacing.xl,
    },
    permissionTitle: {
      fontSize: theme.typography.fontSize.xl,
      fontWeight: theme.typography.fontWeight.bold,
      marginTop: theme.spacing.md,
      marginBottom: theme.spacing.sm,
      textAlign: 'center',
    },
    permissionText: {
      fontSize: theme.typography.fontSize.md,
      textAlign: 'center',
      marginBottom: theme.spacing.xl,
    },
    permissionButton: {
      paddingHorizontal: theme.spacing.xl,
      paddingVertical: theme.spacing.md,
      borderRadius: theme.borderRadius.lg,
    },
    permissionButtonText: {
      color: 'white',
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
    },
    errorText: {
      fontSize: theme.typography.fontSize.lg,
      textAlign: 'center',
      margin: theme.spacing.xl,
    },
    loadingOverlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      justifyContent: 'center',
      alignItems: 'center',
    },
    loadingText: {
      color: 'white',
      fontSize: theme.typography.fontSize.md,
      marginTop: theme.spacing.md,
      fontWeight: theme.typography.fontWeight.medium,
    },
  });

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={styles.camera}
        device={device}
        isActive={!showPreview}
        photo={true}
        frameProcessor={frameProcessor}
        torch={flashEnabled ? 'on' : 'off'}
      />
      
      {!showPreview && (
        <>
          <Animated.View style={[styles.overlay, { opacity: overlayOpacity }]}>
            <View style={styles.scanFrame}>
              <Animated.View
                style={[
                  styles.scanLine,
                  {
                    transform: [
                      {
                        translateY: scanLinePosition.interpolate({
                          inputRange: [0, 1],
                          outputRange: [0, height * 0.6 - 2],
                        }),
                      },
                    ],
                  },
                ]}
              />
            </View>
          </Animated.View>
          
          <View style={styles.instructionContainer}>
            <Text style={styles.instructionText}>
              Position your document within the frame and tap capture
            </Text>
          </View>
          
          <View style={styles.controlsContainer}>
            <View style={styles.controlsRow}>
              <TouchableOpacity
                style={styles.controlButton}
                onPress={toggleFlash}
              >
                <Icon
                  name={flashEnabled ? 'flash-on' : 'flash-off'}
                  size={24}
                  color={theme.colors.text}
                />
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.captureButton}
                onPress={takePicture}
                disabled={isProcessing}
              >
                {isProcessing ? (
                  <ActivityIndicator color="white" size="large" />
                ) : (
                  <Icon name="camera-alt" size={32} color="white" />
                )}
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.controlButton}
                onPress={() => navigation.goBack()}
              >
                <Icon name="close" size={24} color={theme.colors.text} />
              </TouchableOpacity>
            </View>
          </View>
        </>
      )}
      
      <Modal
        visible={showPreview}
        animationType="slide"
        onRequestClose={retakePhoto}
      >
        <View style={styles.previewModal}>
          <View style={styles.previewHeader}>
            <Text style={styles.previewTitle}>Scanned Text</Text>
            <TouchableOpacity onPress={retakePhoto}>
              <Icon name="close" size={24} color={theme.colors.text} />
            </TouchableOpacity>
          </View>
          
          <View style={styles.previewContent}>
            <Text style={styles.previewText}>
              {scanResult?.text || 'No text detected'}
            </Text>
          </View>
          
          <View style={styles.previewActions}>
            <TouchableOpacity
              style={[styles.actionButton, styles.secondaryButton]}
              onPress={retakePhoto}
            >
              <Text style={[styles.actionButtonText, styles.secondaryButtonText]}>
                Retake
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.actionButton, styles.primaryButton]}
              onPress={saveDocument}
              disabled={isProcessing}
            >
              {isProcessing ? (
                <ActivityIndicator color="white" />
              ) : (
                <Text style={[styles.actionButtonText, styles.primaryButtonText]}>
                  Save & Analyze
                </Text>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
      
      {isProcessing && !showPreview && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator color="white" size="large" />
          <Text style={styles.loadingText}>Processing...</Text>
        </View>
      )}
    </View>
  );
};

export default ScannerScreen;