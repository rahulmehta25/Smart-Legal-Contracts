import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Dimensions,
} from 'react-native';
import { useTheme } from '@hooks/useTheme';
import { BoundingBox, ScanResult } from '@types/index';

const { width, height } = Dimensions.get('window');

interface ARScannerProps {
  onTextDetected: (result: ScanResult) => void;
  isActive: boolean;
  boundingBoxes: BoundingBox[];
  scanResult?: ScanResult;
}

const ARScanner: React.FC<ARScannerProps> = ({
  onTextDetected,
  isActive,
  boundingBoxes,
  scanResult,
}) => {
  const { theme } = useTheme();
  
  const scanLinePosition = useRef(new Animated.Value(0)).current;
  const cornerAnimations = useRef([
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
    new Animated.Value(0),
  ]).current;
  const boundingBoxAnimations = useRef(new Map()).current;

  useEffect(() => {
    if (isActive) {
      startScanAnimation();
      animateCorners();
    } else {
      stopScanAnimation();
    }
  }, [isActive]);

  useEffect(() => {
    if (boundingBoxes.length > 0) {
      animateBoundingBoxes();
    }
  }, [boundingBoxes]);

  const startScanAnimation = () => {
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

  const stopScanAnimation = () => {
    scanLinePosition.stopAnimation();
    scanLinePosition.setValue(0);
  };

  const animateCorners = () => {
    const animations = cornerAnimations.map((anim, index) =>
      Animated.loop(
        Animated.sequence([
          Animated.timing(anim, {
            toValue: 1,
            duration: 1000 + index * 200,
            useNativeDriver: true,
          }),
          Animated.timing(anim, {
            toValue: 0,
            duration: 1000 + index * 200,
            useNativeDriver: true,
          }),
        ])
      )
    );
    
    Animated.stagger(200, animations).start();
  };

  const animateBoundingBoxes = () => {
    boundingBoxes.forEach((box, index) => {
      const key = `${box.x}-${box.y}-${index}`;
      
      if (!boundingBoxAnimations.has(key)) {
        boundingBoxAnimations.set(key, new Animated.Value(0));
      }
      
      const animation = boundingBoxAnimations.get(key);
      
      Animated.sequence([
        Animated.timing(animation, {
          toValue: 1,
          duration: 300,
          useNativeDriver: true,
        }),
        Animated.delay(2000),
        Animated.timing(animation, {
          toValue: 0,
          duration: 300,
          useNativeDriver: true,
        }),
      ]).start(() => {
        boundingBoxAnimations.delete(key);
      });
    });
  };

  const renderBoundingBoxes = () => {
    return boundingBoxes.map((box, index) => {
      const key = `${box.x}-${box.y}-${index}`;
      const animation = boundingBoxAnimations.get(key) || new Animated.Value(0);
      
      const confidenceColor = box.confidence > 0.8 
        ? theme.colors.success 
        : box.confidence > 0.6 
        ? theme.colors.warning 
        : theme.colors.error;

      return (
        <Animated.View
          key={key}
          style={[
            styles.boundingBox,
            {
              left: box.x,
              top: box.y,
              width: box.width,
              height: box.height,
              borderColor: confidenceColor,
              opacity: animation,
            },
          ]}
        >
          <View style={[styles.confidenceBadge, { backgroundColor: confidenceColor }]}>
            <Text style={styles.confidenceText}>
              {Math.round(box.confidence * 100)}%
            </Text>
          </View>
          {box.text.length < 50 && (
            <View style={[styles.textPreview, { backgroundColor: theme.colors.background + 'E6' }]}>
              <Text style={[styles.textPreviewText, { color: theme.colors.text }]} numberOfLines={2}>
                {box.text}
              </Text>
            </View>
          )}
        </Animated.View>
      );
    });
  };

  const renderScanFrame = () => {
    const frameWidth = width * 0.85;
    const frameHeight = height * 0.65;
    
    return (
      <View style={[
        styles.scanFrame,
        {
          width: frameWidth,
          height: frameHeight,
          borderColor: isActive ? theme.colors.primary : theme.colors.textSecondary,
        }
      ]}>
        {/* Corner indicators */}
        {[0, 1, 2, 3].map((index) => (
          <Animated.View
            key={index}
            style={[
              styles.corner,
              {
                backgroundColor: theme.colors.primary,
                opacity: cornerAnimations[index],
                ...(index === 0 && { top: -2, left: -2 }),
                ...(index === 1 && { top: -2, right: -2 }),
                ...(index === 2 && { bottom: -2, left: -2 }),
                ...(index === 3 && { bottom: -2, right: -2 }),
              },
            ]}
          />
        ))}
        
        {/* Scan line */}
        {isActive && (
          <Animated.View
            style={[
              styles.scanLine,
              {
                backgroundColor: theme.colors.primary,
                transform: [
                  {
                    translateY: scanLinePosition.interpolate({
                      inputRange: [0, 1],
                      outputRange: [0, frameHeight - 2],
                    }),
                  },
                ],
              },
            ]}
          />
        )}
        
        {/* Grid overlay for better positioning */}
        <View style={styles.gridOverlay}>
          {[1, 2].map((i) => (
            <View
              key={`horizontal-${i}`}
              style={[
                styles.gridLine,
                {
                  top: (frameHeight / 3) * i,
                  width: '100%',
                  height: 1,
                  backgroundColor: theme.colors.textSecondary + '40',
                },
              ]}
            />
          ))}
          {[1, 2].map((i) => (
            <View
              key={`vertical-${i}`}
              style={[
                styles.gridLine,
                {
                  left: (frameWidth / 3) * i,
                  height: '100%',
                  width: 1,
                  backgroundColor: theme.colors.textSecondary + '40',
                },
              ]}
            />
          ))}
        </View>
      </View>
    );
  };

  const renderStatusIndicator = () => {
    if (!isActive) return null;
    
    const statusText = boundingBoxes.length > 0 
      ? `Detected ${boundingBoxes.length} text block${boundingBoxes.length > 1 ? 's' : ''}`
      : 'Scanning for text...';
    
    const statusColor = boundingBoxes.length > 0 ? theme.colors.success : theme.colors.warning;
    
    return (
      <View style={[styles.statusIndicator, { backgroundColor: theme.colors.background + 'E6' }]}>
        <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
        <Text style={[styles.statusText, { color: theme.colors.text }]}>
          {statusText}
        </Text>
      </View>
    );
  };

  const styles = StyleSheet.create({
    container: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      justifyContent: 'center',
      alignItems: 'center',
    },
    scanFrame: {
      borderWidth: 2,
      borderRadius: theme.borderRadius.lg,
      position: 'relative',
      backgroundColor: 'transparent',
    },
    corner: {
      position: 'absolute',
      width: 20,
      height: 20,
      borderRadius: 10,
    },
    scanLine: {
      position: 'absolute',
      left: 0,
      right: 0,
      height: 2,
      shadowColor: theme.colors.primary,
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: 0.8,
      shadowRadius: 4,
      elevation: 5,
    },
    gridOverlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
    },
    gridLine: {
      position: 'absolute',
    },
    boundingBox: {
      position: 'absolute',
      borderWidth: 2,
      borderRadius: 4,
      backgroundColor: 'transparent',
    },
    confidenceBadge: {
      position: 'absolute',
      top: -12,
      right: -1,
      paddingHorizontal: 6,
      paddingVertical: 2,
      borderRadius: 8,
    },
    confidenceText: {
      fontSize: 10,
      fontWeight: 'bold',
      color: 'white',
    },
    textPreview: {
      position: 'absolute',
      bottom: -30,
      left: 0,
      right: 0,
      padding: 4,
      borderRadius: 4,
      maxHeight: 28,
    },
    textPreviewText: {
      fontSize: 10,
      lineHeight: 12,
    },
    statusIndicator: {
      position: 'absolute',
      top: 80,
      alignSelf: 'center',
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      borderRadius: theme.borderRadius.lg,
    },
    statusDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      marginRight: theme.spacing.sm,
    },
    statusText: {
      fontSize: theme.typography.fontSize.sm,
      fontWeight: theme.typography.fontWeight.medium,
    },
  });

  return (
    <View style={styles.container}>
      {renderScanFrame()}
      {renderBoundingBoxes()}
      {renderStatusIndicator()}
    </View>
  );
};

export default ARScanner;