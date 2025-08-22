import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Dimensions,
  Image,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useTheme } from '@hooks/useTheme';

const { width } = Dimensions.get('window');

interface OnboardingSlide {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  icon: string;
  color: string;
}

const onboardingSlides: OnboardingSlide[] = [
  {
    id: '1',
    title: 'Welcome to Arbitration Detector',
    subtitle: 'Protect yourself from hidden clauses',
    description: 'Scan and analyze documents to detect arbitration clauses that could limit your legal rights.',
    icon: 'gavel',
    color: '#2563EB',
  },
  {
    id: '2',
    title: 'AI-Powered Document Scanning',
    subtitle: 'Advanced OCR and text recognition',
    description: 'Use your camera to scan documents with our advanced AI technology that extracts and analyzes text in real-time.',
    icon: 'document-scanner',
    color: '#7C3AED',
  },
  {
    id: '3',
    title: 'Instant Analysis Results',
    subtitle: 'Know your rights immediately',
    description: 'Get detailed analysis of arbitration clauses, risk assessments, and personalized recommendations.',
    icon: 'analytics',
    color: '#059669',
  },
  {
    id: '4',
    title: 'Offline & Secure',
    subtitle: 'Your privacy is protected',
    description: 'Documents are processed securely with offline capabilities. Your sensitive information never leaves your device unless you choose to sync.',
    icon: 'security',
    color: '#DC2626',
  },
];

const OnboardingFlowScreen: React.FC = () => {
  const { theme } = useTheme();
  const navigation = useNavigation();
  const [currentIndex, setCurrentIndex] = useState(0);
  const flatListRef = useRef<FlatList>(null);

  const nextSlide = () => {
    if (currentIndex < onboardingSlides.length - 1) {
      const nextIndex = currentIndex + 1;
      setCurrentIndex(nextIndex);
      flatListRef.current?.scrollToIndex({ index: nextIndex, animated: true });
    } else {
      completeOnboarding();
    }
  };

  const prevSlide = () => {
    if (currentIndex > 0) {
      const prevIndex = currentIndex - 1;
      setCurrentIndex(prevIndex);
      flatListRef.current?.scrollToIndex({ index: prevIndex, animated: true });
    }
  };

  const goToSlide = (index: number) => {
    setCurrentIndex(index);
    flatListRef.current?.scrollToIndex({ index, animated: true });
  };

  const completeOnboarding = () => {
    // Mark onboarding as completed in storage
    // Navigate to main app
    navigation.replace('Home');
  };

  const skipOnboarding = () => {
    completeOnboarding();
  };

  const onScroll = (event: any) => {
    const slideSize = event.nativeEvent.layoutMeasurement.width;
    const index = event.nativeEvent.contentOffset.x / slideSize;
    const roundIndex = Math.round(index);
    
    if (roundIndex !== currentIndex) {
      setCurrentIndex(roundIndex);
    }
  };

  const renderSlide = ({ item, index }: { item: OnboardingSlide; index: number }) => {
    const styles = createStyles(theme, item.color);
    
    return (
      <View style={styles.slide}>
        <View style={styles.iconContainer}>
          <View style={[styles.iconBackground, { backgroundColor: item.color + '20' }]}>
            <Icon name={item.icon} size={80} color={item.color} />
          </View>
        </View>
        
        <View style={styles.content}>
          <Text style={styles.title}>{item.title}</Text>
          <Text style={styles.subtitle}>{item.subtitle}</Text>
          <Text style={styles.description}>{item.description}</Text>
        </View>
      </View>
    );
  };

  const renderPagination = () => {
    const styles = createStyles(theme);
    
    return (
      <View style={styles.pagination}>
        {onboardingSlides.map((_, index) => (
          <TouchableOpacity
            key={index}
            style={[
              styles.paginationDot,
              {
                backgroundColor: index === currentIndex ? theme.colors.primary : theme.colors.border,
                width: index === currentIndex ? 30 : 10,
              },
            ]}
            onPress={() => goToSlide(index)}
          />
        ))}
      </View>
    );
  };

  const createStyles = (theme: any, slideColor?: string) => StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    slide: {
      width,
      flex: 1,
      alignItems: 'center',
      justifyContent: 'center',
      paddingHorizontal: theme.spacing.xl,
    },
    iconContainer: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
    },
    iconBackground: {
      width: 160,
      height: 160,
      borderRadius: 80,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.xl,
    },
    content: {
      flex: 1,
      alignItems: 'center',
      paddingTop: theme.spacing.xl,
    },
    title: {
      fontSize: theme.typography.fontSize.xxl,
      fontWeight: theme.typography.fontWeight.bold,
      color: theme.colors.text,
      textAlign: 'center',
      marginBottom: theme.spacing.md,
    },
    subtitle: {
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.medium,
      color: slideColor || theme.colors.primary,
      textAlign: 'center',
      marginBottom: theme.spacing.lg,
    },
    description: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.textSecondary,
      textAlign: 'center',
      lineHeight: 24,
      paddingHorizontal: theme.spacing.md,
    },
    navigationContainer: {
      paddingHorizontal: theme.spacing.xl,
      paddingVertical: theme.spacing.lg,
      backgroundColor: theme.colors.background,
    },
    pagination: {
      flexDirection: 'row',
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.xl,
    },
    paginationDot: {
      height: 10,
      borderRadius: 5,
      marginHorizontal: 5,
    },
    buttonContainer: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    button: {
      paddingVertical: theme.spacing.md,
      paddingHorizontal: theme.spacing.xl,
      borderRadius: theme.borderRadius.lg,
      minWidth: 100,
      alignItems: 'center',
    },
    primaryButton: {
      backgroundColor: theme.colors.primary,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 8,
      elevation: 4,
    },
    secondaryButton: {
      backgroundColor: 'transparent',
    },
    buttonText: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.bold,
    },
    primaryButtonText: {
      color: 'white',
    },
    secondaryButtonText: {
      color: theme.colors.textSecondary,
    },
    skipButton: {
      position: 'absolute',
      top: 60,
      right: theme.spacing.xl,
      paddingVertical: theme.spacing.sm,
      paddingHorizontal: theme.spacing.md,
    },
    skipButtonText: {
      color: theme.colors.textSecondary,
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
    },
  });

  const styles = createStyles(theme);

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.skipButton} onPress={skipOnboarding}>
        <Text style={styles.skipButtonText}>Skip</Text>
      </TouchableOpacity>

      <FlatList
        ref={flatListRef}
        data={onboardingSlides}
        renderItem={renderSlide}
        keyExtractor={(item) => item.id}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onScroll={onScroll}
        scrollEventThrottle={16}
      />

      <View style={styles.navigationContainer}>
        {renderPagination()}
        
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.button, styles.secondaryButton]}
            onPress={prevSlide}
            disabled={currentIndex === 0}
          >
            <Text
              style={[
                styles.buttonText,
                styles.secondaryButtonText,
                { opacity: currentIndex === 0 ? 0.3 : 1 },
              ]}
            >
              Previous
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, styles.primaryButton]}
            onPress={nextSlide}
          >
            <Text style={[styles.buttonText, styles.primaryButtonText]}>
              {currentIndex === onboardingSlides.length - 1 ? 'Get Started' : 'Next'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

export default OnboardingFlowScreen;