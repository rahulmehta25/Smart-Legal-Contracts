import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Share,
  ActivityIndicator,
} from 'react-native';
import { RouteProp, useRoute, useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useTheme } from '@hooks/useTheme';
import { RootStackParamList, Document, AnalysisStatus } from '@types/index';
import { documentService } from '@services/documentService';
import { analysisService } from '@services/analysisService';

type DocumentDetailsScreenRouteProp = RouteProp<RootStackParamList, 'DocumentDetails'>;

const DocumentDetailsScreen: React.FC = () => {
  const { theme } = useTheme();
  const route = useRoute<DocumentDetailsScreenRouteProp>();
  const navigation = useNavigation();
  const { documentId } = route.params;

  const [document, setDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);

  useEffect(() => {
    loadDocument();
  }, [documentId]);

  const loadDocument = async () => {
    try {
      setLoading(true);
      const doc = await documentService.getDocumentById(documentId);
      setDocument(doc);
    } catch (error) {
      console.error('Error loading document:', error);
      Alert.alert('Error', 'Failed to load document details');
    } finally {
      setLoading(false);
    }
  };

  const startAnalysis = async () => {
    if (!document) return;

    try {
      setAnalyzing(true);
      await analysisService.analyzeDocument(document);
      
      // Update document status
      const updatedDoc = { ...document, analysisStatus: AnalysisStatus.PROCESSING };
      setDocument(updatedDoc);
      
      Alert.alert(
        'Analysis Started',
        'Your document is being analyzed. You will be notified when complete.',
        [
          {
            text: 'View Progress',
            onPress: () => navigation.navigate('Analysis', { documentId: document.id }),
          },
          { text: 'OK' },
        ]
      );
    } catch (error) {
      console.error('Error starting analysis:', error);
      Alert.alert('Error', 'Failed to start document analysis');
    } finally {
      setAnalyzing(false);
    }
  };

  const shareDocument = async () => {
    if (!document) return;

    try {
      const message = `Document: ${document.name}\n\nExtracted Text:\n${document.extractedText.substring(0, 500)}${document.extractedText.length > 500 ? '...' : ''}`;
      
      await Share.share({
        message,
        title: document.name,
      });
    } catch (error) {
      console.error('Error sharing document:', error);
    }
  };

  const deleteDocument = async () => {
    if (!document) return;

    Alert.alert(
      'Delete Document',
      'Are you sure you want to delete this document? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await documentService.deleteDocument(document.id);
              await analysisService.deleteAnalysisByDocumentId(document.id);
              navigation.goBack();
            } catch (error) {
              console.error('Error deleting document:', error);
              Alert.alert('Error', 'Failed to delete document');
            }
          },
        },
      ]
    );
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusColor = (status: AnalysisStatus) => {
    switch (status) {
      case AnalysisStatus.COMPLETED:
        return theme.colors.success;
      case AnalysisStatus.PROCESSING:
        return theme.colors.warning;
      case AnalysisStatus.FAILED:
        return theme.colors.error;
      default:
        return theme.colors.textSecondary;
    }
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    loadingContainer: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
    },
    scrollContent: {
      padding: theme.spacing.md,
    },
    headerCard: {
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.lg,
      borderRadius: theme.borderRadius.lg,
      marginBottom: theme.spacing.lg,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    documentIcon: {
      width: 60,
      height: 60,
      borderRadius: 30,
      backgroundColor: theme.colors.primary + '20',
      justifyContent: 'center',
      alignItems: 'center',
      alignSelf: 'center',
      marginBottom: theme.spacing.md,
    },
    documentName: {
      fontSize: theme.typography.fontSize.xl,
      fontWeight: theme.typography.fontWeight.bold,
      color: theme.colors.text,
      textAlign: 'center',
      marginBottom: theme.spacing.sm,
    },
    statusContainer: {
      flexDirection: 'row',
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
    },
    statusBadge: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      borderRadius: theme.borderRadius.lg,
    },
    statusText: {
      color: 'white',
      fontSize: theme.typography.fontSize.sm,
      fontWeight: theme.typography.fontWeight.medium,
      marginLeft: theme.spacing.xs,
    },
    metadataGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      justifyContent: 'space-between',
    },
    metadataItem: {
      width: '48%',
      marginBottom: theme.spacing.sm,
    },
    metadataLabel: {
      fontSize: theme.typography.fontSize.sm,
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.xs,
    },
    metadataValue: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.text,
      fontWeight: theme.typography.fontWeight.medium,
    },
    actionsCard: {
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.lg,
      borderRadius: theme.borderRadius.lg,
      marginBottom: theme.spacing.lg,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    sectionTitle: {
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.bold,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    actionButton: {
      flexDirection: 'row',
      alignItems: 'center',
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.lg,
      marginBottom: theme.spacing.sm,
    },
    primaryAction: {
      backgroundColor: theme.colors.primary,
    },
    secondaryAction: {
      backgroundColor: theme.colors.background,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    dangerAction: {
      backgroundColor: theme.colors.error + '10',
      borderWidth: 1,
      borderColor: theme.colors.error,
    },
    actionIcon: {
      marginRight: theme.spacing.md,
    },
    actionText: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
    },
    primaryActionText: {
      color: 'white',
    },
    secondaryActionText: {
      color: theme.colors.text,
    },
    dangerActionText: {
      color: theme.colors.error,
    },
    contentCard: {
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.lg,
      borderRadius: theme.borderRadius.lg,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    extractedText: {
      fontSize: theme.typography.fontSize.md,
      lineHeight: 24,
      color: theme.colors.text,
    },
    noContentText: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.textSecondary,
      textAlign: 'center',
      fontStyle: 'italic',
    },
  });

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={[styles.metadataValue, { marginTop: theme.spacing.md }]}>
          Loading document...
        </Text>
      </View>
    );
  }

  if (!document) {
    return (
      <View style={styles.loadingContainer}>
        <Icon name="error-outline" size={64} color={theme.colors.error} />
        <Text style={[styles.metadataValue, { color: theme.colors.error, marginTop: theme.spacing.md }]}>
          Document not found
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
        {/* Document Header */}
        <View style={styles.headerCard}>
          <View style={styles.documentIcon}>
            <Icon name="description" size={30} color={theme.colors.primary} />
          </View>
          
          <Text style={styles.documentName} numberOfLines={2}>
            {document.name}
          </Text>
          
          <View style={styles.statusContainer}>
            <View style={[styles.statusBadge, { backgroundColor: getStatusColor(document.analysisStatus) }]}>
              <Icon
                name={
                  document.analysisStatus === AnalysisStatus.COMPLETED
                    ? 'check-circle'
                    : document.analysisStatus === AnalysisStatus.PROCESSING
                    ? 'hourglass-empty'
                    : document.analysisStatus === AnalysisStatus.FAILED
                    ? 'error'
                    : 'schedule'
                }
                size={16}
                color="white"
              />
              <Text style={styles.statusText}>
                {document.analysisStatus.toUpperCase()}
              </Text>
            </View>
          </View>
          
          <View style={styles.metadataGrid}>
            <View style={styles.metadataItem}>
              <Text style={styles.metadataLabel}>File Size</Text>
              <Text style={styles.metadataValue}>{formatFileSize(document.size)}</Text>
            </View>
            <View style={styles.metadataItem}>
              <Text style={styles.metadataLabel}>File Type</Text>
              <Text style={styles.metadataValue}>{document.type.toUpperCase()}</Text>
            </View>
            <View style={styles.metadataItem}>
              <Text style={styles.metadataLabel}>Created</Text>
              <Text style={styles.metadataValue}>{formatDate(document.createdAt)}</Text>
            </View>
            <View style={styles.metadataItem}>
              <Text style={styles.metadataLabel}>Updated</Text>
              <Text style={styles.metadataValue}>{formatDate(document.updatedAt)}</Text>
            </View>
          </View>
        </View>

        {/* Actions */}
        <View style={styles.actionsCard}>
          <Text style={styles.sectionTitle}>Actions</Text>
          
          {document.analysisStatus === AnalysisStatus.PENDING && (
            <TouchableOpacity
              style={[styles.actionButton, styles.primaryAction]}
              onPress={startAnalysis}
              disabled={analyzing}
            >
              {analyzing ? (
                <ActivityIndicator size="small" color="white" style={styles.actionIcon} />
              ) : (
                <Icon name="analytics" size={20} color="white" style={styles.actionIcon} />
              )}
              <Text style={[styles.actionText, styles.primaryActionText]}>
                {analyzing ? 'Starting Analysis...' : 'Start Analysis'}
              </Text>
            </TouchableOpacity>
          )}
          
          {document.analysisStatus === AnalysisStatus.COMPLETED && (
            <TouchableOpacity
              style={[styles.actionButton, styles.primaryAction]}
              onPress={() => navigation.navigate('Analysis', { documentId: document.id })}
            >
              <Icon name="visibility" size={20} color="white" style={styles.actionIcon} />
              <Text style={[styles.actionText, styles.primaryActionText]}>
                View Analysis Results
              </Text>
            </TouchableOpacity>
          )}
          
          <TouchableOpacity
            style={[styles.actionButton, styles.secondaryAction]}
            onPress={shareDocument}
          >
            <Icon name="share" size={20} color={theme.colors.text} style={styles.actionIcon} />
            <Text style={[styles.actionText, styles.secondaryActionText]}>
              Share Document
            </Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={[styles.actionButton, styles.dangerAction]}
            onPress={deleteDocument}
          >
            <Icon name="delete" size={20} color={theme.colors.error} style={styles.actionIcon} />
            <Text style={[styles.actionText, styles.dangerActionText]}>
              Delete Document
            </Text>
          </TouchableOpacity>
        </View>

        {/* Extracted Content */}
        <View style={styles.contentCard}>
          <Text style={styles.sectionTitle}>Extracted Text</Text>
          {document.extractedText ? (
            <Text style={styles.extractedText}>{document.extractedText}</Text>
          ) : (
            <Text style={styles.noContentText}>
              No text content available for this document
            </Text>
          )}
        </View>
      </ScrollView>
    </View>
  );
};

export default DocumentDetailsScreen;