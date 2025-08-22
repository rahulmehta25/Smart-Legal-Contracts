import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  TextInput,
  Modal,
  Alert,
} from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useTheme } from '@hooks/useTheme';
import { Document, AnalysisStatus, RiskLevel } from '@types/index';
import { documentService } from '@services/documentService';
import { analysisService } from '@services/analysisService';

interface DocumentWithAnalysis extends Document {
  riskLevel?: RiskLevel;
  hasArbitrationClause?: boolean;
  confidence?: number;
}

const HistoryScreen: React.FC = () => {
  const { theme } = useTheme();
  const navigation = useNavigation();
  
  const [documents, setDocuments] = useState<DocumentWithAnalysis[]>([]);
  const [filteredDocuments, setFilteredDocuments] = useState<DocumentWithAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterModalVisible, setFilterModalVisible] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'analyzed' | 'pending' | 'high_risk'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'name' | 'risk'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useFocusEffect(
    React.useCallback(() => {
      loadDocuments();
    }, [])
  );

  useEffect(() => {
    filterAndSortDocuments();
  }, [documents, searchQuery, selectedFilter, sortBy, sortOrder]);

  const loadDocuments = async () => {
    try {
      setLoading(true);
      
      const allDocuments = await documentService.getAllDocuments();
      const documentsWithAnalysis: DocumentWithAnalysis[] = [];
      
      for (const doc of allDocuments) {
        const analysis = await analysisService.getAnalysisByDocumentId(doc.id);
        documentsWithAnalysis.push({
          ...doc,
          riskLevel: analysis?.riskLevel,
          hasArbitrationClause: analysis?.hasArbitrationClause,
          confidence: analysis?.confidence,
        });
      }
      
      setDocuments(documentsWithAnalysis);
    } catch (error) {
      console.error('Error loading documents:', error);
      Alert.alert('Error', 'Failed to load document history');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const filterAndSortDocuments = () => {
    let filtered = documents;

    // Apply search filter
    if (searchQuery.trim()) {
      filtered = filtered.filter(doc =>
        doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        doc.extractedText.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply status filter
    switch (selectedFilter) {
      case 'analyzed':
        filtered = filtered.filter(doc => doc.analysisStatus === AnalysisStatus.COMPLETED);
        break;
      case 'pending':
        filtered = filtered.filter(doc => 
          doc.analysisStatus === AnalysisStatus.PENDING || 
          doc.analysisStatus === AnalysisStatus.PROCESSING
        );
        break;
      case 'high_risk':
        filtered = filtered.filter(doc => 
          doc.riskLevel === RiskLevel.HIGH || doc.riskLevel === RiskLevel.CRITICAL
        );
        break;
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (sortBy) {
        case 'date':
          comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
          break;
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'risk':
          const riskOrder = { critical: 4, high: 3, medium: 2, low: 1 };
          const aRisk = a.riskLevel ? riskOrder[a.riskLevel] : 0;
          const bRisk = b.riskLevel ? riskOrder[b.riskLevel] : 0;
          comparison = aRisk - bRisk;
          break;
      }
      
      return sortOrder === 'desc' ? -comparison : comparison;
    });

    setFilteredDocuments(filtered);
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadDocuments();
  };

  const deleteDocument = async (documentId: string) => {
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
              await documentService.deleteDocument(documentId);
              await analysisService.deleteAnalysisByDocumentId(documentId);
              setDocuments(prev => prev.filter(doc => doc.id !== documentId));
            } catch (error) {
              console.error('Error deleting document:', error);
              Alert.alert('Error', 'Failed to delete document');
            }
          },
        },
      ]
    );
  };

  const getRiskColor = (riskLevel?: RiskLevel) => {
    if (!riskLevel) return theme.colors.textSecondary;
    
    switch (riskLevel) {
      case RiskLevel.LOW:
        return theme.colors.success;
      case RiskLevel.MEDIUM:
        return theme.colors.warning;
      case RiskLevel.HIGH:
        return theme.colors.error;
      case RiskLevel.CRITICAL:
        return '#DC2626';
      default:
        return theme.colors.textSecondary;
    }
  };

  const getStatusIcon = (status: AnalysisStatus) => {
    switch (status) {
      case AnalysisStatus.COMPLETED:
        return 'check-circle';
      case AnalysisStatus.PROCESSING:
        return 'hourglass-empty';
      case AnalysisStatus.PENDING:
        return 'schedule';
      case AnalysisStatus.FAILED:
        return 'error';
      default:
        return 'help';
    }
  };

  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const renderFilterModal = () => (
    <Modal
      visible={filterModalVisible}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setFilterModalVisible(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={[styles.modalContent, { backgroundColor: theme.colors.surface }]}>
          <View style={styles.modalHeader}>
            <Text style={[styles.modalTitle, { color: theme.colors.text }]}>
              Filter & Sort
            </Text>
            <TouchableOpacity onPress={() => setFilterModalVisible(false)}>
              <Icon name="close" size={24} color={theme.colors.text} />
            </TouchableOpacity>
          </View>

          <View style={styles.filterSection}>
            <Text style={[styles.filterLabel, { color: theme.colors.text }]}>Filter by:</Text>
            {[
              { key: 'all', label: 'All Documents' },
              { key: 'analyzed', label: 'Analyzed' },
              { key: 'pending', label: 'Pending Analysis' },
              { key: 'high_risk', label: 'High Risk' },
            ].map((filter) => (
              <TouchableOpacity
                key={filter.key}
                style={[
                  styles.filterOption,
                  selectedFilter === filter.key && { backgroundColor: theme.colors.primary + '20' },
                ]}
                onPress={() => setSelectedFilter(filter.key as any)}
              >
                <Text
                  style={[
                    styles.filterOptionText,
                    { color: selectedFilter === filter.key ? theme.colors.primary : theme.colors.text },
                  ]}
                >
                  {filter.label}
                </Text>
                {selectedFilter === filter.key && (
                  <Icon name="check" size={20} color={theme.colors.primary} />
                )}
              </TouchableOpacity>
            ))}
          </View>

          <View style={styles.filterSection}>
            <Text style={[styles.filterLabel, { color: theme.colors.text }]}>Sort by:</Text>
            {[
              { key: 'date', label: 'Date' },
              { key: 'name', label: 'Name' },
              { key: 'risk', label: 'Risk Level' },
            ].map((sort) => (
              <TouchableOpacity
                key={sort.key}
                style={[
                  styles.filterOption,
                  sortBy === sort.key && { backgroundColor: theme.colors.primary + '20' },
                ]}
                onPress={() => setSortBy(sort.key as any)}
              >
                <Text
                  style={[
                    styles.filterOptionText,
                    { color: sortBy === sort.key ? theme.colors.primary : theme.colors.text },
                  ]}
                >
                  {sort.label}
                </Text>
                {sortBy === sort.key && (
                  <TouchableOpacity
                    onPress={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                  >
                    <Icon
                      name={sortOrder === 'asc' ? 'arrow-upward' : 'arrow-downward'}
                      size={20}
                      color={theme.colors.primary}
                    />
                  </TouchableOpacity>
                )}
              </TouchableOpacity>
            ))}
          </View>
        </View>
      </View>
    </Modal>
  );

  const renderDocumentItem = ({ item }: { item: DocumentWithAnalysis }) => (
    <TouchableOpacity
      style={[styles.documentCard, { backgroundColor: theme.colors.surface }]}
      onPress={() => {
        if (item.analysisStatus === AnalysisStatus.COMPLETED) {
          navigation.navigate('Analysis', { documentId: item.id });
        } else {
          navigation.navigate('DocumentDetails', { documentId: item.id });
        }
      }}
    >
      <View style={styles.documentHeader}>
        <View style={styles.documentIcon}>
          <Icon name="description" size={24} color={theme.colors.primary} />
          {item.riskLevel && (
            <View
              style={[
                styles.riskIndicator,
                { backgroundColor: getRiskColor(item.riskLevel) },
              ]}
            />
          )}
        </View>
        <View style={styles.documentInfo}>
          <Text style={[styles.documentName, { color: theme.colors.text }]} numberOfLines={1}>
            {item.name}
          </Text>
          <View style={styles.documentMeta}>
            <Text style={[styles.documentDate, { color: theme.colors.textSecondary }]}>
              {formatDate(item.createdAt)}
            </Text>
            <Text style={[styles.documentSize, { color: theme.colors.textSecondary }]}>
              {formatFileSize(item.size)}
            </Text>
          </View>
        </View>
        <View style={styles.documentActions}>
          <View style={styles.statusContainer}>
            <Icon
              name={getStatusIcon(item.analysisStatus)}
              size={16}
              color={
                item.analysisStatus === AnalysisStatus.COMPLETED
                  ? theme.colors.success
                  : item.analysisStatus === AnalysisStatus.FAILED
                  ? theme.colors.error
                  : theme.colors.warning
              }
            />
          </View>
          <TouchableOpacity
            style={styles.deleteButton}
            onPress={() => deleteDocument(item.id)}
          >
            <Icon name="delete" size={20} color={theme.colors.error} />
          </TouchableOpacity>
        </View>
      </View>
      
      {item.analysisStatus === AnalysisStatus.COMPLETED && item.hasArbitrationClause !== undefined && (
        <View style={styles.analysisPreview}>
          <View style={styles.analysisResults}>
            <Text style={[styles.analysisStatus, { color: theme.colors.textSecondary }]}>
              {item.hasArbitrationClause ? 'Arbitration clause detected' : 'No arbitration clause'}
            </Text>
            {item.confidence && (
              <Text style={[styles.confidenceScore, { color: theme.colors.textSecondary }]}>
                {Math.round(item.confidence * 100)}% confidence
              </Text>
            )}
          </View>
          {item.riskLevel && (
            <View style={[styles.riskBadge, { backgroundColor: getRiskColor(item.riskLevel) }]}>
              <Text style={styles.riskText}>{item.riskLevel.toUpperCase()}</Text>
            </View>
          )}
        </View>
      )}
    </TouchableOpacity>
  );

  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <Icon name="description" size={64} color={theme.colors.textSecondary} />
      <Text style={[styles.emptyStateTitle, { color: theme.colors.text }]}>
        No documents found
      </Text>
      <Text style={[styles.emptyStateText, { color: theme.colors.textSecondary }]}>
        {searchQuery
          ? 'Try adjusting your search or filter criteria'
          : 'Scan your first document to get started'}
      </Text>
    </View>
  );

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    searchContainer: {
      flexDirection: 'row',
      padding: theme.spacing.md,
      backgroundColor: theme.colors.surface,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    searchInput: {
      flex: 1,
      height: 40,
      backgroundColor: theme.colors.background,
      borderRadius: theme.borderRadius.lg,
      paddingHorizontal: theme.spacing.md,
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.text,
      marginRight: theme.spacing.sm,
    },
    filterButton: {
      width: 40,
      height: 40,
      borderRadius: theme.borderRadius.lg,
      backgroundColor: theme.colors.primary,
      justifyContent: 'center',
      alignItems: 'center',
    },
    documentCard: {
      marginHorizontal: theme.spacing.md,
      marginVertical: theme.spacing.xs,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.lg,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    documentHeader: {
      flexDirection: 'row',
      alignItems: 'flex-start',
    },
    documentIcon: {
      width: 40,
      height: 40,
      borderRadius: 20,
      backgroundColor: theme.colors.primary + '20',
      justifyContent: 'center',
      alignItems: 'center',
      marginRight: theme.spacing.md,
      position: 'relative',
    },
    riskIndicator: {
      position: 'absolute',
      top: -2,
      right: -2,
      width: 12,
      height: 12,
      borderRadius: 6,
      borderWidth: 2,
      borderColor: theme.colors.surface,
    },
    documentInfo: {
      flex: 1,
    },
    documentName: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
      marginBottom: theme.spacing.xs,
    },
    documentMeta: {
      flexDirection: 'row',
      justifyContent: 'space-between',
    },
    documentDate: {
      fontSize: theme.typography.fontSize.sm,
    },
    documentSize: {
      fontSize: theme.typography.fontSize.sm,
    },
    documentActions: {
      alignItems: 'center',
    },
    statusContainer: {
      marginBottom: theme.spacing.sm,
    },
    deleteButton: {
      padding: theme.spacing.xs,
    },
    analysisPreview: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginTop: theme.spacing.md,
      paddingTop: theme.spacing.md,
      borderTopWidth: 1,
      borderTopColor: theme.colors.border,
    },
    analysisResults: {
      flex: 1,
    },
    analysisStatus: {
      fontSize: theme.typography.fontSize.sm,
      marginBottom: 2,
    },
    confidenceScore: {
      fontSize: theme.typography.fontSize.xs,
    },
    riskBadge: {
      paddingHorizontal: theme.spacing.sm,
      paddingVertical: theme.spacing.xs,
      borderRadius: theme.borderRadius.sm,
    },
    riskText: {
      color: 'white',
      fontSize: theme.typography.fontSize.xs,
      fontWeight: theme.typography.fontWeight.bold,
    },
    emptyState: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      paddingHorizontal: theme.spacing.xl,
    },
    emptyStateTitle: {
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.bold,
      marginTop: theme.spacing.md,
      marginBottom: theme.spacing.sm,
    },
    emptyStateText: {
      fontSize: theme.typography.fontSize.md,
      textAlign: 'center',
    },
    modalOverlay: {
      flex: 1,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      justifyContent: 'flex-end',
    },
    modalContent: {
      borderTopLeftRadius: theme.borderRadius.xl,
      borderTopRightRadius: theme.borderRadius.xl,
      padding: theme.spacing.lg,
      maxHeight: '80%',
    },
    modalHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: theme.spacing.lg,
    },
    modalTitle: {
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.bold,
    },
    filterSection: {
      marginBottom: theme.spacing.lg,
    },
    filterLabel: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
      marginBottom: theme.spacing.md,
    },
    filterOption: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingVertical: theme.spacing.md,
      paddingHorizontal: theme.spacing.sm,
      borderRadius: theme.borderRadius.lg,
      marginBottom: theme.spacing.xs,
    },
    filterOptionText: {
      fontSize: theme.typography.fontSize.md,
    },
  });

  return (
    <View style={styles.container}>
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchInput}
          placeholder="Search documents..."
          placeholderTextColor={theme.colors.textSecondary}
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
        <TouchableOpacity
          style={styles.filterButton}
          onPress={() => setFilterModalVisible(true)}
        >
          <Icon name="filter-list" size={20} color="white" />
        </TouchableOpacity>
      </View>

      <FlatList
        data={filteredDocuments}
        renderItem={renderDocumentItem}
        keyExtractor={(item) => item.id}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={theme.colors.primary}
            colors={[theme.colors.primary]}
          />
        }
        ListEmptyComponent={renderEmptyState}
        showsVerticalScrollIndicator={false}
      />

      {renderFilterModal()}
    </View>
  );
};

export default HistoryScreen;