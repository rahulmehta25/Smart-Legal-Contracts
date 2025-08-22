import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  RefreshControl,
  Alert,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useTheme } from '@hooks/useTheme';
import { Document, ArbitrationAnalysis, RiskLevel } from '@types/index';
import { documentService } from '@services/documentService';
import { analysisService } from '@services/analysisService';

const { width } = Dimensions.get('window');

interface DashboardStats {
  totalDocuments: number;
  analyzedDocuments: number;
  highRiskDocuments: number;
  pendingAnalyses: number;
}

const HomeScreen: React.FC = () => {
  const { theme } = useTheme();
  const [stats, setStats] = useState<DashboardStats>({
    totalDocuments: 0,
    analyzedDocuments: 0,
    highRiskDocuments: 0,
    pendingAnalyses: 0,
  });
  const [recentDocuments, setRecentDocuments] = useState<Document[]>([]);
  const [recentAnalyses, setRecentAnalyses] = useState<ArbitrationAnalysis[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    React.useCallback(() => {
      loadDashboardData();
    }, [])
  );

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load statistics
      const documents = await documentService.getAllDocuments();
      const analyses = await analysisService.getAllAnalyses();
      
      const dashboardStats: DashboardStats = {
        totalDocuments: documents.length,
        analyzedDocuments: analyses.length,
        highRiskDocuments: analyses.filter(a => 
          a.riskLevel === RiskLevel.HIGH || a.riskLevel === RiskLevel.CRITICAL
        ).length,
        pendingAnalyses: documents.filter(d => 
          d.analysisStatus === 'pending' || d.analysisStatus === 'processing'
        ).length,
      };
      
      setStats(dashboardStats);
      
      // Load recent documents (last 5)
      const sortedDocuments = documents
        .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
        .slice(0, 5);
      setRecentDocuments(sortedDocuments);
      
      // Load recent analyses (last 5)
      const sortedAnalyses = analyses
        .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
        .slice(0, 5);
      setRecentAnalyses(sortedAnalyses);
      
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      Alert.alert('Error', 'Failed to load dashboard data. Please try again.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadDashboardData();
  };

  const getRiskColor = (riskLevel: RiskLevel) => {
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

  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const StatCard: React.FC<{
    title: string;
    value: number;
    icon: string;
    color: string;
  }> = ({ title, value, icon, color }) => (
    <View style={[styles.statCard, { backgroundColor: theme.colors.surface }]}>
      <View style={[styles.statIcon, { backgroundColor: color + '20' }]}>
        <Icon name={icon} size={24} color={color} />
      </View>
      <Text style={[styles.statValue, { color: theme.colors.text }]}>
        {value}
      </Text>
      <Text style={[styles.statTitle, { color: theme.colors.textSecondary }]}>
        {title}
      </Text>
    </View>
  );

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    scrollContent: {
      padding: theme.spacing.md,
    },
    welcomeSection: {
      marginBottom: theme.spacing.lg,
    },
    welcomeTitle: {
      fontSize: theme.typography.fontSize.xl,
      fontWeight: theme.typography.fontWeight.bold,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    welcomeSubtitle: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.textSecondary,
    },
    statsContainer: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      justifyContent: 'space-between',
      marginBottom: theme.spacing.lg,
    },
    statCard: {
      width: (width - theme.spacing.md * 3) / 2,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.lg,
      marginBottom: theme.spacing.sm,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    statIcon: {
      width: 48,
      height: 48,
      borderRadius: 24,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.sm,
    },
    statValue: {
      fontSize: theme.typography.fontSize.xl,
      fontWeight: theme.typography.fontWeight.bold,
      marginBottom: theme.spacing.xs,
    },
    statTitle: {
      fontSize: theme.typography.fontSize.sm,
      fontWeight: theme.typography.fontWeight.medium,
    },
    sectionTitle: {
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.bold,
      color: theme.colors.text,
      marginBottom: theme.spacing.md,
    },
    documentCard: {
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.lg,
      marginBottom: theme.spacing.sm,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    documentHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      marginBottom: theme.spacing.xs,
    },
    documentName: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
      color: theme.colors.text,
      flex: 1,
      marginRight: theme.spacing.sm,
    },
    documentStatus: {
      fontSize: theme.typography.fontSize.sm,
      fontWeight: theme.typography.fontWeight.medium,
      paddingHorizontal: theme.spacing.xs,
      paddingVertical: 2,
      borderRadius: theme.borderRadius.sm,
    },
    documentDate: {
      fontSize: theme.typography.fontSize.sm,
      color: theme.colors.textSecondary,
    },
    analysisCard: {
      backgroundColor: theme.colors.surface,
      padding: theme.spacing.md,
      borderRadius: theme.borderRadius.lg,
      marginBottom: theme.spacing.sm,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    analysisHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: theme.spacing.xs,
    },
    analysisResult: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
      color: theme.colors.text,
    },
    riskBadge: {
      paddingHorizontal: theme.spacing.sm,
      paddingVertical: theme.spacing.xs,
      borderRadius: theme.borderRadius.sm,
    },
    riskText: {
      fontSize: theme.typography.fontSize.sm,
      fontWeight: theme.typography.fontWeight.medium,
      color: 'white',
    },
    analysisDetails: {
      fontSize: theme.typography.fontSize.sm,
      color: theme.colors.textSecondary,
    },
    emptyState: {
      alignItems: 'center',
      padding: theme.spacing.xl,
    },
    emptyStateIcon: {
      marginBottom: theme.spacing.md,
    },
    emptyStateText: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.textSecondary,
      textAlign: 'center',
    },
  });

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={theme.colors.primary}
            colors={[theme.colors.primary]}
          />
        }
      >
        {/* Welcome Section */}
        <View style={styles.welcomeSection}>
          <Text style={styles.welcomeTitle}>Welcome back!</Text>
          <Text style={styles.welcomeSubtitle}>
            Here's what's happening with your documents
          </Text>
        </View>

        {/* Statistics Cards */}
        <View style={styles.statsContainer}>
          <StatCard
            title="Total Documents"
            value={stats.totalDocuments}
            icon="description"
            color={theme.colors.primary}
          />
          <StatCard
            title="Analyzed"
            value={stats.analyzedDocuments}
            icon="analytics"
            color={theme.colors.success}
          />
          <StatCard
            title="High Risk"
            value={stats.highRiskDocuments}
            icon="warning"
            color={theme.colors.error}
          />
          <StatCard
            title="Pending"
            value={stats.pendingAnalyses}
            icon="schedule"
            color={theme.colors.warning}
          />
        </View>

        {/* Recent Documents */}
        <Text style={styles.sectionTitle}>Recent Documents</Text>
        {recentDocuments.length > 0 ? (
          recentDocuments.map((document) => (
            <TouchableOpacity
              key={document.id}
              style={styles.documentCard}
              onPress={() => {
                // Navigate to document details
              }}
            >
              <View style={styles.documentHeader}>
                <Text style={styles.documentName} numberOfLines={2}>
                  {document.name}
                </Text>
                <Text
                  style={[
                    styles.documentStatus,
                    {
                      backgroundColor:
                        document.analysisStatus === 'completed'
                          ? theme.colors.success + '20'
                          : document.analysisStatus === 'processing'
                          ? theme.colors.warning + '20'
                          : theme.colors.textSecondary + '20',
                      color:
                        document.analysisStatus === 'completed'
                          ? theme.colors.success
                          : document.analysisStatus === 'processing'
                          ? theme.colors.warning
                          : theme.colors.textSecondary,
                    },
                  ]}
                >
                  {document.analysisStatus}
                </Text>
              </View>
              <Text style={styles.documentDate}>
                {formatDate(document.createdAt)}
              </Text>
            </TouchableOpacity>
          ))
        ) : (
          <View style={styles.emptyState}>
            <Icon
              name="description"
              size={48}
              color={theme.colors.textSecondary}
              style={styles.emptyStateIcon}
            />
            <Text style={styles.emptyStateText}>
              No documents yet. Scan your first document to get started!
            </Text>
          </View>
        )}

        {/* Recent Analyses */}
        <Text style={styles.sectionTitle}>Recent Analyses</Text>
        {recentAnalyses.length > 0 ? (
          recentAnalyses.map((analysis) => (
            <TouchableOpacity
              key={analysis.id}
              style={styles.analysisCard}
              onPress={() => {
                // Navigate to analysis details
              }}
            >
              <View style={styles.analysisHeader}>
                <Text style={styles.analysisResult}>
                  {analysis.hasArbitrationClause
                    ? 'Arbitration Clause Detected'
                    : 'No Arbitration Clause'}
                </Text>
                <View
                  style={[
                    styles.riskBadge,
                    { backgroundColor: getRiskColor(analysis.riskLevel) },
                  ]}
                >
                  <Text style={styles.riskText}>
                    {analysis.riskLevel.toUpperCase()}
                  </Text>
                </View>
              </View>
              <Text style={styles.analysisDetails}>
                Confidence: {Math.round(analysis.confidence * 100)}% •{' '}
                {formatDate(analysis.createdAt)}
              </Text>
            </TouchableOpacity>
          ))
        ) : (
          <View style={styles.emptyState}>
            <Icon
              name="analytics"
              size={48}
              color={theme.colors.textSecondary}
              style={styles.emptyStateIcon}
            />
            <Text style={styles.emptyStateText}>
              No analyses yet. Scan and analyze documents to see results here.
            </Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
};

export default HomeScreen;