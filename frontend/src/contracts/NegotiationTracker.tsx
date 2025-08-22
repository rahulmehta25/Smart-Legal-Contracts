/**
 * NegotiationTracker.tsx - Change Tracking Component
 * 
 * Comprehensive negotiation management with side-by-side comparison,
 * change attribution, comment threads, approval workflows, and audit trails.
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  NegotiationSession,
  NegotiationParticipant,
  ContractChange,
  NegotiationComment,
  NegotiationDeadline,
  ChangeImpact,
  ContractDraft,
  UUID,
  Timestamp
} from './types';

interface NegotiationTrackerProps {
  contractId?: UUID;
  sessionId?: UUID;
  readonly?: boolean;
  showSideBySide?: boolean;
  enableComments?: boolean;
  enableApprovalWorkflow?: boolean;
  currentUserId?: UUID;
  className?: string;
}

interface TrackerState {
  activeView: 'changes' | 'comments' | 'participants' | 'deadlines' | 'approval' | 'audit';
  comparisonMode: 'side-by-side' | 'inline' | 'overlay';
  selectedChangeId: UUID | null;
  filterBy: 'all' | 'pending' | 'accepted' | 'rejected' | 'my-changes';
  sortBy: 'timestamp' | 'author' | 'impact' | 'type';
  showResolvedComments: boolean;
  selectedVersions: {
    original: string;
    compare: string;
  };
  expandedChangeIds: Set<UUID>;
  newComment: {
    targetId: UUID | null;
    targetType: 'clause' | 'section' | 'change' | 'contract';
    content: string;
    parentCommentId?: UUID;
  };
}

interface SessionData {
  session: NegotiationSession | null;
  changes: ContractChange[];
  comments: NegotiationComment[];
  participants: NegotiationParticipant[];
  deadlines: NegotiationDeadline[];
  versions: Array<{
    id: string;
    name: string;
    timestamp: Timestamp;
    author: UUID;
  }>;
}

interface ChangeCardProps {
  change: ContractChange;
  participants: NegotiationParticipant[];
  onApprove?: (changeId: UUID) => void;
  onReject?: (changeId: UUID, reason: string) => void;
  onComment?: (changeId: UUID) => void;
  onExpand?: (changeId: UUID) => void;
  isExpanded: boolean;
  readonly?: boolean;
  currentUserId?: UUID;
}

interface CommentThreadProps {
  comments: NegotiationComment[];
  targetId: UUID;
  targetType: 'clause' | 'section' | 'change' | 'contract';
  participants: NegotiationParticipant[];
  onAddComment?: (content: string, parentId?: UUID) => void;
  onResolveComment?: (commentId: UUID) => void;
  readonly?: boolean;
  currentUserId?: UUID;
}

// Change impact calculation service
const changeImpactService = {
  calculateImpact: (change: ContractChange): ChangeImpact => {
    // Simplified impact calculation
    const baseImpact = {
      riskChange: 0,
      affectedClauses: [],
      complianceImpact: [],
      costImplication: 0,
      timeImplication: 0
    };

    switch (change.changeType) {
      case 'clause_addition':
        baseImpact.riskChange = -5; // Generally reduces risk by adding protection
        baseImpact.timeImplication = 2; // 2 days additional review time
        break;
      case 'clause_removal':
        baseImpact.riskChange = 15; // Increases risk by removing protection
        baseImpact.timeImplication = 1; // 1 day review time
        break;
      case 'clause_modification':
        baseImpact.riskChange = Math.random() * 10 - 5; // Can increase or decrease risk
        baseImpact.timeImplication = 3; // 3 days review time
        break;
      case 'variable_change':
        baseImpact.riskChange = Math.random() * 6 - 3; // Minimal risk change
        baseImpact.timeImplication = 0.5; // Half day review time
        break;
    }

    return baseImpact;
  },

  getChangeTypeIcon: (changeType: string): string => {
    switch (changeType) {
      case 'clause_addition': return '➕';
      case 'clause_removal': return '➖';
      case 'clause_modification': return '✏️';
      case 'variable_change': return '🔧';
      case 'reordering': return '🔄';
      default: return '📝';
    }
  },

  getChangeTypeLabel: (changeType: string): string => {
    switch (changeType) {
      case 'clause_addition': return 'Clause Added';
      case 'clause_removal': return 'Clause Removed';
      case 'clause_modification': return 'Clause Modified';
      case 'variable_change': return 'Variable Changed';
      case 'reordering': return 'Content Reordered';
      default: return 'Unknown Change';
    }
  }
};

const ChangeCard: React.FC<ChangeCardProps> = ({
  change,
  participants,
  onApprove,
  onReject,
  onComment,
  onExpand,
  isExpanded,
  readonly = false,
  currentUserId
}) => {
  const [showRejectReason, setShowRejectReason] = useState(false);
  const [rejectReason, setRejectReason] = useState('');

  const author = participants.find(p => p.userId === change.proposedBy);
  const canApprove = !readonly && currentUserId && 
    participants.find(p => p.userId === currentUserId)?.permissions
      .some(perm => perm.action === 'approve');

  const impact = changeImpactService.calculateImpact(change);
  const changeIcon = changeImpactService.getChangeTypeIcon(change.changeType);
  const changeLabel = changeImpactService.getChangeTypeLabel(change.changeType);

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'accepted': return 'green';
      case 'rejected': return 'red';
      case 'pending': return 'orange';
      default: return 'gray';
    }
  };

  const handleReject = () => {
    if (rejectReason.trim() && onReject) {
      onReject(change.id, rejectReason);
      setShowRejectReason(false);
      setRejectReason('');
    }
  };

  return (
    <div className="change-card" data-change-id={change.id}>
      <div className="change-header">
        <div className="change-info">
          <div className="change-type">
            <span className="change-icon">{changeIcon}</span>
            <span className="change-label">{changeLabel}</span>
          </div>
          
          <div className="change-meta">
            <span className="author">by {author?.userId || 'Unknown'}</span>
            <span className="timestamp">{change.proposedAt.toLocaleString()}</span>
            <span className={`status-badge ${getStatusColor(change.status)}`}>
              {change.status}
            </span>
          </div>
        </div>

        <div className="change-actions">
          {onComment && (
            <button
              onClick={() => onComment(change.id)}
              className="btn-icon"
              title="Add Comment"
            >
              💬
            </button>
          )}
          
          {onExpand && (
            <button
              onClick={() => onExpand(change.id)}
              className="btn-icon"
              title={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          )}
        </div>
      </div>

      <div className="change-content">
        <div className="change-summary">
          {change.reason && (
            <p className="change-reason">{change.reason}</p>
          )}
          
          <div className="change-details">
            <div className="change-diff">
              {change.previousValue && (
                <div className="diff-section">
                  <h5>Previous:</h5>
                  <div className="diff-content removed">
                    {typeof change.previousValue === 'string' 
                      ? change.previousValue 
                      : JSON.stringify(change.previousValue, null, 2)
                    }
                  </div>
                </div>
              )}
              
              <div className="diff-section">
                <h5>Proposed:</h5>
                <div className="diff-content added">
                  {typeof change.newValue === 'string' 
                    ? change.newValue 
                    : JSON.stringify(change.newValue, null, 2)
                  }
                </div>
              </div>
            </div>
          </div>
        </div>

        {isExpanded && (
          <div className="change-expanded">
            <div className="impact-analysis">
              <h5>Impact Analysis</h5>
              <div className="impact-metrics">
                <div className="impact-item">
                  <span className="impact-label">Risk Change:</span>
                  <span className={`impact-value ${impact.riskChange > 0 ? 'negative' : 'positive'}`}>
                    {impact.riskChange > 0 ? '+' : ''}{impact.riskChange.toFixed(1)}%
                  </span>
                </div>
                
                <div className="impact-item">
                  <span className="impact-label">Review Time:</span>
                  <span className="impact-value">
                    {impact.timeImplication} day{impact.timeImplication !== 1 ? 's' : ''}
                  </span>
                </div>
                
                {impact.affectedClauses.length > 0 && (
                  <div className="impact-item">
                    <span className="impact-label">Affected Clauses:</span>
                    <span className="impact-value">{impact.affectedClauses.length}</span>
                  </div>
                )}
              </div>
            </div>

            {change.relatedChanges && change.relatedChanges.length > 0 && (
              <div className="related-changes">
                <h5>Related Changes</h5>
                <div className="related-list">
                  {change.relatedChanges.map(relatedId => (
                    <span key={relatedId} className="related-tag">
                      Change #{relatedId.slice(-6)}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {canApprove && change.status === 'pending' && (
        <div className="change-approval">
          <div className="approval-actions">
            <button
              onClick={() => onApprove?.(change.id)}
              className="btn btn-sm btn-success"
            >
              ✓ Approve
            </button>
            
            <button
              onClick={() => setShowRejectReason(true)}
              className="btn btn-sm btn-danger"
            >
              ✗ Reject
            </button>
          </div>

          {showRejectReason && (
            <div className="reject-reason-section">
              <textarea
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                placeholder="Please provide a reason for rejection..."
                rows={2}
                className="reject-reason-input"
              />
              <div className="reject-actions">
                <button
                  onClick={handleReject}
                  className="btn btn-sm btn-danger"
                  disabled={!rejectReason.trim()}
                >
                  Reject with Reason
                </button>
                <button
                  onClick={() => {
                    setShowRejectReason(false);
                    setRejectReason('');
                  }}
                  className="btn btn-sm btn-outline"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const CommentThread: React.FC<CommentThreadProps> = ({
  comments,
  targetId,
  targetType,
  participants,
  onAddComment,
  onResolveComment,
  readonly = false,
  currentUserId
}) => {
  const [newCommentContent, setNewCommentContent] = useState('');
  const [replyToId, setReplyToId] = useState<UUID | null>(null);

  // Group comments by thread (parent-child relationships)
  const threadedComments = useMemo(() => {
    const commentMap = new Map<UUID, NegotiationComment & { replies: NegotiationComment[] }>();
    const rootComments: (NegotiationComment & { replies: NegotiationComment[] })[] = [];

    // First pass: create comment map with empty replies
    comments.forEach(comment => {
      commentMap.set(comment.id, { ...comment, replies: [] });
    });

    // Second pass: organize into threads
    comments.forEach(comment => {
      const commentWithReplies = commentMap.get(comment.id)!;
      
      if (comment.parentCommentId) {
        const parent = commentMap.get(comment.parentCommentId);
        if (parent) {
          parent.replies.push(commentWithReplies);
        }
      } else {
        rootComments.push(commentWithReplies);
      }
    });

    return rootComments.sort((a, b) => 
      new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
    );
  }, [comments]);

  const handleAddComment = useCallback(() => {
    if (!newCommentContent.trim() || !onAddComment) return;
    
    onAddComment(newCommentContent, replyToId || undefined);
    setNewCommentContent('');
    setReplyToId(null);
  }, [newCommentContent, replyToId, onAddComment]);

  const renderComment = (comment: NegotiationComment & { replies: NegotiationComment[] }, level = 0) => {
    const author = participants.find(p => p.userId === comment.author);
    const canResolve = !readonly && currentUserId && 
      (comment.author === currentUserId || 
       participants.find(p => p.userId === currentUserId)?.permissions
         .some(perm => perm.action === 'approve'));

    return (
      <div key={comment.id} className={`comment-item level-${level}`} data-comment-id={comment.id}>
        <div className="comment-header">
          <div className="comment-author">
            <strong>{author?.userId || 'Unknown'}</strong>
            <span className="comment-timestamp">
              {comment.createdAt.toLocaleString()}
            </span>
          </div>
          
          <div className="comment-actions">
            {!comment.isResolved && !readonly && (
              <button
                onClick={() => setReplyToId(comment.id)}
                className="btn-link"
              >
                Reply
              </button>
            )}
            
            {canResolve && !comment.isResolved && (
              <button
                onClick={() => onResolveComment?.(comment.id)}
                className="btn-link"
              >
                Resolve
              </button>
            )}
            
            {comment.isResolved && (
              <span className="resolved-badge">✓ Resolved</span>
            )}
          </div>
        </div>
        
        <div className="comment-content">
          <p>{comment.content}</p>
          
          {comment.attachments && comment.attachments.length > 0 && (
            <div className="comment-attachments">
              {comment.attachments.map(attachment => (
                <a
                  key={attachment.id}
                  href={attachment.url}
                  className="attachment-link"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  📎 {attachment.filename}
                </a>
              ))}
            </div>
          )}
        </div>
        
        {replyToId === comment.id && (
          <div className="reply-form">
            <textarea
              value={newCommentContent}
              onChange={(e) => setNewCommentContent(e.target.value)}
              placeholder={`Reply to ${author?.userId || 'comment'}...`}
              rows={2}
              className="reply-input"
            />
            <div className="reply-actions">
              <button
                onClick={handleAddComment}
                className="btn btn-sm btn-primary"
                disabled={!newCommentContent.trim()}
              >
                Reply
              </button>
              <button
                onClick={() => {
                  setReplyToId(null);
                  setNewCommentContent('');
                }}
                className="btn btn-sm btn-outline"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
        
        {/* Render replies with increased indentation */}
        {comment.replies.length > 0 && (
          <div className="comment-replies">
            {comment.replies.map(reply => renderComment(reply, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="comment-thread" data-target-id={targetId}>
      <div className="thread-header">
        <h4>Comments ({comments.length})</h4>
        <div className="thread-info">
          <span>Target: {targetType}</span>
          <span>Resolved: {comments.filter(c => c.isResolved).length}</span>
        </div>
      </div>
      
      <div className="comments-list">
        {threadedComments.map(comment => renderComment(comment))}
        
        {threadedComments.length === 0 && (
          <div className="empty-comments">
            <p>No comments yet. Start the discussion!</p>
          </div>
        )}
      </div>
      
      {!readonly && !replyToId && (
        <div className="new-comment-form">
          <textarea
            value={newCommentContent}
            onChange={(e) => setNewCommentContent(e.target.value)}
            placeholder="Add a comment..."
            rows={3}
            className="new-comment-input"
          />
          <div className="comment-form-actions">
            <button
              onClick={handleAddComment}
              className="btn btn-primary"
              disabled={!newCommentContent.trim()}
            >
              Add Comment
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export const NegotiationTracker: React.FC<NegotiationTrackerProps> = ({
  contractId,
  sessionId,
  readonly = false,
  showSideBySide = true,
  enableComments = true,
  enableApprovalWorkflow = true,
  currentUserId,
  className = ''
}) => {
  const [state, setState] = useState<TrackerState>({
    activeView: 'changes',
    comparisonMode: 'side-by-side',
    selectedChangeId: null,
    filterBy: 'all',
    sortBy: 'timestamp',
    showResolvedComments: false,
    selectedVersions: {
      original: 'latest',
      compare: 'previous'
    },
    expandedChangeIds: new Set(),
    newComment: {
      targetId: null,
      targetType: 'contract',
      content: '',
      parentCommentId: undefined
    }
  });

  // Mock data - in real implementation, this would come from API
  const [sessionData, setSessionData] = useState<SessionData>({
    session: null,
    changes: [],
    comments: [],
    participants: [],
    deadlines: [],
    versions: []
  });

  // Load session data
  useEffect(() => {
    if (contractId || sessionId) {
      // Mock data loading
      const mockSession: NegotiationSession = {
        id: sessionId || `session-${contractId}`,
        contractId: contractId || 'contract-1',
        participants: [
          {
            userId: 'user-1',
            partyId: 'party-1',
            role: 'primary',
            permissions: [
              { action: 'edit', scope: 'all' },
              { action: 'approve', scope: 'all' }
            ]
          },
          {
            userId: 'user-2',
            partyId: 'party-2',
            role: 'primary',
            permissions: [
              { action: 'edit', scope: 'all' },
              { action: 'approve', scope: 'all' }
            ]
          }
        ],
        changes: [],
        comments: [],
        status: 'active',
        startedAt: new Date(Date.now() - 86400000), // 1 day ago
        currentRound: 1,
        deadlines: []
      };

      const mockChanges: ContractChange[] = [
        {
          id: 'change-1',
          changeType: 'clause_modification',
          targetId: 'clause-1',
          previousValue: 'The term of this agreement shall be for a period of one (1) year.',
          newValue: 'The term of this agreement shall be for a period of two (2) years.',
          proposedBy: 'user-1',
          proposedAt: new Date(Date.now() - 3600000), // 1 hour ago
          status: 'pending',
          reason: 'Extended term provides better value and stability',
          impact: changeImpactService.calculateImpact({} as ContractChange)
        },
        {
          id: 'change-2',
          changeType: 'clause_addition',
          targetId: 'section-2',
          newValue: 'Force Majeure: Neither party shall be liable for any delay or failure to perform due to circumstances beyond their reasonable control.',
          proposedBy: 'user-2',
          proposedAt: new Date(Date.now() - 1800000), // 30 minutes ago
          status: 'pending',
          reason: 'Adding force majeure protection',
          impact: changeImpactService.calculateImpact({} as ContractChange)
        }
      ];

      const mockComments: NegotiationComment[] = [
        {
          id: 'comment-1',
          targetId: 'change-1',
          targetType: 'change',
          content: 'I think two years is too long. Can we compromise on 18 months?',
          author: 'user-2',
          createdAt: new Date(Date.now() - 1800000),
          isResolved: false
        },
        {
          id: 'comment-2',
          targetId: 'comment-1',
          targetType: 'change',
          content: 'That sounds reasonable. Let me discuss with my team.',
          author: 'user-1',
          createdAt: new Date(Date.now() - 900000),
          isResolved: false,
          parentCommentId: 'comment-1'
        }
      ];

      setSessionData({
        session: mockSession,
        changes: mockChanges,
        comments: mockComments,
        participants: mockSession.participants,
        deadlines: [],
        versions: [
          { id: 'latest', name: 'Current Version', timestamp: new Date(), author: 'user-1' },
          { id: 'previous', name: 'Previous Version', timestamp: new Date(Date.now() - 3600000), author: 'user-2' }
        ]
      });
    }
  }, [contractId, sessionId]);

  // Filter and sort changes
  const filteredChanges = useMemo(() => {
    let filtered = sessionData.changes;

    // Apply filters
    switch (state.filterBy) {
      case 'pending':
        filtered = filtered.filter(change => change.status === 'pending');
        break;
      case 'accepted':
        filtered = filtered.filter(change => change.status === 'accepted');
        break;
      case 'rejected':
        filtered = filtered.filter(change => change.status === 'rejected');
        break;
      case 'my-changes':
        filtered = filtered.filter(change => change.proposedBy === currentUserId);
        break;
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (state.sortBy) {
        case 'timestamp':
          return new Date(b.proposedAt).getTime() - new Date(a.proposedAt).getTime();
        case 'author':
          return a.proposedBy.localeCompare(b.proposedBy);
        case 'impact':
          return (b.impact?.riskChange || 0) - (a.impact?.riskChange || 0);
        case 'type':
          return a.changeType.localeCompare(b.changeType);
        default:
          return 0;
      }
    });

    return filtered;
  }, [sessionData.changes, state.filterBy, state.sortBy, currentUserId]);

  // Handle change approval
  const handleApproveChange = useCallback((changeId: UUID) => {
    setSessionData(prev => ({
      ...prev,
      changes: prev.changes.map(change =>
        change.id === changeId ? { ...change, status: 'accepted' } : change
      )
    }));
  }, []);

  // Handle change rejection
  const handleRejectChange = useCallback((changeId: UUID, reason: string) => {
    setSessionData(prev => ({
      ...prev,
      changes: prev.changes.map(change =>
        change.id === changeId ? { ...change, status: 'rejected', reason } : change
      )
    }));
  }, []);

  // Handle adding comments
  const handleAddComment = useCallback((content: string, parentId?: UUID) => {
    if (!state.newComment.targetId) return;

    const newComment: NegotiationComment = {
      id: `comment-${Date.now()}`,
      targetId: state.newComment.targetId,
      targetType: state.newComment.targetType,
      content,
      author: currentUserId || 'current-user',
      createdAt: new Date(),
      isResolved: false,
      parentCommentId: parentId
    };

    setSessionData(prev => ({
      ...prev,
      comments: [...prev.comments, newComment]
    }));
  }, [state.newComment, currentUserId]);

  // Handle change expansion
  const handleExpandChange = useCallback((changeId: UUID) => {
    setState(prev => {
      const newExpanded = new Set(prev.expandedChangeIds);
      if (newExpanded.has(changeId)) {
        newExpanded.delete(changeId);
      } else {
        newExpanded.add(changeId);
      }
      return { ...prev, expandedChangeIds: newExpanded };
    });
  }, []);

  const pendingChanges = sessionData.changes.filter(change => change.status === 'pending');
  const sessionActive = sessionData.session?.status === 'active';

  return (
    <div className={`negotiation-tracker ${className}`} id="negotiation-tracker-main">
      {/* Header */}
      <div className="tracker-header" id="tracker-header">
        <div className="header-title">
          <h3>Negotiation Tracker</h3>
          {sessionData.session && (
            <div className="session-info">
              <span className={`session-status ${sessionData.session.status}`}>
                {sessionData.session.status.toUpperCase()}
              </span>
              <span>Round {sessionData.session.currentRound}</span>
              <span>{pendingChanges.length} pending changes</span>
            </div>
          )}
        </div>

        <div className="header-actions">
          <select
            value={state.comparisonMode}
            onChange={(e) => setState(prev => ({ 
              ...prev, 
              comparisonMode: e.target.value as any 
            }))}
            className="comparison-mode-select"
          >
            <option value="side-by-side">Side by Side</option>
            <option value="inline">Inline Changes</option>
            <option value="overlay">Overlay Mode</option>
          </select>
          
          <button
            onClick={() => setState(prev => ({ 
              ...prev, 
              showResolvedComments: !prev.showResolvedComments 
            }))}
            className={`btn btn-outline ${state.showResolvedComments ? 'active' : ''}`}
          >
            {state.showResolvedComments ? 'Hide' : 'Show'} Resolved
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="tracker-tabs" id="tracker-tabs">
        {[
          { key: 'changes', label: 'Changes', count: sessionData.changes.length },
          { key: 'comments', label: 'Comments', count: sessionData.comments.length },
          { key: 'participants', label: 'Participants', count: sessionData.participants.length },
          { key: 'deadlines', label: 'Deadlines', count: sessionData.deadlines.length },
          ...(enableApprovalWorkflow ? [{ key: 'approval', label: 'Approval', count: pendingChanges.length }] : []),
          { key: 'audit', label: 'Audit Trail', count: 0 }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setState(prev => ({ ...prev, activeView: tab.key as any }))}
            className={`tab ${state.activeView === tab.key ? 'active' : ''}`}
          >
            {tab.label}
            {tab.count > 0 && <span className="tab-count">({tab.count})</span>}
          </button>
        ))}
      </div>

      {/* Filters and Controls */}
      <div className="tracker-controls" id="tracker-controls">
        <div className="filter-controls">
          <select
            value={state.filterBy}
            onChange={(e) => setState(prev => ({ ...prev, filterBy: e.target.value as any }))}
          >
            <option value="all">All Changes</option>
            <option value="pending">Pending</option>
            <option value="accepted">Accepted</option>
            <option value="rejected">Rejected</option>
            {currentUserId && <option value="my-changes">My Changes</option>}
          </select>

          <select
            value={state.sortBy}
            onChange={(e) => setState(prev => ({ ...prev, sortBy: e.target.value as any }))}
          >
            <option value="timestamp">Sort by Time</option>
            <option value="author">Sort by Author</option>
            <option value="impact">Sort by Impact</option>
            <option value="type">Sort by Type</option>
          </select>
        </div>

        {showSideBySide && (
          <div className="version-selector">
            <label>Compare:</label>
            <select
              value={state.selectedVersions.original}
              onChange={(e) => setState(prev => ({
                ...prev,
                selectedVersions: { ...prev.selectedVersions, original: e.target.value }
              }))}
            >
              {sessionData.versions.map(version => (
                <option key={version.id} value={version.id}>
                  {version.name}
                </option>
              ))}
            </select>
            <span>vs</span>
            <select
              value={state.selectedVersions.compare}
              onChange={(e) => setState(prev => ({
                ...prev,
                selectedVersions: { ...prev.selectedVersions, compare: e.target.value }
              }))}
            >
              {sessionData.versions.map(version => (
                <option key={version.id} value={version.id}>
                  {version.name}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Tab Content */}
      <div className="tab-content" id="tab-content">
        {/* Changes Tab */}
        {state.activeView === 'changes' && (
          <div className="changes-content">
            <div className="changes-list">
              {filteredChanges.map(change => (
                <ChangeCard
                  key={change.id}
                  change={change}
                  participants={sessionData.participants}
                  onApprove={handleApproveChange}
                  onReject={handleRejectChange}
                  onComment={(changeId) => setState(prev => ({
                    ...prev,
                    newComment: { ...prev.newComment, targetId: changeId, targetType: 'change' }
                  }))}
                  onExpand={handleExpandChange}
                  isExpanded={state.expandedChangeIds.has(change.id)}
                  readonly={readonly}
                  currentUserId={currentUserId}
                />
              ))}

              {filteredChanges.length === 0 && (
                <div className="empty-state">
                  <h4>No changes found</h4>
                  <p>Try adjusting your filter criteria.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Comments Tab */}
        {state.activeView === 'comments' && enableComments && (
          <div className="comments-content">
            <CommentThread
              comments={sessionData.comments.filter(comment => 
                state.showResolvedComments || !comment.isResolved
              )}
              targetId={contractId || 'contract'}
              targetType="contract"
              participants={sessionData.participants}
              onAddComment={handleAddComment}
              onResolveComment={(commentId) => {
                setSessionData(prev => ({
                  ...prev,
                  comments: prev.comments.map(comment =>
                    comment.id === commentId ? { ...comment, isResolved: true, resolvedAt: new Date() } : comment
                  )
                }));
              }}
              readonly={readonly}
              currentUserId={currentUserId}
            />
          </div>
        )}

        {/* Participants Tab */}
        {state.activeView === 'participants' && (
          <div className="participants-content">
            <div className="participants-list">
              {sessionData.participants.map(participant => (
                <div key={participant.userId} className="participant-card">
                  <div className="participant-info">
                    <h4>{participant.userId}</h4>
                    <span className="participant-role">{participant.role}</span>
                    <span className="participant-party">Party: {participant.partyId}</span>
                  </div>
                  
                  <div className="participant-permissions">
                    <h5>Permissions:</h5>
                    <div className="permissions-list">
                      {participant.permissions.map((permission, index) => (
                        <span key={index} className="permission-badge">
                          {permission.action} ({permission.scope})
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  {participant.lastActive && (
                    <div className="last-active">
                      Last active: {participant.lastActive.toLocaleString()}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Approval Tab */}
        {state.activeView === 'approval' && enableApprovalWorkflow && (
          <div className="approval-content">
            <div className="approval-summary">
              <h4>Pending Approvals</h4>
              <p>{pendingChanges.length} changes require approval</p>
            </div>
            
            <div className="approval-queue">
              {pendingChanges.map(change => (
                <div key={change.id} className="approval-item">
                  <div className="approval-change-info">
                    <span className="change-type">
                      {changeImpactService.getChangeTypeIcon(change.changeType)}
                      {changeImpactService.getChangeTypeLabel(change.changeType)}
                    </span>
                    <span className="change-author">by {change.proposedBy}</span>
                    <span className="change-time">{change.proposedAt.toLocaleString()}</span>
                  </div>
                  
                  <div className="approval-actions">
                    <button
                      onClick={() => handleApproveChange(change.id)}
                      className="btn btn-sm btn-success"
                      disabled={readonly}
                    >
                      ✓ Approve
                    </button>
                    <button
                      onClick={() => handleRejectChange(change.id, 'Rejected from approval queue')}
                      className="btn btn-sm btn-danger"
                      disabled={readonly}
                    >
                      ✗ Reject
                    </button>
                  </div>
                </div>
              ))}
              
              {pendingChanges.length === 0 && (
                <div className="empty-state">
                  <h4>No pending approvals</h4>
                  <p>All changes have been reviewed.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Audit Trail Tab */}
        {state.activeView === 'audit' && (
          <div className="audit-content">
            <div className="audit-trail">
              <h4>Audit Trail</h4>
              <p>Comprehensive audit trail would be displayed here showing all actions, timestamps, and user attributions.</p>
              
              <div className="audit-entries">
                {sessionData.changes.map(change => (
                  <div key={change.id} className="audit-entry">
                    <div className="audit-timestamp">
                      {change.proposedAt.toLocaleString()}
                    </div>
                    <div className="audit-action">
                      {change.proposedBy} proposed {changeImpactService.getChangeTypeLabel(change.changeType).toLowerCase()}
                    </div>
                    <div className="audit-status">
                      Status: {change.status}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NegotiationTracker;