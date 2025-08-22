"""
Git-like version control system for documents.
Provides versioning, branching, merging, and audit trail functionality.
"""

import hashlib
import json
import os
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import uuid
from collections import defaultdict
import shutil
import tempfile

from .diff_engine import AdvancedDiffEngine, ComparisonResult, DiffResult


class VersionStatus(Enum):
    """Status of a document version."""
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ConflictType(Enum):
    """Types of merge conflicts."""
    CONTENT_CONFLICT = "content_conflict"
    STRUCTURAL_CONFLICT = "structural_conflict"
    SEMANTIC_CONFLICT = "semantic_conflict"


@dataclass
class Author:
    """Represents an author of document changes."""
    name: str
    email: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DocumentVersion:
    """Represents a specific version of a document."""
    version_id: str
    parent_versions: List[str]
    content: str
    content_hash: str
    author: Author
    timestamp: datetime
    message: str
    status: VersionStatus
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()


@dataclass
class DocumentBranch:
    """Represents a branch in the document version tree."""
    branch_name: str
    base_version_id: str
    head_version_id: str
    created_by: Author
    created_at: datetime
    description: str = ""
    is_protected: bool = False


@dataclass
class MergeConflict:
    """Represents a merge conflict between document versions."""
    conflict_type: ConflictType
    base_content: str
    branch_a_content: str
    branch_b_content: str
    position: Tuple[int, int]
    description: str
    resolution_options: List[str]


@dataclass
class MergeResult:
    """Result of a merge operation."""
    success: bool
    merged_content: str
    conflicts: List[MergeConflict]
    merge_version_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentVersionTracker:
    """
    Git-like version control system for documents.
    Manages versions, branches, merges, and audit trails.
    """
    
    def __init__(self, repository_path: Optional[str] = None):
        """
        Initialize version tracker.
        
        Args:
            repository_path: Path to store version data (optional)
        """
        self.repository_path = repository_path or tempfile.mkdtemp(prefix="doc_version_")
        self.versions: Dict[str, DocumentVersion] = {}
        self.branches: Dict[str, DocumentBranch] = {}
        self.current_branch = "main"
        
        # Initialize repository structure
        self._initialize_repository()
        
        # Diff engine for comparisons
        self.diff_engine = AdvancedDiffEngine()
    
    def _initialize_repository(self):
        """Initialize the repository directory structure."""
        os.makedirs(self.repository_path, exist_ok=True)
        os.makedirs(os.path.join(self.repository_path, "objects"), exist_ok=True)
        os.makedirs(os.path.join(self.repository_path, "refs"), exist_ok=True)
        os.makedirs(os.path.join(self.repository_path, "branches"), exist_ok=True)
        
        # Create initial main branch if it doesn't exist
        if "main" not in self.branches:
            self.branches["main"] = DocumentBranch(
                branch_name="main",
                base_version_id="",
                head_version_id="",
                created_by=Author("system", "system@example.com"),
                created_at=datetime.now(),
                description="Main branch"
            )
    
    def create_version(self, content: str, message: str, author: Author,
                      status: VersionStatus = VersionStatus.DRAFT,
                      tags: List[str] = None) -> str:
        """
        Create a new version of the document.
        
        Args:
            content: Document content
            message: Commit message
            author: Author information
            status: Version status
            tags: Optional tags for the version
            
        Returns:
            Version ID of the created version
        """
        version_id = str(uuid.uuid4())
        
        # Get parent versions (current head of branch)
        parent_versions = []
        if self.current_branch in self.branches:
            head_version_id = self.branches[self.current_branch].head_version_id
            if head_version_id:
                parent_versions = [head_version_id]
        
        version = DocumentVersion(
            version_id=version_id,
            parent_versions=parent_versions,
            content=content,
            content_hash="",  # Will be computed in __post_init__
            author=author,
            timestamp=datetime.now(),
            message=message,
            status=status,
            tags=tags or []
        )
        
        # Store version
        self.versions[version_id] = version
        self._persist_version(version)
        
        # Update branch head
        if self.current_branch in self.branches:
            self.branches[self.current_branch].head_version_id = version_id
        
        return version_id
    
    def create_branch(self, branch_name: str, base_version_id: str,
                     author: Author, description: str = "") -> bool:
        """
        Create a new branch from a base version.
        
        Args:
            branch_name: Name of the new branch
            base_version_id: Version ID to branch from
            author: Author creating the branch
            description: Optional description
            
        Returns:
            True if branch created successfully
        """
        if branch_name in self.branches:
            return False  # Branch already exists
        
        if base_version_id and base_version_id not in self.versions:
            return False  # Base version doesn't exist
        
        branch = DocumentBranch(
            branch_name=branch_name,
            base_version_id=base_version_id,
            head_version_id=base_version_id,
            created_by=author,
            created_at=datetime.now(),
            description=description
        )
        
        self.branches[branch_name] = branch
        self._persist_branch(branch)
        
        return True
    
    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to a different branch.
        
        Args:
            branch_name: Name of branch to switch to
            
        Returns:
            True if switch successful
        """
        if branch_name not in self.branches:
            return False
        
        self.current_branch = branch_name
        return True
    
    def merge_branches(self, source_branch: str, target_branch: str,
                      author: Author, message: str = "") -> MergeResult:
        """
        Merge one branch into another.
        
        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            author: Author performing the merge
            message: Merge message
            
        Returns:
            MergeResult with outcome and any conflicts
        """
        if source_branch not in self.branches or target_branch not in self.branches:
            return MergeResult(
                success=False,
                merged_content="",
                conflicts=[],
                merge_version_id="",
                metadata={"error": "Branch not found"}
            )
        
        source_head = self.branches[source_branch].head_version_id
        target_head = self.branches[target_branch].head_version_id
        
        if not source_head or not target_head:
            return MergeResult(
                success=False,
                merged_content="",
                conflicts=[],
                merge_version_id="",
                metadata={"error": "Branch has no commits"}
            )
        
        source_version = self.versions[source_head]
        target_version = self.versions[target_head]
        
        # Find common ancestor
        common_ancestor = self._find_common_ancestor(source_head, target_head)
        
        # Perform three-way merge
        merge_result = self._three_way_merge(
            common_ancestor,
            source_version,
            target_version
        )
        
        if merge_result.success:
            # Create merge commit
            merge_message = message or f"Merge {source_branch} into {target_branch}"
            merge_version_id = self.create_version(
                content=merge_result.merged_content,
                message=merge_message,
                author=author,
                status=VersionStatus.DRAFT
            )
            
            # Update merge version to have two parents
            merge_version = self.versions[merge_version_id]
            merge_version.parent_versions = [target_head, source_head]
            
            merge_result.merge_version_id = merge_version_id
        
        return merge_result
    
    def get_version_history(self, branch_name: str = None) -> List[DocumentVersion]:
        """
        Get version history for a branch.
        
        Args:
            branch_name: Branch to get history for (current branch if None)
            
        Returns:
            List of versions in chronological order
        """
        branch_name = branch_name or self.current_branch
        
        if branch_name not in self.branches:
            return []
        
        head_version_id = self.branches[branch_name].head_version_id
        if not head_version_id:
            return []
        
        # Traverse version tree
        history = []
        visited = set()
        stack = [head_version_id]
        
        while stack:
            version_id = stack.pop()
            if version_id in visited or version_id not in self.versions:
                continue
            
            visited.add(version_id)
            version = self.versions[version_id]
            history.append(version)
            
            # Add parent versions to stack
            stack.extend(version.parent_versions)
        
        # Sort by timestamp (most recent first)
        history.sort(key=lambda v: v.timestamp, reverse=True)
        return history
    
    def compare_versions(self, version_a_id: str, version_b_id: str) -> ComparisonResult:
        """
        Compare two versions of the document.
        
        Args:
            version_a_id: First version ID
            version_b_id: Second version ID
            
        Returns:
            Comparison result
        """
        if version_a_id not in self.versions or version_b_id not in self.versions:
            raise ValueError("Version not found")
        
        version_a = self.versions[version_a_id]
        version_b = self.versions[version_b_id]
        
        return self.diff_engine.compare_documents(version_a.content, version_b.content)
    
    def get_blame_information(self, version_id: str) -> Dict[str, Any]:
        """
        Get blame information showing who changed each line.
        
        Args:
            version_id: Version to get blame for
            
        Returns:
            Blame information mapping lines to authors and versions
        """
        if version_id not in self.versions:
            raise ValueError("Version not found")
        
        version = self.versions[version_id]
        lines = version.content.split('\n')
        
        blame_info = {
            'version_id': version_id,
            'lines': [],
            'authors': {},
            'versions': {}
        }
        
        # For each line, trace back through history to find origin
        history = self.get_version_history()
        
        for line_num, line_content in enumerate(lines):
            # Find the version where this line was last modified
            line_blame = self._trace_line_history(line_content, line_num, history)
            
            blame_info['lines'].append({
                'line_number': line_num + 1,
                'content': line_content,
                'author': line_blame['author'],
                'version_id': line_blame['version_id'],
                'timestamp': line_blame['timestamp']
            })
        
        return blame_info
    
    def rollback_to_version(self, version_id: str, author: Author,
                           message: str = "") -> str:
        """
        Rollback to a previous version.
        
        Args:
            version_id: Version to rollback to
            author: Author performing rollback
            message: Rollback message
            
        Returns:
            ID of the new version created by rollback
        """
        if version_id not in self.versions:
            raise ValueError("Version not found")
        
        target_version = self.versions[version_id]
        rollback_message = message or f"Rollback to version {version_id}"
        
        return self.create_version(
            content=target_version.content,
            message=rollback_message,
            author=author,
            status=VersionStatus.DRAFT,
            tags=["rollback"]
        )
    
    def get_audit_trail(self, start_date: datetime = None, 
                       end_date: datetime = None) -> List[Dict[str, Any]]:
        """
        Get audit trail of all changes.
        
        Args:
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            List of audit events
        """
        audit_events = []
        
        for version in self.versions.values():
            if start_date and version.timestamp < start_date:
                continue
            if end_date and version.timestamp > end_date:
                continue
            
            audit_events.append({
                'event_type': 'version_created',
                'version_id': version.version_id,
                'author': {
                    'name': version.author.name,
                    'email': version.author.email
                },
                'timestamp': version.timestamp.isoformat(),
                'message': version.message,
                'status': version.status.value,
                'content_hash': version.content_hash,
                'parent_versions': version.parent_versions
            })
        
        # Add branch creation events
        for branch in self.branches.values():
            if start_date and branch.created_at < start_date:
                continue
            if end_date and branch.created_at > end_date:
                continue
            
            audit_events.append({
                'event_type': 'branch_created',
                'branch_name': branch.branch_name,
                'base_version_id': branch.base_version_id,
                'author': {
                    'name': branch.created_by.name,
                    'email': branch.created_by.email
                },
                'timestamp': branch.created_at.isoformat(),
                'description': branch.description
            })
        
        # Sort by timestamp
        audit_events.sort(key=lambda e: e['timestamp'])
        return audit_events
    
    def _find_common_ancestor(self, version_a_id: str, version_b_id: str) -> str:
        """Find the common ancestor of two versions."""
        # Get all ancestors of version A
        ancestors_a = self._get_all_ancestors(version_a_id)
        
        # Traverse version B's ancestry until we find a common ancestor
        visited = set()
        queue = [version_b_id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id in ancestors_a:
                return current_id  # Found common ancestor
            
            if current_id in self.versions:
                queue.extend(self.versions[current_id].parent_versions)
        
        return ""  # No common ancestor found
    
    def _get_all_ancestors(self, version_id: str) -> Set[str]:
        """Get all ancestor versions of a given version."""
        ancestors = set()
        queue = [version_id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id in ancestors:
                continue
            
            ancestors.add(current_id)
            
            if current_id in self.versions:
                queue.extend(self.versions[current_id].parent_versions)
        
        return ancestors
    
    def _three_way_merge(self, base_version_id: str,
                        source_version: DocumentVersion,
                        target_version: DocumentVersion) -> MergeResult:
        """Perform three-way merge of document versions."""
        if not base_version_id:
            # No common ancestor - simple two-way merge
            return self._two_way_merge(source_version, target_version)
        
        base_version = self.versions.get(base_version_id)
        if not base_version:
            return self._two_way_merge(source_version, target_version)
        
        # Compare base to source and base to target
        base_to_source = self.diff_engine.compare_documents(
            base_version.content, source_version.content
        )
        base_to_target = self.diff_engine.compare_documents(
            base_version.content, target_version.content
        )
        
        conflicts = []
        merged_content = base_version.content
        
        # Simple merge strategy - this could be enhanced
        source_changes = {(d.old_position, d.new_position): d 
                         for d in base_to_source.differences}
        target_changes = {(d.old_position, d.new_position): d 
                         for d in base_to_target.differences}
        
        # Check for overlapping changes (conflicts)
        for source_pos, source_diff in source_changes.items():
            for target_pos, target_diff in target_changes.items():
                if self._positions_overlap(source_pos[0], target_pos[0]):
                    conflict = MergeConflict(
                        conflict_type=ConflictType.CONTENT_CONFLICT,
                        base_content=source_diff.old_content,
                        branch_a_content=source_diff.new_content,
                        branch_b_content=target_diff.new_content,
                        position=source_pos[0],
                        description="Overlapping changes detected",
                        resolution_options=["keep_source", "keep_target", "manual_merge"]
                    )
                    conflicts.append(conflict)
        
        if conflicts:
            return MergeResult(
                success=False,
                merged_content="",
                conflicts=conflicts,
                merge_version_id=""
            )
        
        # Apply changes (simplified - in practice would be more complex)
        # For now, apply target changes to base
        merged_content = target_version.content
        
        return MergeResult(
            success=True,
            merged_content=merged_content,
            conflicts=[],
            merge_version_id="",
            metadata={
                'base_version': base_version_id,
                'source_changes': len(base_to_source.differences),
                'target_changes': len(base_to_target.differences)
            }
        )
    
    def _two_way_merge(self, version_a: DocumentVersion,
                      version_b: DocumentVersion) -> MergeResult:
        """Perform simple two-way merge."""
        # For simplicity, just use version B's content
        # In practice, this would be more sophisticated
        return MergeResult(
            success=True,
            merged_content=version_b.content,
            conflicts=[],
            merge_version_id=""
        )
    
    def _positions_overlap(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> bool:
        """Check if two position ranges overlap."""
        return not (pos_a[1] <= pos_b[0] or pos_b[1] <= pos_a[0])
    
    def _trace_line_history(self, line_content: str, line_num: int,
                           history: List[DocumentVersion]) -> Dict[str, Any]:
        """Trace the history of a specific line."""
        # Simplified implementation - returns the most recent version info
        if history:
            latest_version = history[0]
            return {
                'author': latest_version.author.name,
                'version_id': latest_version.version_id,
                'timestamp': latest_version.timestamp.isoformat()
            }
        
        return {
            'author': 'unknown',
            'version_id': '',
            'timestamp': datetime.now().isoformat()
        }
    
    def _persist_version(self, version: DocumentVersion):
        """Persist version data to disk."""
        version_path = os.path.join(
            self.repository_path, "objects", version.version_id + ".json"
        )
        
        version_data = asdict(version)
        version_data['timestamp'] = version.timestamp.isoformat()
        version_data['author']['timestamp'] = version.author.timestamp.isoformat()
        
        with open(version_path, 'w') as f:
            json.dump(version_data, f, indent=2)
    
    def _persist_branch(self, branch: DocumentBranch):
        """Persist branch data to disk."""
        branch_path = os.path.join(
            self.repository_path, "branches", branch.branch_name + ".json"
        )
        
        branch_data = asdict(branch)
        branch_data['created_at'] = branch.created_at.isoformat()
        branch_data['created_by']['timestamp'] = branch.created_by.timestamp.isoformat()
        
        with open(branch_path, 'w') as f:
            json.dump(branch_data, f, indent=2)
    
    def load_repository(self):
        """Load repository data from disk."""
        # Load versions
        objects_path = os.path.join(self.repository_path, "objects")
        if os.path.exists(objects_path):
            for filename in os.listdir(objects_path):
                if filename.endswith(".json"):
                    with open(os.path.join(objects_path, filename), 'r') as f:
                        version_data = json.load(f)
                        
                        # Convert timestamps back to datetime objects
                        version_data['timestamp'] = datetime.fromisoformat(
                            version_data['timestamp']
                        )
                        version_data['author']['timestamp'] = datetime.fromisoformat(
                            version_data['author']['timestamp']
                        )
                        
                        # Recreate objects
                        author = Author(**version_data['author'])
                        version_data['author'] = author
                        version_data['status'] = VersionStatus(version_data['status'])
                        
                        version = DocumentVersion(**version_data)
                        self.versions[version.version_id] = version
        
        # Load branches
        branches_path = os.path.join(self.repository_path, "branches")
        if os.path.exists(branches_path):
            for filename in os.listdir(branches_path):
                if filename.endswith(".json"):
                    with open(os.path.join(branches_path, filename), 'r') as f:
                        branch_data = json.load(f)
                        
                        # Convert timestamps
                        branch_data['created_at'] = datetime.fromisoformat(
                            branch_data['created_at']
                        )
                        branch_data['created_by']['timestamp'] = datetime.fromisoformat(
                            branch_data['created_by']['timestamp']
                        )
                        
                        # Recreate objects
                        author = Author(**branch_data['created_by'])
                        branch_data['created_by'] = author
                        
                        branch = DocumentBranch(**branch_data)
                        self.branches[branch.branch_name] = branch


# Utility functions

def create_document_repository(repository_path: str) -> DocumentVersionTracker:
    """
    Create a new document repository.
    
    Args:
        repository_path: Path where repository should be created
        
    Returns:
        DocumentVersionTracker instance
    """
    tracker = DocumentVersionTracker(repository_path)
    return tracker


def load_document_repository(repository_path: str) -> DocumentVersionTracker:
    """
    Load an existing document repository.
    
    Args:
        repository_path: Path to existing repository
        
    Returns:
        DocumentVersionTracker instance with loaded data
    """
    tracker = DocumentVersionTracker(repository_path)
    tracker.load_repository()
    return tracker