import React, { useState, useEffect } from 'react';
import { MagnifyingGlassIcon, FunnelIcon, ArrowDownTrayIcon, PlusIcon, EyeIcon, PencilIcon, TrashIcon } from '@heroicons/react/24/outline';
import AdminLayout from '../../src/components/admin/AdminLayout';
import UserTable from '../../src/components/admin/UserTable';
import UserModal from '../../src/components/admin/UserModal';
import FilterPanel from '../../src/components/admin/FilterPanel';

interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'enterprise';
  status: 'active' | 'inactive' | 'suspended';
  lastLogin: string;
  createdAt: string;
  documentsProcessed: number;
  subscription: 'free' | 'pro' | 'enterprise';
}

const UsersPage: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [isUserModalOpen, setIsUserModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedRole, setSelectedRole] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalUsers, setTotalUsers] = useState(0);
  const usersPerPage = 20;

  useEffect(() => {
    fetchUsers();
  }, [currentPage, searchQuery, selectedRole, selectedStatus]);

  const fetchUsers = async () => {
    setIsLoading(true);
    // Mock API call - replace with actual implementation
    setTimeout(() => {
      const mockUsers: User[] = Array.from({ length: 100 }, (_, i) => ({
        id: `user-${i + 1}`,
        name: `User ${i + 1}`,
        email: `user${i + 1}@example.com`,
        role: ['admin', 'user', 'enterprise'][i % 3] as User['role'],
        status: ['active', 'inactive', 'suspended'][i % 3] as User['status'],
        lastLogin: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
        createdAt: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
        documentsProcessed: Math.floor(Math.random() * 1000),
        subscription: ['free', 'pro', 'enterprise'][i % 3] as User['subscription']
      }));

      // Apply filters
      let filteredUsers = mockUsers.filter(user => {
        const matchesSearch = user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                             user.email.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesRole = selectedRole === 'all' || user.role === selectedRole;
        const matchesStatus = selectedStatus === 'all' || user.status === selectedStatus;
        
        return matchesSearch && matchesRole && matchesStatus;
      });

      const startIndex = (currentPage - 1) * usersPerPage;
      const paginatedUsers = filteredUsers.slice(startIndex, startIndex + usersPerPage);

      setUsers(paginatedUsers);
      setTotalUsers(filteredUsers.length);
      setIsLoading(false);
    }, 500);
  };

  const handleUserAction = (action: 'view' | 'edit' | 'delete' | 'impersonate', user: User) => {
    switch (action) {
      case 'view':
      case 'edit':
        setSelectedUser(user);
        setIsUserModalOpen(true);
        break;
      case 'delete':
        if (confirm('Are you sure you want to delete this user?')) {
          // Handle delete
          console.log('Deleting user:', user.id);
        }
        break;
      case 'impersonate':
        if (confirm(`Impersonate user ${user.name}?`)) {
          // Handle impersonation
          console.log('Impersonating user:', user.id);
        }
        break;
    }
  };

  const handleExport = () => {
    // Export users data
    console.log('Exporting users data...');
  };

  const totalPages = Math.ceil(totalUsers / usersPerPage);

  return (
    <AdminLayout>
      <div className="space-y-6" id="users-page-container">
        {/* Header */}
        <div className="sm:flex sm:items-center sm:justify-between" id="users-header">
          <div id="users-title-section">
            <h1 className="text-2xl font-semibold text-gray-900" id="users-title">
              User Management
            </h1>
            <p className="mt-1 text-sm text-gray-500" id="users-subtitle">
              Manage user accounts, permissions, and access
            </p>
          </div>
          <div className="mt-4 flex space-x-3 sm:mt-0" id="users-actions">
            <button
              onClick={handleExport}
              className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              id="export-users-btn"
            >
              <ArrowDownTrayIcon className="h-4 w-4 mr-2" id="export-icon" />
              Export
            </button>
            <button
              onClick={() => setIsUserModalOpen(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
              id="add-user-btn"
            >
              <PlusIcon className="h-4 w-4 mr-2" id="add-user-icon" />
              Add User
            </button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="bg-white rounded-lg border border-gray-200 p-6" id="search-filters-section">
          <div className="flex flex-col sm:flex-row gap-4" id="search-filters-content">
            {/* Search Bar */}
            <div className="flex-1" id="search-bar-container">
              <div className="relative" id="search-input-wrapper">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" id="search-icon" />
                <input
                  type="text"
                  placeholder="Search users by name or email..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  id="search-input"
                />
              </div>
            </div>

            {/* Quick Filters */}
            <div className="flex space-x-3" id="quick-filters">
              <select
                value={selectedRole}
                onChange={(e) => setSelectedRole(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-blue-500 focus:border-blue-500"
                id="role-filter"
              >
                <option value="all">All Roles</option>
                <option value="admin">Admin</option>
                <option value="user">User</option>
                <option value="enterprise">Enterprise</option>
              </select>

              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-blue-500 focus:border-blue-500"
                id="status-filter"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
                <option value="suspended">Suspended</option>
              </select>

              <button
                onClick={() => setIsFilterOpen(!isFilterOpen)}
                className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                id="advanced-filter-btn"
              >
                <FunnelIcon className="h-4 w-4 mr-1" id="filter-icon" />
                Filters
              </button>
            </div>
          </div>

          {/* Advanced Filters Panel */}
          {isFilterOpen && (
            <div className="mt-4 pt-4 border-t border-gray-200" id="advanced-filters-panel">
              <FilterPanel
                onClose={() => setIsFilterOpen(false)}
                onApply={(filters) => {
                  console.log('Applied filters:', filters);
                  setIsFilterOpen(false);
                }}
              />
            </div>
          )}
        </div>

        {/* Users Table */}
        <div className="bg-white rounded-lg border border-gray-200" id="users-table-container">
          <UserTable
            users={users}
            isLoading={isLoading}
            onUserAction={handleUserAction}
          />

          {/* Pagination */}
          <div className="px-6 py-4 border-t border-gray-200" id="pagination-container">
            <div className="flex items-center justify-between" id="pagination-content">
              <div className="text-sm text-gray-700" id="pagination-info">
                Showing {((currentPage - 1) * usersPerPage) + 1} to {Math.min(currentPage * usersPerPage, totalUsers)} of {totalUsers} users
              </div>
              <div className="flex space-x-2" id="pagination-buttons">
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  id="prev-page-btn"
                >
                  Previous
                </button>
                
                {/* Page Numbers */}
                <div className="hidden sm:flex space-x-1" id="page-numbers">
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    const pageNum = currentPage <= 3 ? i + 1 : currentPage - 2 + i;
                    if (pageNum > totalPages) return null;
                    
                    return (
                      <button
                        key={pageNum}
                        onClick={() => setCurrentPage(pageNum)}
                        className={`px-3 py-2 rounded-md text-sm font-medium ${
                          pageNum === currentPage
                            ? 'bg-blue-600 text-white'
                            : 'border border-gray-300 text-gray-700 bg-white hover:bg-gray-50'
                        }`}
                        id={`page-${pageNum}-btn`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                </div>

                <button
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  id="next-page-btn"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* User Modal */}
        {isUserModalOpen && (
          <UserModal
            user={selectedUser}
            isOpen={isUserModalOpen}
            onClose={() => {
              setIsUserModalOpen(false);
              setSelectedUser(null);
            }}
            onSave={(userData) => {
              // Handle save
              console.log('Saving user:', userData);
              setIsUserModalOpen(false);
              setSelectedUser(null);
              fetchUsers();
            }}
          />
        )}
      </div>
    </AdminLayout>
  );
};

export default UsersPage;