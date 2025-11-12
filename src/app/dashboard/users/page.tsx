'use client'

import { useState, useEffect } from 'react'
import DashboardLayout from '@/components/DashboardLayout'
import { 
  PlusIcon, 
  MagnifyingGlassIcon,
  PencilIcon,
  TrashIcon,
  UserCircleIcon,
  XMarkIcon
} from '@heroicons/react/24/outline'

interface User {
  _id: string
  ssoId: string
  name: string
  email: string
  role: string
  skills: Array<{ skill: string; level: number }>
  experienceYears: number
  currentWorkload: number
  isAvailable: boolean
  department?: string
}

interface UserModalProps {
  user: User | null
  isOpen: boolean
  onClose: () => void
  onSave: (user: Partial<User>) => void
  isCreating: boolean
  saveLoading: boolean
}

const roles = ['Software Engineer', 'DevOps Engineer', 'Data Scientist', 'QA Engineer', 'Project Manager', 'System Analyst']
const departments = ['Engineering', 'Data Science', 'Quality Assurance', 'Operations', 'Management']

function UserModal({ user, isOpen, onClose, onSave, isCreating, saveLoading }: UserModalProps) {
  const [formData, setFormData] = useState<Partial<User>>({
    ssoId: '',
    name: '',
    email: '',
    role: 'Software Engineer',
    skills: [],
    experienceYears: 0,
    currentWorkload: 0,
    isAvailable: true,
    department: ''
  })

  const [newSkill, setNewSkill] = useState({ skill: '', level: 5 })

  useEffect(() => {
    if (user) {
      setFormData(user)
    } else {
      setFormData({
        ssoId: '',
        name: '',
        email: '',
        role: 'Software Engineer',
        skills: [],
        experienceYears: 0,
        currentWorkload: 0,
        isAvailable: true,
        department: ''
      })
    }
  }, [user])

  const addSkill = () => {
    if (newSkill.skill.trim()) {
      setFormData(prev => ({
        ...prev,
        skills: [...(prev.skills || []), { ...newSkill }]
      }))
      setNewSkill({ skill: '', level: 5 })
    }
  }

  const removeSkill = (index: number) => {
    setFormData(prev => ({
      ...prev,
      skills: prev.skills?.filter((_, i) => i !== index) || []
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSave(formData)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">
            {isCreating ? 'Add New User' : 'Edit User'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                SSO ID *
              </label>
              <input
                type="text"
                required
                value={formData.ssoId || ''}
                onChange={(e) => setFormData(prev => ({ ...prev, ssoId: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                placeholder="emp001"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Full Name *
              </label>
              <input
                type="text"
                required
                value={formData.name || ''}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                placeholder="John Doe"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Email *
            </label>
            <input
              type="email"
              required
              value={formData.email || ''}
              onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
              placeholder="john.doe@netapp.com"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Role *
              </label>
              <select
                required
                value={formData.role || ''}
                onChange={(e) => setFormData(prev => ({ ...prev, role: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
              >
                {roles.map(role => (
                  <option key={role} value={role}>
                    {role.charAt(0).toUpperCase() + role.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Experience (Years) <span className="text-gray-500 text-xs">(0-50)</span>
              </label>
              <input
                type="number"
                min="0"
                max="50"
                step="1"
                value={formData.experienceYears || ''}
                onChange={(e) => {
                  const value = e.target.value === '' ? 0 : Math.max(0, Math.min(50, parseInt(e.target.value) || 0))
                  setFormData(prev => ({ ...prev, experienceYears: value }))
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                placeholder="Enter years of experience"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Current Workload (%) <span className="text-gray-500 text-xs">(0-100)</span>
              </label>
              <input
                type="number"
                min="0"
                max="100"
                step="1"
                value={formData.currentWorkload || ''}
                onChange={(e) => {
                  const value = e.target.value === '' ? 0 : Math.max(0, Math.min(100, parseInt(e.target.value) || 0))
                  setFormData(prev => ({ ...prev, currentWorkload: value }))
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                placeholder="Enter current workload percentage"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Department *
            </label>
            <select
              required
              value={formData.department || ''}
              onChange={(e) => setFormData(prev => ({ ...prev, department: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
            >
              <option value="">Select Department</option>
              {departments.map(dept => (
                <option key={dept} value={dept}>
                  {dept}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData.isAvailable || false}
                onChange={(e) => setFormData(prev => ({ ...prev, isAvailable: e.target.checked }))}
                className="mr-2 h-4 w-4 text-netapp-primary focus:ring-netapp-primary border-gray-300 rounded"
              />
              <span className="text-sm font-medium text-gray-700">Available for assignments</span>
            </label>
          </div>

          {/* Skills Section */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Skills
            </label>
            
            {/* Add Skill */}
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={newSkill.skill}
                onChange={(e) => setNewSkill(prev => ({ ...prev, skill: e.target.value }))}
                placeholder="Skill name (e.g., Python, React)"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
              />
              <input
                type="number"
                min="1"
                max="10"
                value={newSkill.level}
                onChange={(e) => setNewSkill(prev => ({ ...prev, level: parseInt(e.target.value) }))}
                className="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
              />
              <button
                type="button"
                onClick={addSkill}
                className="px-4 py-2 bg-netapp-primary text-white rounded-md hover:bg-netapp-dark"
              >
                Add
              </button>
            </div>

            {/* Skills List */}
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {formData.skills?.map((skill, index) => (
                <div key={index} className="flex items-center justify-between bg-gray-50 px-3 py-2 rounded-lg">
                  <span className="text-sm font-medium">{skill.skill}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Level {skill.level}/10</span>
                    <button
                      type="button"
                      onClick={() => removeSkill(index)}
                      className="text-red-600 hover:text-red-800"
                    >
                      <XMarkIcon className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex justify-end space-x-3 pt-6 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saveLoading}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              {saveLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  {isCreating ? 'Creating...' : 'Saving...'}
                </>
              ) : (
                isCreating ? 'Create User' : 'Save Changes'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function UsersPage() {
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [viewMode, setViewMode] = useState<'cards' | 'list'>('cards')
  const [saveLoading, setSaveLoading] = useState(false)
  const [deleteLoading, setDeleteLoading] = useState<string | null>(null)
  const [importLoading, setImportLoading] = useState(false)

  useEffect(() => {
    fetchUsers()
  }, [])

  const fetchUsers = async () => {
    try {
      const response = await fetch('/api/users')
      const result = await response.json()
      if (result.success) {
        setUsers(result.data)
      }
    } catch (error) {
      console.error('Failed to fetch users:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreateUser = () => {
    setSelectedUser(null)
    setIsCreating(true)
    setIsModalOpen(true)
  }

  const handleEditUser = (user: User) => {
    setSelectedUser(user)
    setIsCreating(false)
    setIsModalOpen(true)
  }

  const handleSaveUser = async (userData: Partial<User>) => {
    setSaveLoading(true)
    try {
      const url = isCreating ? '/api/users' : `/api/users/${selectedUser?._id}`
      const method = isCreating ? 'POST' : 'PUT'
      
      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData)
      })

      const result = await response.json()
      if (result.success) {
        await fetchUsers()
        setIsModalOpen(false)
        alert(isCreating ? 'User created successfully!' : 'User updated successfully!')
      } else {
        alert(`Failed to save user: ${result.error}`)
      }
    } catch (error) {
      console.error('Failed to save user:', error)
      alert('Failed to save user due to network error')
    } finally {
      setSaveLoading(false)
    }
  }

  const handleDeleteUser = async (userId: string) => {
    if (confirm('Are you sure you want to delete this user?')) {
      setDeleteLoading(userId)
      try {
        const response = await fetch(`/api/users/${userId}`, { method: 'DELETE' })
        const result = await response.json()
        if (result.success) {
          await fetchUsers()
          alert('User deleted successfully!')
        } else {
          alert(`Failed to delete user: ${result.error}`)
        }
      } catch (error) {
        console.error('Failed to delete user:', error)
        alert('Failed to delete user due to network error')
      } finally {
        setDeleteLoading(null)
      }
    }
  }

  const handleImportUsers = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      setImportLoading(true)
      const response = await fetch('/api/users/import', {
        method: 'POST',
        body: formData
      })

      const result = await response.json()
      
      if (result.success) {
        alert(`Successfully imported ${result.data.imported} users!`)
        await fetchUsers()
      } else {
        alert(`Import failed: ${result.error}`)
      }
    } catch (error) {
      console.error('Import failed:', error)
      alert('Import failed due to network error')
    } finally {
      setImportLoading(false)
      // Reset file input
      event.target.value = ''
    }
  }

  const handleExportUsers = async () => {
    try {
      const response = await fetch('/api/users/import')
      const blob = await response.blob()
      
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `users-export-${new Date().toISOString().split('T')[0]}.csv`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export failed:', error)
      alert('Export failed')
    }
  }

  const filteredUsers = users.filter(user =>
    user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.ssoId.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.role.toLowerCase().includes(searchQuery.toLowerCase())
  )

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-lg text-gray-600">Loading users...</div>
        </div>
      </DashboardLayout>
    )
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Users</h2>
            <p className="text-gray-600">Manage team members and their skills</p>
          </div>
          <div className="flex items-center space-x-3">
            <input
              type="file"
              accept=".csv"
              onChange={handleImportUsers}
              className="hidden"
              id="user-csv-upload"
            />
            <label
              htmlFor="user-csv-upload"
              className={`text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200 flex items-center ${
                importLoading 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-netapp-secondary hover:bg-netapp-primary cursor-pointer'
              }`}
            >
              {importLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Importing...
                </>
              ) : (
                'Import CSV'
              )}
            </label>
            <button
              onClick={handleExportUsers}
              className="bg-gray-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-gray-700 transition-colors duration-200 flex items-center"
            >
              Export CSV
            </button>
            <button
              onClick={handleCreateUser}
              className="btn-primary flex items-center"
            >
              <PlusIcon className="w-4 h-4 mr-2" />
              Add User
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between bg-white p-4 rounded-lg shadow">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <MagnifyingGlassIcon className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search users..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent w-64"
              />
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setViewMode('cards')}
              className={`p-2 rounded ${viewMode === 'cards' ? 'bg-netapp-primary text-white' : 'bg-gray-200 text-gray-600'}`}
            >
              Cards
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded ${viewMode === 'list' ? 'bg-netapp-primary text-white' : 'bg-gray-200 text-gray-600'}`}
            >
              List
            </button>
          </div>
        </div>

        {/* Users Display */}
        {viewMode === 'cards' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredUsers.map((user) => (
              <div key={user._id} className="card hover:shadow-lg transition-shadow duration-200">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center">
                    <div className="w-12 h-12 bg-netapp-primary rounded-full flex items-center justify-center">
                      <UserCircleIcon className="w-8 h-8 text-white" />
                    </div>
                    <div className="ml-3">
                      <h3 className="text-lg font-semibold text-gray-900">{user.name}</h3>
                      <p className="text-sm text-gray-500">{user.ssoId}</p>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleEditUser(user)}
                      className="text-gray-400 hover:text-netapp-primary"
                    >
                      <PencilIcon className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeleteUser(user._id)}
                      disabled={deleteLoading === user._id}
                      className="text-gray-400 hover:text-red-600 disabled:opacity-50"
                    >
                      {deleteLoading === user._id ? (
                        <div className="w-4 h-4 border border-red-600 border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <TrashIcon className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Role:</span>
                    <span className="font-medium capitalize">{user.role}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Experience:</span>
                    <span className="font-medium">{user.experienceYears} years</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Workload:</span>
                    <span className="font-medium">{user.currentWorkload}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Status:</span>
                    <span className={`font-medium ${user.isAvailable ? 'text-green-600' : 'text-red-600'}`}>
                      {user.isAvailable ? 'Available' : 'Unavailable'}
                    </span>
                  </div>
                </div>

                {user.skills && user.skills.length > 0 && (
                  <div className="mt-4">
                    <div className="text-sm text-gray-600 mb-2">Top Skills:</div>
                    <div className="flex flex-wrap gap-1">
                      {user.skills.slice(0, 3).map((skill, index) => (
                        <span
                          key={index}
                          className="inline-block px-2 py-1 text-xs bg-netapp-light text-netapp-primary rounded"
                        >
                          {skill.skill} ({skill.level})
                        </span>
                      ))}
                      {user.skills.length > 3 && (
                        <span className="inline-block px-2 py-1 text-xs bg-gray-200 text-gray-600 rounded">
                          +{user.skills.length - 3} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      User
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Role
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Experience
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Workload
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredUsers.map((user) => (
                    <tr key={user._id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-10 h-10 bg-netapp-primary rounded-full flex items-center justify-center">
                            <UserCircleIcon className="w-6 h-6 text-white" />
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">{user.name}</div>
                            <div className="text-sm text-gray-500">{user.email}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm text-gray-900 capitalize">{user.role}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm text-gray-900">{user.experienceYears} years</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                            <div
                              className="bg-netapp-primary h-2 rounded-full"
                              style={{ width: `${Math.min(user.currentWorkload, 100)}%` }}
                            ></div>
                          </div>
                          <span className="text-sm text-gray-900">{user.currentWorkload}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          user.isAvailable 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {user.isAvailable ? 'Available' : 'Unavailable'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex space-x-2">
                          <button
                            onClick={() => handleEditUser(user)}
                            className="text-netapp-primary hover:text-netapp-dark"
                          >
                            <PencilIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDeleteUser(user._id)}
                            disabled={deleteLoading === user._id}
                            className="text-red-600 hover:text-red-900 disabled:opacity-50"
                          >
                            {deleteLoading === user._id ? (
                              <div className="w-4 h-4 border border-red-600 border-t-transparent rounded-full animate-spin" />
                            ) : (
                              <TrashIcon className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {filteredUsers.length === 0 && (
          <div className="card text-center py-12">
            <UserCircleIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No users found</h3>
            <p className="text-gray-500 mb-4">
              {searchQuery ? 'Try adjusting your search query.' : 'Get started by adding your first user.'}
            </p>
            {!searchQuery && (
              <button onClick={handleCreateUser} className="btn-primary">
                Add User
              </button>
            )}
          </div>
        )}
      </div>

      <UserModal
        user={selectedUser}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSaveUser}
        isCreating={isCreating}
        saveLoading={saveLoading}
      />
    </DashboardLayout>
  )
}