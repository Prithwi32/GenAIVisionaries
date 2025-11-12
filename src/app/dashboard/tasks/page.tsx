'use client'

import { useState, useEffect } from 'react'
import DashboardLayout from '@/components/DashboardLayout'
import MicrosoftSSO, { getCurrentUser } from '@/components/MicrosoftSSO'
import { 
  PlusIcon, 
  MagnifyingGlassIcon,
  PencilIcon,
  TrashIcon,
  ClipboardDocumentListIcon,
  XMarkIcon,
  UserIcon,
  CalendarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'

interface Task {
  _id: string
  title: string
  description: string
  storyPoints: number
  priority: 'low' | 'medium' | 'high' | 'critical'
  status: 'pending' | 'in-progress' | 'completed' | 'cancelled'
  assignedTo?: {
    _id: string
    name: string
    ssoId: string
    email: string
  }
  requiredSkills: string[]
  estimatedHours?: number
  deadline?: string
  createdBy: {
    _id: string
    name: string
    ssoId: string
  }
  completedAt?: string
  difficultyLevel: number
  project?: string
  tags: string[]
  createdAt: string
  updatedAt: string
}

interface User {
  _id: string
  name: string
  ssoId: string
  email: string
}

interface TaskModalProps {
  task: Task | null
  isOpen: boolean
  onClose: () => void
  onSave: (task: Partial<Task>) => void
  isCreating: boolean
  users: User[]
  saveLoading: boolean
}

const priorities = ['low', 'medium', 'high', 'critical']
const statuses = ['pending', 'in-progress', 'completed', 'cancelled']

const priorityColors = {
  low: 'bg-green-100 text-green-800',
  medium: 'bg-yellow-100 text-yellow-800',
  high: 'bg-orange-100 text-orange-800',
  critical: 'bg-red-100 text-red-800'
}

const statusColors = {
  pending: 'bg-gray-100 text-gray-800',
  'in-progress': 'bg-blue-100 text-blue-800',
  completed: 'bg-green-100 text-green-800',
  cancelled: 'bg-red-100 text-red-800'
}

function TaskModal({ task, isOpen, onClose, onSave, isCreating, users, saveLoading }: TaskModalProps) {
  const [formData, setFormData] = useState<Partial<Task>>({
    title: '',
    description: '',
    storyPoints: 5,
    priority: 'medium',
    status: 'pending',
    assignedTo: undefined,
    requiredSkills: [],
    estimatedHours: 0,
    deadline: '',
    difficultyLevel: 5,
    project: '',
    tags: []
  })

  const [newSkill, setNewSkill] = useState('')
  const [newTag, setNewTag] = useState('')

  useEffect(() => {
    if (task) {
      setFormData({
        ...task,
        deadline: task.deadline ? new Date(task.deadline).toISOString().split('T')[0] : '',
        assignedTo: task.assignedTo?._id ? { _id: task.assignedTo._id } as any : undefined
      })
    } else {
      setFormData({
        title: '',
        description: '',
        storyPoints: 5,
        priority: 'medium',
        status: 'pending',
        assignedTo: undefined,
        requiredSkills: [],
        estimatedHours: 0,
        deadline: '',
        difficultyLevel: 5,
        project: '',
        tags: []
      })
    }
  }, [task])

  const addSkill = () => {
    if (newSkill.trim() && !formData.requiredSkills?.includes(newSkill.trim())) {
      setFormData(prev => ({
        ...prev,
        requiredSkills: [...(prev.requiredSkills || []), newSkill.trim()]
      }))
      setNewSkill('')
    }
  }

  const removeSkill = (skill: string) => {
    setFormData(prev => ({
      ...prev,
      requiredSkills: prev.requiredSkills?.filter(s => s !== skill) || []
    }))
  }

  const addTag = () => {
    if (newTag.trim() && !formData.tags?.includes(newTag.trim())) {
      setFormData(prev => ({
        ...prev,
        tags: [...(prev.tags || []), newTag.trim()]
      }))
      setNewTag('')
    }
  }

  const removeTag = (tag: string) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags?.filter(t => t !== tag) || []
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const submitData: Partial<Task> = {
      ...formData,
      assignedTo: (formData.assignedTo && typeof formData.assignedTo === 'object') 
        ? { 
            _id: formData.assignedTo._id, 
            name: formData.assignedTo.name || '',
            ssoId: formData.assignedTo.ssoId || '',
            email: formData.assignedTo.email || ''
          } 
        : undefined,
      createdBy: task?.createdBy || undefined // Let the API handle the createdBy field
    }
    onSave(submitData)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">
            {isCreating ? 'Create New Task' : 'Edit Task'}
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Task Title *
                </label>
                <input
                  type="text"
                  required
                  value={formData.title || ''}
                  onChange={(e) => setFormData(prev => ({ ...prev, title: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  placeholder="Enter task title"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description *
                </label>
                <textarea
                  required
                  rows={4}
                  value={formData.description || ''}
                  onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  placeholder="Describe the task requirements and objectives"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Story Points
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    value={formData.storyPoints || 5}
                    onChange={(e) => setFormData(prev => ({ ...prev, storyPoints: parseInt(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Difficulty Level
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={formData.difficultyLevel || 5}
                    onChange={(e) => setFormData(prev => ({ ...prev, difficultyLevel: parseInt(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Priority
                  </label>
                  <select
                    value={formData.priority || 'medium'}
                    onChange={(e) => setFormData(prev => ({ ...prev, priority: e.target.value as any }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  >
                    {priorities.map(priority => (
                      <option key={priority} value={priority}>
                        {priority.charAt(0).toUpperCase() + priority.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Status
                  </label>
                  <select
                    value={formData.status || 'pending'}
                    onChange={(e) => setFormData(prev => ({ ...prev, status: e.target.value as any }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  >
                    {statuses.map(status => (
                      <option key={status} value={status}>
                        {status.charAt(0).toUpperCase() + status.slice(1).replace('-', ' ')}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Right Column */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Assigned To
                </label>
                <select
                  value={formData.assignedTo?._id || ''}
                  onChange={(e) => {
                    const userId = e.target.value
                    setFormData(prev => ({
                      ...prev,
                      assignedTo: userId ? { _id: userId } as any : undefined
                    }))
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                >
                  <option value="">Unassigned</option>
                  {users.map(user => (
                    <option key={user._id} value={user._id}>
                      {user.name} ({user.ssoId})
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Estimated Hours
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="0.5"
                    value={formData.estimatedHours || 0}
                    onChange={(e) => setFormData(prev => ({ ...prev, estimatedHours: parseFloat(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Deadline
                  </label>
                  <input
                    type="date"
                    value={formData.deadline || ''}
                    onChange={(e) => setFormData(prev => ({ ...prev, deadline: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Project
                </label>
                <input
                  type="text"
                  value={formData.project || ''}
                  onChange={(e) => setFormData(prev => ({ ...prev, project: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  placeholder="Project name"
                />
              </div>

              {/* Required Skills */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Required Skills
                </label>
                <div className="flex gap-2 mb-2">
                  <input
                    type="text"
                    value={newSkill}
                    onChange={(e) => setNewSkill(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addSkill())}
                    placeholder="Add required skill"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  />
                  <button
                    type="button"
                    onClick={addSkill}
                    className="px-4 py-2 bg-netapp-primary text-white rounded-md hover:bg-netapp-dark"
                  >
                    Add
                  </button>
                </div>
                <div className="flex flex-wrap gap-1">
                  {formData.requiredSkills?.map((skill, index) => (
                    <span
                      key={index}
                      className="inline-flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded"
                    >
                      {skill}
                      <button
                        type="button"
                        onClick={() => removeSkill(skill)}
                        className="ml-1 text-blue-600 hover:text-blue-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              </div>

              {/* Tags */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tags
                </label>
                <div className="flex gap-2 mb-2">
                  <input
                    type="text"
                    value={newTag}
                    onChange={(e) => setNewTag(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                    placeholder="Add tag"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
                  />
                  <button
                    type="button"
                    onClick={addTag}
                    className="px-4 py-2 bg-netapp-secondary text-white rounded-md hover:bg-netapp-primary"
                  >
                    Add
                  </button>
                </div>
                <div className="flex flex-wrap gap-1">
                  {formData.tags?.map((tag, index) => (
                    <span
                      key={index}
                      className="inline-flex items-center px-2 py-1 text-xs bg-gray-100 text-gray-800 rounded"
                    >
                      {tag}
                      <button
                        type="button"
                        onClick={() => removeTag(tag)}
                        className="ml-1 text-gray-600 hover:text-gray-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              </div>
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
                isCreating ? 'Create Task' : 'Save Changes'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<Task[]>([])
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTask, setSelectedTask] = useState<Task | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [saveLoading, setSaveLoading] = useState(false)
  const [deleteLoading, setDeleteLoading] = useState<string | null>(null)
  const [importLoading, setImportLoading] = useState(false)

  useEffect(() => {
    fetchTasks()
    fetchUsers()
  }, [])

  const fetchTasks = async () => {
    try {
      const response = await fetch('/api/tasks')
      const result = await response.json()
      if (result.success) {
        setTasks(result.data)
      }
    } catch (error) {
      console.error('Failed to fetch tasks:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchUsers = async () => {
    try {
      const response = await fetch('/api/users')
      const result = await response.json()
      if (result.success) {
        setUsers(result.data)
      }
    } catch (error) {
      console.error('Failed to fetch users:', error)
    }
  }

  const handleCreateTask = () => {
    setSelectedTask(null)
    setIsCreating(true)
    setIsModalOpen(true)
  }

  const handleEditTask = (task: Task) => {
    setSelectedTask(task)
    setIsCreating(false)
    setIsModalOpen(true)
  }

  const handleSaveTask = async (taskData: Partial<Task>) => {
    setSaveLoading(true)
    try {
      const url = isCreating ? '/api/tasks' : `/api/tasks/${selectedTask?._id}`
      const method = isCreating ? 'POST' : 'PUT'
      
      // Get current authenticated Microsoft user
      const currentUser = getCurrentUser()
      let createdById = typeof taskData.createdBy === 'object' ? taskData.createdBy?._id : taskData.createdBy
      
      if (!createdById && currentUser) {
        // Try to find user by SSO email in the users list
        const foundUser = users.find(u => u.email === currentUser.email || u.ssoId === currentUser.ssoId)
        if (foundUser) {
          createdById = foundUser._id
        } else {
          // If user not found in database, use their Microsoft ID as fallback
          createdById = currentUser.id
        }
      }
      
      const taskPayload = {
        ...taskData,
        createdBy: createdById || undefined // Let the API handle default creation
      }
      
      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(taskPayload)
      })

      const result = await response.json()
      if (result.success) {
        await fetchTasks()
        setIsModalOpen(false)
        alert(isCreating ? 'Task created successfully!' : 'Task updated successfully!')
      } else {
        alert(`Failed to save task: ${result.error}`)
      }
    } catch (error) {
      console.error('Failed to save task:', error)
      alert('Failed to save task due to network error')
    } finally {
      setSaveLoading(false)
    }
  }

  const handleDeleteTask = async (taskId: string) => {
    if (confirm('Are you sure you want to delete this task?')) {
      setDeleteLoading(taskId)
      try {
        const response = await fetch(`/api/tasks/${taskId}`, { method: 'DELETE' })
        const result = await response.json()
        if (result.success) {
          await fetchTasks()
          alert('Task deleted successfully!')
        } else {
          alert(`Failed to delete task: ${result.error}`)
        }
      } catch (error) {
        console.error('Failed to delete task:', error)
        alert('Failed to delete task due to network error')
      } finally {
        setDeleteLoading(null)
      }
    }
  }

  const handleImportTasks = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      setImportLoading(true)
      const response = await fetch('/api/tasks/import', {
        method: 'POST',
        body: formData
      })

      const result = await response.json()
      
      if (result.success) {
        alert(`Successfully imported ${result.data.imported} tasks!`)
        await fetchTasks()
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

  const handleExportTasks = async () => {
    try {
      const response = await fetch('/api/tasks/import')
      const blob = await response.blob()
      
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `tasks-export-${new Date().toISOString().split('T')[0]}.csv`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export failed:', error)
      alert('Export failed')
    }
  }

  const filteredTasks = tasks.filter(task => {
    const matchesSearch = task.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         task.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         task.project?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         task.assignedTo?.name.toLowerCase().includes(searchQuery.toLowerCase())
    
    const matchesStatus = statusFilter === 'all' || task.status === statusFilter
    
    return matchesSearch && matchesStatus
  })

  const tasksByStatus = {
    pending: filteredTasks.filter(t => t.status === 'pending'),
    'in-progress': filteredTasks.filter(t => t.status === 'in-progress'),
    completed: filteredTasks.filter(t => t.status === 'completed')
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <ClockIcon className="w-4 h-4" />
      case 'in-progress': return <ExclamationTriangleIcon className="w-4 h-4" />
      case 'completed': return <CheckCircleIcon className="w-4 h-4" />
      default: return <ClipboardDocumentListIcon className="w-4 h-4" />
    }
  }

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <div className="text-lg text-gray-600">Loading tasks...</div>
        </div>
      </DashboardLayout>
    )
  }

  return (
    <DashboardLayout>
      <div className="flex flex-col h-full space-y-6" style={{ height: 'calc(100vh - 140px)' }}>
        {/* Microsoft SSO Authentication Status */}
        <MicrosoftSSO />
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Tasks & User Stories</h2>
            <p className="text-gray-600">Manage project tasks and track progress</p>
          </div>
          <div className="flex items-center space-x-3">
            <input
              type="file"
              accept=".csv"
              onChange={handleImportTasks}
              className="hidden"
              id="task-csv-upload"
            />
            <label
              htmlFor="task-csv-upload"
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
              onClick={handleExportTasks}
              className="bg-gray-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-gray-700 transition-colors duration-200 flex items-center"
            >
              Export CSV
            </button>
            <button onClick={handleCreateTask} className="btn-primary flex items-center">
              <PlusIcon className="w-4 h-4 mr-2" />
              Create Task
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
                placeholder="Search tasks..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent w-64"
              />
            </div>

            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-netapp-primary focus:border-transparent"
            >
              <option value="all">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="in-progress">In Progress</option>
              <option value="completed">Completed</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </div>

          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-gray-400 rounded-full mr-2"></div>
              <span>Pending: {tasksByStatus.pending.length}</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-blue-400 rounded-full mr-2"></div>
              <span>In Progress: {tasksByStatus['in-progress'].length}</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
              <span>Completed: {tasksByStatus.completed.length}</span>
            </div>
          </div>
        </div>

        {/* Tasks Grid - Kanban Style */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6" style={{ height: 'calc(100vh - 220px)' }}>
          {Object.entries(tasksByStatus).map(([status, statusTasks]) => (
            <div key={status} className="card flex flex-col h-full">
              <div className="flex items-center justify-between mb-4 flex-shrink-0">
                <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                  {getStatusIcon(status)}
                  <span className="ml-2 capitalize">{status.replace('-', ' ')}</span>
                </h3>
                <span className="bg-gray-100 text-gray-800 text-sm font-medium px-2 py-1 rounded-full">
                  {statusTasks.length}
                </span>
              </div>

              <div className="space-y-3 flex-1 min-h-0 kanban-column pr-2 mb-10" style={{ maxHeight: 'calc(100vh - 320px)', overflowY: 'auto' }}>
                {statusTasks.map((task) => (
                  <div
                    key={task._id}
                    className="bg-gray-50 border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow duration-200 cursor-pointer"
                    onClick={() => handleEditTask(task)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="text-sm font-medium text-gray-900 line-clamp-2">
                        {task.title}
                      </h4>
                      <div className="flex space-x-1 ml-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleEditTask(task)
                          }}
                          className="text-gray-400 hover:text-netapp-primary"
                        >
                          <PencilIcon className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteTask(task._id)
                          }}
                          disabled={deleteLoading === task._id}
                          className="text-gray-400 hover:text-red-600 disabled:opacity-50"
                        >
                          {deleteLoading === task._id ? (
                            <div className="w-3 h-3 border border-red-600 border-t-transparent rounded-full animate-spin" />
                          ) : (
                            <TrashIcon className="w-3 h-3" />
                          )}
                        </button>
                      </div>
                    </div>

                    <p className="text-xs text-gray-600 mb-3 line-clamp-2">
                      {task.description}
                    </p>

                    <div className="flex items-center justify-between mb-2">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${priorityColors[task.priority]}`}>
                        {task.priority}
                      </span>
                      <div className="text-xs text-gray-500">
                        SP: {task.storyPoints}
                      </div>
                    </div>

                    {task.assignedTo && (
                      <div className="flex items-center text-xs text-gray-600 mb-2">
                        <UserIcon className="w-3 h-3 mr-1" />
                        {task.assignedTo.name}
                      </div>
                    )}

                    {task.deadline && (
                      <div className="flex items-center text-xs text-gray-600 mb-2">
                        <CalendarIcon className="w-3 h-3 mr-1" />
                        {new Date(task.deadline).toLocaleDateString()}
                      </div>
                    )}

                    {task.requiredSkills.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {task.requiredSkills.slice(0, 2).map((skill, index) => (
                          <span
                            key={index}
                            className="inline-block px-2 py-1 text-xs bg-netapp-light text-netapp-primary rounded"
                          >
                            {skill}
                          </span>
                        ))}
                        {task.requiredSkills.length > 2 && (
                          <span className="inline-block px-2 py-1 text-xs bg-gray-200 text-gray-600 rounded">
                            +{task.requiredSkills.length - 2}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}

                {statusTasks.length === 0 && (
                  <div className="text-center text-gray-500 py-8">
                    <ClipboardDocumentListIcon className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                    <p className="text-sm">No {status.replace('-', ' ')} tasks</p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {filteredTasks.length === 0 && (
          <div className="card text-center py-12">
            <ClipboardDocumentListIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No tasks found</h3>
            <p className="text-gray-500 mb-4">
              {searchQuery ? 'Try adjusting your search query.' : 'Get started by creating your first task.'}
            </p>
            {!searchQuery && (
              <button onClick={handleCreateTask} className="btn-primary">
                Create Task
              </button>
            )}
          </div>
        )}
      </div>

      <TaskModal
        task={selectedTask}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSaveTask}
        isCreating={isCreating}
        users={users}
        saveLoading={saveLoading}
      />
    </DashboardLayout>
  )
}