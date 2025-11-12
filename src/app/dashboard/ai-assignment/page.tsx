'use client'

import { useState, useEffect } from 'react'
import DashboardLayout from '@/components/DashboardLayout'
import { 
  CpuChipIcon,
  PlayIcon,
  StopIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  UserGroupIcon,
  ClipboardDocumentListIcon,
  ClockIcon,
  SparklesIcon,
  ArrowPathIcon,
  DocumentArrowDownIcon,
  EnvelopeIcon
} from '@heroicons/react/24/outline'

interface AssignmentResult {
  success: boolean
  message: string
  data?: {
    totalTasks: number
    assignedTasks: number
    unassignedTasks: number
    emailsSent: number
    assignments: any[]
  }
  error?: string
}

interface Stats {
  totalUsers: number
  availableUsers: number
  totalTasks: number
  unassignedTasks: number
  avgWorkload: number
}

export default function AIAssignmentPage() {
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<AssignmentResult | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [showResults, setShowResults] = useState(false)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      // Fetch users and tasks to calculate stats
      const [usersResponse, tasksResponse] = await Promise.all([
        fetch('/api/users'),
        fetch('/api/tasks')
      ])
      
      const usersResult = await usersResponse.json()
      const tasksResult = await tasksResponse.json()
      
      if (usersResult.success && tasksResult.success) {
        const users = usersResult.data
        const tasks = tasksResult.data
        
        const availableUsers = users.filter((u: any) => u.isAvailable).length
        const unassignedTasks = tasks.filter((t: any) => !t.assignedTo).length
        const totalWorkload = users.reduce((sum: number, u: any) => sum + (u.currentWorkload || 0), 0)
        const avgWorkload = users.length > 0 ? Math.round(totalWorkload / users.length) : 0
        
        setStats({
          totalUsers: users.length,
          availableUsers,
          totalTasks: tasks.length,
          unassignedTasks,
          avgWorkload
        })
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`])
  }

  const runAIAssignment = async () => {
    setIsRunning(true)
    setProgress(0)
    setResult(null)
    setLogs([])
    setShowResults(false)
    
    try {
      addLog('Starting AI task assignment process...')
      setProgress(10)
      
      addLog('Analyzing available users and their skills...')
      setProgress(20)
      
      addLog('Identifying unassigned tasks...')
      setProgress(30)
      
      addLog('Running machine learning model...')
      setProgress(50)
      
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + 5
          if (newProgress >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return newProgress
        })
      }, 500)
      
      const response = await fetch('/api/ai/assign-tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      clearInterval(progressInterval)
      const assignmentResult = await response.json()
      
      setProgress(100)
      setResult(assignmentResult)
      
      if (assignmentResult.success) {
        addLog(`✅ Assignment completed successfully!`)
        addLog(`📊 ${assignmentResult.data?.assignedTasks} tasks assigned`)
        addLog(`📧 ${assignmentResult.data?.emailsSent} notification emails sent`)
        if (assignmentResult.data?.unassignedTasks > 0) {
          addLog(`⚠️ ${assignmentResult.data?.unassignedTasks} tasks remain unassigned`)
        }
      } else {
        addLog(`❌ Assignment failed: ${assignmentResult.error}`)
      }
      
      // Refresh stats after assignment
      await fetchStats()
      setShowResults(true)
      
    } catch (error) {
      console.error('AI assignment failed:', error)
      setResult({
        success: false,
        error: 'Network error or server unavailable',
        message: 'Failed to run AI assignment'
      })
      addLog('❌ Assignment failed due to network error')
    } finally {
      setIsRunning(false)
      setProgress(100)
    }
  }

  const downloadResults = () => {
    if (!result?.data?.assignments) return
    
    const csvContent = [
      'Task Title,Assigned User,User SSO ID,Confidence Score,Priority,Deadline',
      ...result.data.assignments.map((assignment: any) => 
        `"${assignment['Task Title']}","${assignment['Assigned User Name']}","${assignment['Assigned User SSO ID']}",${assignment['Confidence Score']},"${assignment['Predicted Priority']}","${assignment['Estimated Deadline']}"`
      )
    ].join('\n')
    
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ai-assignments-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    window.URL.revokeObjectURL(url)
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-netapp-primary to-netapp-secondary rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold mb-2 flex items-center">
                <SparklesIcon className="w-8 h-8 mr-3" />
                AI Task Assignment
              </h2>
              <p className="text-netapp-light">
                Let artificial intelligence automatically assign tasks to the most suitable team members 
                based on skills, experience, and current workload.
              </p>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold">AI</div>
              <div className="text-sm text-netapp-light">Powered</div>
            </div>
          </div>
        </div>

        {/* Stats Overview */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div className="card text-center">
              <UserGroupIcon className="w-8 h-8 text-netapp-primary mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">{stats.totalUsers}</div>
              <div className="text-sm text-gray-600">Total Users</div>
            </div>
            
            <div className="card text-center">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2">
                <CheckCircleIcon className="w-5 h-5 text-green-600" />
              </div>
              <div className="text-2xl font-bold text-gray-900">{stats.availableUsers}</div>
              <div className="text-sm text-gray-600">Available Users</div>
            </div>
            
            <div className="card text-center">
              <ClipboardDocumentListIcon className="w-8 h-8 text-netapp-secondary mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">{stats.totalTasks}</div>
              <div className="text-sm text-gray-600">Total Tasks</div>
            </div>
            
            <div className="card text-center">
              <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-2">
                <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600" />
              </div>
              <div className="text-2xl font-bold text-gray-900">{stats.unassignedTasks}</div>
              <div className="text-sm text-gray-600">Unassigned Tasks</div>
            </div>
            
            <div className="card text-center">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-2">
                <ClockIcon className="w-5 h-5 text-blue-600" />
              </div>
              <div className="text-2xl font-bold text-gray-900">{stats.avgWorkload}%</div>
              <div className="text-sm text-gray-600">Avg Workload</div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Control Panel */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Assignment Control</h3>
            
            {!stats?.unassignedTasks ? (
              <div className="text-center py-8">
                <CheckCircleIcon className="w-16 h-16 text-green-500 mx-auto mb-4" />
                <h4 className="text-lg font-medium text-gray-900 mb-2">All Tasks Assigned!</h4>
                <p className="text-gray-600 mb-4">There are no unassigned tasks at the moment.</p>
                <button
                  onClick={fetchStats}
                  className="btn-secondary flex items-center mx-auto"
                >
                  <ArrowPathIcon className="w-4 h-4 mr-2" />
                  Refresh Stats
                </button>
              </div>
            ) : stats.availableUsers === 0 ? (
              <div className="text-center py-8">
                <ExclamationTriangleIcon className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
                <h4 className="text-lg font-medium text-gray-900 mb-2">No Available Users</h4>
                <p className="text-gray-600">All team members are currently unavailable for assignment.</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-medium text-blue-900 mb-2">Ready for AI Assignment</h4>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• {stats.unassignedTasks} tasks waiting for assignment</li>
                    <li>• {stats.availableUsers} team members available</li>
                    <li>• Skills and workload analysis ready</li>
                  </ul>
                </div>

                <button
                  onClick={runAIAssignment}
                  disabled={isRunning}
                  className={`w-full flex items-center justify-center py-3 px-6 rounded-lg font-medium text-lg transition-colors duration-200 ${
                    isRunning
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-netapp-primary to-netapp-secondary hover:from-netapp-dark hover:to-netapp-primary text-white shadow-lg'
                  }`}
                >
                  {isRunning ? (
                    <>
                      <ArrowPathIcon className="w-5 h-5 mr-2 animate-spin" />
                      Running AI Assignment...
                    </>
                  ) : (
                    <>
                      <PlayIcon className="w-5 h-5 mr-2" />
                      Start AI Assignment
                    </>
                  )}
                </button>

                {isRunning && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Progress</span>
                      <span className="text-gray-900 font-medium">{progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-netapp-primary to-netapp-secondary h-2 rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Assignment Results</h3>
              {result?.success && result.data && (
                <button
                  onClick={downloadResults}
                  className="btn-secondary flex items-center text-sm"
                >
                  <DocumentArrowDownIcon className="w-4 h-4 mr-1" />
                  Download CSV
                </button>
              )}
            </div>

            {!result ? (
              <div className="text-center py-8 text-gray-500">
                <CpuChipIcon className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                <p>Run AI assignment to see results here</p>
              </div>
            ) : result.success ? (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <CheckCircleIcon className="w-5 h-5 text-green-600 mr-2" />
                    <span className="font-medium text-green-900">Assignment Successful!</span>
                  </div>
                  <p className="text-green-800 text-sm">{result.message}</p>
                </div>

                {result.data && (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-900">{result.data.assignedTasks}</div>
                      <div className="text-sm text-blue-700">Tasks Assigned</div>
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded-lg">
                      <div className="text-2xl font-bold text-green-900">{result.data.emailsSent}</div>
                      <div className="text-sm text-green-700">Emails Sent</div>
                    </div>
                  </div>
                )}

                {result.data && result.data.unassignedTasks > 0 && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex items-center mb-2">
                      <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 mr-2" />
                      <span className="font-medium text-yellow-900">Partially Assigned</span>
                    </div>
                    <p className="text-yellow-800 text-sm">
                      {result.data.unassignedTasks} tasks could not be assigned automatically.
                      Consider manual assignment or adjusting task requirements.
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <ExclamationTriangleIcon className="w-5 h-5 text-red-600 mr-2" />
                  <span className="font-medium text-red-900">Assignment Failed</span>
                </div>
                <p className="text-red-800 text-sm">{result.error || 'Unknown error occurred'}</p>
              </div>
            )}
          </div>
        </div>

        {/* Activity Log */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Activity Log</h3>
          <div className="bg-gray-900 rounded-lg p-4 max-h-64 overflow-y-auto">
            {logs.length === 0 ? (
              <p className="text-gray-400 text-sm">No activity to display</p>
            ) : (
              <div className="space-y-1">
                {logs.map((log, index) => (
                  <div key={index} className="text-sm font-mono text-gray-300">
                    {log}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* How It Works */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">How AI Assignment Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-blue-600 font-bold">1</span>
              </div>
              <h4 className="font-medium text-gray-900 mb-2">Skill Analysis</h4>
              <p className="text-sm text-gray-600">
                Analyzes task requirements and matches them with team member skills and experience levels.
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-green-600 font-bold">2</span>
              </div>
              <h4 className="font-medium text-gray-900 mb-2">Workload Balancing</h4>
              <p className="text-sm text-gray-600">
                Considers current workload and availability to ensure optimal resource distribution.
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-purple-600 font-bold">3</span>
              </div>
              <h4 className="font-medium text-gray-900 mb-2">Smart Assignment</h4>
              <p className="text-sm text-gray-600">
                Uses machine learning to make optimal assignments and automatically sends notifications.
              </p>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}