import DashboardLayout from '@/components/DashboardLayout'
import { 
  UserGroupIcon, 
  ClipboardDocumentListIcon,
  CpuChipIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'

async function getStats() {
  // In a real app, this would fetch from your API
  return {
    totalUsers: 24,
    totalTasks: 67,
    completedTasks: 23,
    pendingTasks: 31,
    inProgressTasks: 13,
    aiAssignments: 15
  }
}

export default async function DashboardPage() {
  const stats = await getStats()

  const statCards = [
    {
      name: 'Total Users',
      value: stats.totalUsers,
      icon: UserGroupIcon,
      color: 'bg-blue-500',
      change: '+12%',
      changeType: 'positive'
    },
    {
      name: 'Total Tasks',
      value: stats.totalTasks,
      icon: ClipboardDocumentListIcon,
      color: 'bg-green-500',
      change: '+8%',
      changeType: 'positive'
    },
    {
      name: 'Completed Tasks',
      value: stats.completedTasks,
      icon: CheckCircleIcon,
      color: 'bg-purple-500',
      change: '+18%',
      changeType: 'positive'
    },
    {
      name: 'AI Assignments',
      value: stats.aiAssignments,
      icon: CpuChipIcon,
      color: 'bg-orange-500',
      change: '+25%',
      changeType: 'positive'
    }
  ]

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Welcome Section */}
        <div className="bg-gradient-to-r from-netapp-primary to-netapp-secondary rounded-lg p-6 text-white">
          <h2 className="text-2xl font-bold mb-2">Welcome to GenAI Visionaries Dashboard</h2>
          <p className="text-netapp-light">
            Intelligent task assignment powered by machine learning. Monitor your team performance 
            and let AI optimize your workflow.
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {statCards.map((stat) => (
            <div key={stat.name} className="card">
              <div className="flex items-center">
                <div className={`p-3 rounded-lg ${stat.color}`}>
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
                <div className="ml-4 flex-1">
                  <p className="text-sm font-medium text-gray-500">{stat.name}</p>
                  <div className="flex items-center">
                    <p className="text-2xl font-semibold text-gray-900">{stat.value}</p>
                    <span className="ml-2 text-sm text-green-600 font-medium">
                      {stat.change}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Task Status Overview */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Task Status Overview</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
                <div className="flex items-center">
                  <ClockIcon className="w-5 h-5 text-yellow-600 mr-3" />
                  <span className="text-sm font-medium text-gray-900">Pending Tasks</span>
                </div>
                <span className="text-lg font-semibold text-yellow-600">{stats.pendingTasks}</span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                <div className="flex items-center">
                  <ExclamationTriangleIcon className="w-5 h-5 text-blue-600 mr-3" />
                  <span className="text-sm font-medium text-gray-900">In Progress</span>
                </div>
                <span className="text-lg font-semibold text-blue-600">{stats.inProgressTasks}</span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                <div className="flex items-center">
                  <CheckCircleIcon className="w-5 h-5 text-green-600 mr-3" />
                  <span className="text-sm font-medium text-gray-900">Completed</span>
                </div>
                <span className="text-lg font-semibold text-green-600">{stats.completedTasks}</span>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent AI Assignments</h3>
            <div className="space-y-3">
              {[
                {
                  task: 'User Authentication System',
                  assignee: 'John Doe',
                  time: '2 hours ago',
                  confidence: 0.95
                },
                {
                  task: 'Landing Page Design',
                  assignee: 'Jane Smith',
                  time: '4 hours ago',
                  confidence: 0.87
                },
                {
                  task: 'Database Migration',
                  assignee: 'Mike Johnson',
                  time: '6 hours ago',
                  confidence: 0.92
                }
              ].map((activity, index) => (
                <div key={index} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">{activity.task}</p>
                    <p className="text-xs text-gray-500">Assigned to {activity.assignee} • {activity.time}</p>
                  </div>
                  <div className="text-right">
                    <span className="inline-block px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                      {Math.round(activity.confidence * 100)}% match
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="card text-center">
            <UserGroupIcon className="w-12 h-12 text-netapp-primary mx-auto mb-3" />
            <h4 className="text-lg font-medium text-gray-900 mb-2">Manage Users</h4>
            <p className="text-sm text-gray-500 mb-4">Add, edit, and manage team members and their skills</p>
            <a href="/dashboard/users" className="btn-primary inline-block">
              View Users
            </a>
          </div>

          <div className="card text-center">
            <ClipboardDocumentListIcon className="w-12 h-12 text-netapp-secondary mx-auto mb-3" />
            <h4 className="text-lg font-medium text-gray-900 mb-2">Manage Tasks</h4>
            <p className="text-sm text-gray-500 mb-4">Create, assign, and track project tasks and user stories</p>
            <a href="/dashboard/tasks" className="btn-secondary inline-block">
              View Tasks
            </a>
          </div>

          <div className="card text-center">
            <CpuChipIcon className="w-12 h-12 text-netapp-orange mx-auto mb-3" />
            <h4 className="text-lg font-medium text-gray-900 mb-2">AI Assignment</h4>
            <p className="text-sm text-gray-500 mb-4">Let AI automatically assign tasks to optimal team members</p>
            <a href="/dashboard/ai-assignment" className="bg-netapp-orange hover:bg-orange-600 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 inline-block">
              Run AI Assignment
            </a>
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}