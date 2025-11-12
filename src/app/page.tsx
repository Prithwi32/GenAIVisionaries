import Link from 'next/link'
import { ArrowRightIcon, CpuChipIcon, UserGroupIcon, ClipboardDocumentListIcon } from '@heroicons/react/24/outline'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-netapp-light to-white">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-netapp-primary rounded-lg flex items-center justify-center">
                  <CpuChipIcon className="w-5 h-5 text-white" />
                </div>
              </div>
              <div className="ml-3">
                <h1 className="text-xl font-bold text-gray-900">GenAI Visionaries</h1>
                <p className="text-sm text-netapp-primary font-medium">Powered by NetApp</p>
              </div>
            </div>
            <Link href="/dashboard" className="btn-primary">
              Get Started
              <ArrowRightIcon className="w-4 h-4 ml-2 inline" />
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h2 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Intelligent Task Assignment
            <span className="block text-netapp-primary">Powered by AI</span>
          </h2>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Revolutionize your project management with our AI-powered system that automatically 
            assigns tasks to the best-suited team members, seamlessly integrating with JIRA and Azure Boards.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
            <Link href="/dashboard" className="btn-primary text-lg px-8 py-3">
              Launch Dashboard
              <ArrowRightIcon className="w-5 h-5 ml-2 inline" />
            </Link>
            <button className="bg-white text-netapp-primary border-2 border-netapp-primary hover:bg-netapp-light font-medium text-lg px-8 py-3 rounded-lg transition-colors duration-200">
              Watch Demo
            </button>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mt-20">
          <div className="card text-center">
            <div className="w-12 h-12 bg-netapp-primary rounded-lg flex items-center justify-center mx-auto mb-4">
              <CpuChipIcon className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">AI-Powered Matching</h3>
            <p className="text-gray-600">
              Advanced machine learning algorithms analyze team skills, experience, and workload to make optimal task assignments.
            </p>
          </div>

          <div className="card text-center">
            <div className="w-12 h-12 bg-netapp-secondary rounded-lg flex items-center justify-center mx-auto mb-4">
              <UserGroupIcon className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Team Management</h3>
            <p className="text-gray-600">
              Comprehensive user management with skill tracking, experience levels, and real-time availability monitoring.
            </p>
          </div>

          <div className="card text-center">
            <div className="w-12 h-12 bg-netapp-orange rounded-lg flex items-center justify-center mx-auto mb-4">
              <ClipboardDocumentListIcon className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Smart Prioritization</h3>
            <p className="text-gray-600">
              Automatic priority assignment and realistic deadline estimation based on task complexity and team capacity.
            </p>
          </div>
        </div>

        {/* Integration Section */}
        <div className="mt-20 text-center">
          <h3 className="text-2xl font-bold text-gray-900 mb-4">Seamless Integration</h3>
          <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
            Built to work harmoniously with your existing project management tools, 
            ensuring a smooth transition and enhanced productivity.
          </p>
          <div className="flex items-center justify-center space-x-8">
            <div className="bg-white p-4 rounded-lg shadow-md">
              <span className="text-blue-600 font-semibold">JIRA Integration</span>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-md">
              <span className="text-blue-800 font-semibold">Azure Boards</span>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-md">
              <span className="text-netapp-primary font-semibold">NetApp Data Fabric</span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-200 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-500">
            <p>&copy; 2024 NetApp, Inc. All rights reserved. GenAI Visionaries - Intelligent Task Assignment System</p>
          </div>
        </div>
      </footer>
    </div>
  )
}