'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  HomeIcon, 
  UserGroupIcon, 
  ClipboardDocumentListIcon,
  CpuChipIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon
} from '@heroicons/react/24/outline'

interface DashboardLayoutProps {
  children: React.ReactNode
}

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Users', href: '/dashboard/users', icon: UserGroupIcon },
  { name: 'Tasks', href: '/dashboard/tasks', icon: ClipboardDocumentListIcon },
  { name: 'AI Assignment', href: '/dashboard/ai-assignment', icon: CpuChipIcon },
  { name: 'Settings', href: '/dashboard/settings', icon: Cog6ToothIcon },
]

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const pathname = usePathname()

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-64 bg-white shadow-lg flex flex-col">
        {/* Logo */}
        <div className="flex items-center justify-center h-16 px-4 bg-netapp-primary">
          <Link href="/" className="flex items-center">
            <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center">
              <CpuChipIcon className="w-5 h-5 text-netapp-primary" />
            </div>
            <div className="ml-3">
              <div className="text-white font-bold text-lg">GenAI Visionaries</div>
              <div className="text-netapp-light text-xs">Powered by NetApp</div>
            </div>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigation.map((item) => {
            const isActive = pathname === item.href
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`sidebar-item ${isActive ? 'active' : ''}`}
              >
                <item.icon className="w-5 h-5 mr-3" />
                {item.name}
              </Link>
            )
          })}
        </nav>

        {/* User Profile Section */}
        <div className="border-t border-gray-200 p-4">
          <div className="flex items-center">
            <div className="w-10 h-10 bg-netapp-primary rounded-full flex items-center justify-center">
              <span className="text-white font-medium text-sm">AD</span>
            </div>
            <div className="ml-3 flex-1">
              <div className="text-sm font-medium text-gray-900">Admin User</div>
              <div className="text-xs text-gray-500">admin@netapp.com</div>
            </div>
          </div>
          
          <button className="mt-3 w-full flex items-center px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors duration-200">
            <ArrowRightOnRectangleIcon className="w-4 h-4 mr-2" />
            Sign out
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <header className="bg-white shadow-sm border-b border-gray-200 h-16 flex items-center justify-between px-6">
          <div className="flex items-center">
            <h1 className="text-2xl font-semibold text-gray-900">
              {navigation.find(item => item.href === pathname)?.name || 'Dashboard'}
            </h1>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Quick Stats */}
            <div className="hidden md:flex items-center space-x-6 text-sm">
              <div className="flex items-center">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                <span className="text-gray-600">System Online</span>
              </div>
              <div className="text-gray-500">
                Last Updated: {new Date().toLocaleTimeString()}
              </div>
            </div>
            
            {/* Notifications */}
            <button className="relative p-2 text-gray-400 hover:text-gray-600">
              <span className="sr-only">View notifications</span>
              <div className="w-6 h-6 bg-gray-200 rounded-full"></div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-400 rounded-full"></div>
            </button>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 p-6 overflow-hidden flex flex-col">
          {children}
        </main>
      </div>
    </div>
  )
}