'use client'

import { useEffect, useState } from 'react'

interface MicrosoftUser {
  id: string
  name: string
  email: string
  ssoId: string
}

interface MicrosoftSSOProps {
  onUserAuthenticated?: (user: MicrosoftUser) => void
}

export default function MicrosoftSSO({ onUserAuthenticated }: MicrosoftSSOProps) {
  const [user, setUser] = useState<MicrosoftUser | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  useEffect(() => {
    // Check if user details are already in localStorage
    const storedUserDetails = localStorage.getItem('userDetails')
    if (storedUserDetails) {
      try {
        const user = JSON.parse(storedUserDetails)
        setUser(user)
        setIsAuthenticated(true)
      } catch (error) {
        // Clear invalid stored data
        localStorage.removeItem('userDetails')
      }
    }

    // Simulate Microsoft SSO authentication (replace with actual Microsoft Graph SDK)
    const simulateMicrosoftAuth = () => {
      // In a real implementation, you would use Microsoft Graph SDK or MSAL
      // This is a placeholder for demonstration
      const microsoftUser: MicrosoftUser = {
        id: '67582a1f9b8c123456789abc', // This would come from Microsoft Graph
        name: 'John Doe', // From Microsoft profile
        email: 'john.doe@netapp.com', // From Microsoft profile
        ssoId: 'john.doe@netapp.com' // Microsoft email as SSO ID
      }

      localStorage.setItem('userDetails', JSON.stringify(microsoftUser))
      setUser(microsoftUser)
      setIsAuthenticated(true)
      
      if (onUserAuthenticated) {
        onUserAuthenticated(microsoftUser)
      }
    }

    // Auto-authenticate for demo purposes (remove in production)
    if (!storedUserDetails) {
      simulateMicrosoftAuth()
    }
  }, [onUserAuthenticated])

  const handleSignOut = () => {
    localStorage.removeItem('userDetails')
    setUser(null)
    setIsAuthenticated(false)
  }

  if (!isAuthenticated || !user) {
    return (
      <div className="p-4 bg-blue-50 rounded-lg">
        <h3 className="text-lg font-medium text-blue-900">Microsoft SSO Authentication</h3>
        <p className="text-blue-700 mt-2">Connecting to Microsoft Account...</p>
        <div className="mt-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 bg-green-50 rounded-lg">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-green-900">Connected to Microsoft</h3>
          <p className="text-green-700">
            Signed in as: <strong>{user.name}</strong> ({user.ssoId})
          </p>
        </div>
        <button
          onClick={handleSignOut}
          className="px-4 py-2 text-sm bg-red-500 text-white rounded hover:bg-red-600"
        >
          Sign Out
        </button>
      </div>
    </div>
  )
}

// Helper function to get current authenticated user
// Helper function to get current authenticated user
export function getCurrentUser(): MicrosoftUser | null {
  try {
    const userDetails = localStorage.getItem('userDetails')
    return userDetails ? JSON.parse(userDetails) : null
  } catch (error) {
    return null
  }
}

// Helper function to ensure user is authenticated
export function ensureAuthenticated(): boolean {
  const user = getCurrentUser()
  return user !== null && !!user.email && !!user.ssoId
}