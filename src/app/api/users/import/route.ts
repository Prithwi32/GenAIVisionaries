import { NextRequest, NextResponse } from 'next/server'
import connectDB from '@/lib/mongodb'
import User from '@/models/User'

export async function POST(request: NextRequest) {
  try {
    console.log('Starting user import process...')
    await connectDB()
    console.log('Database connected successfully')
    
    const formData = await request.formData()
    const file = formData.get('file') as File
    
    if (!file) {
      console.log('No file uploaded')
      return NextResponse.json(
        { success: false, error: 'No file uploaded' },
        { status: 400 }
      )
    }
    
    console.log('File received:', file.name, 'Size:', file.size, 'Type:', file.type)
    
    const text = await file.text()
    console.log('File text length:', text.length)
    
    const lines = text.split(/\r?\n/)
    
    if (lines.length < 2) {
      return NextResponse.json(
        { success: false, error: 'CSV file must have header and at least one data row' },
        { status: 400 }
      )
    }
    
    // Parse header - use simple split for Employee_Dataset.csv format
    const headerLine = lines[0].trim()
    console.log('Raw header line:', headerLine)
    
    const header = headerLine.split(',').map(h => h.trim().replace(/^["']|["']$/g, ''))
    
    console.log('Parsed header:', header)
    console.log('File has', lines.length, 'lines')
    
    // Check for Employee_Dataset.csv format (which is what we expect)
    const expectedColumns = ['SSO ID', 'Full Name', 'Email', 'Role', 'Experience (Years)', 'Current Workload (%)', 'Department', 'Skills']
    const hasAllRequiredColumns = ['SSO ID', 'Full Name', 'Email', 'Role'].every(col => header.includes(col))
    
    console.log('Has required Employee_Dataset columns:', hasAllRequiredColumns)
    
    if (!hasAllRequiredColumns) {
      console.log('Missing required columns. Expected: SSO ID, Full Name, Email, Role')
      console.log('Found header:', header)
      return NextResponse.json(
        { 
          success: false, 
          error: `CSV must have Employee Dataset format with columns: SSO ID, Full Name, Email, Role. Found: [${header.join(', ')}]` 
        },
        { status: 400 }
      )
    }
    
    const users = []
    const errors = []
    
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim()
      if (!line) continue
      
      try {
        // Parse CSV line using simple split (Employee_Dataset.csv doesn't use complex quoting)
        const values = line.split(',').map(cell => cell.trim().replace(/^["']|["']$/g, ''))
        console.log(`Line ${i + 1} values:`, values)
        
        if (values.length !== header.length) {
          errors.push(`Line ${i + 1}: Column count mismatch`)
          continue
        }
        
        const userData: any = {}
        
        header.forEach((col, index) => {
          let value = values[index] || ''
          
          // Handle both standard and Employee_Dataset.csv formats
          switch (col) {
            case 'ssoId':
            case 'SSO ID':
              userData.ssoId = value.toString()
              break
            case 'name':
            case 'Full Name':
              userData.name = value.toString()
              break
            case 'email':
            case 'Email':
              userData.email = value.toString()
              break
            case 'role':
            case 'Role':
              userData.role = value.toString()
              break
            case 'department':
            case 'Department':
              userData.department = value.toString()
              break
            case 'skills':
            case 'Skills':
              try {
                if (value && value.startsWith('[')) {
                  userData.skills = JSON.parse(value)
                } else if (value) {
                  // Handle semicolon or comma separated skills
                  const separator = value.includes(';') ? ';' : ','
                  const skillList = value.split(separator).map(s => s.trim()).filter(s => s)
                  userData.skills = skillList.map(skill => ({ skill: skill, level: 7 }))
                } else {
                  userData.skills = []
                }
              } catch {
                // Fallback for any parsing issues
                if (value) {
                  const separator = value.includes(';') ? ';' : ','
                  const skillList = value.split(separator).map(s => s.trim()).filter(s => s)
                  userData.skills = skillList.map(skill => ({ skill: skill, level: 7 }))
                } else {
                  userData.skills = []
                }
              }
              break
            case 'experienceYears':
            case 'Experience (Years)':
              userData.experienceYears = Math.max(0, Math.min(50, parseInt(value) || 0))
              break
            case 'currentWorkload':
            case 'Current Workload (%)':
              // Handle percentage values
              const workloadStr = value?.toString().replace('%', '') || '0'
              userData.currentWorkload = Math.max(0, Math.min(100, parseFloat(workloadStr) || 0))
              break
            case 'isAvailable':
              if (value !== undefined && value !== '') {
                userData.isAvailable = value?.toString().toLowerCase() === 'true' || value === '1'
              } else {
                // For Employee_Dataset, determine availability based on workload
                userData.isAvailable = (userData.currentWorkload || 0) < 95
              }
              break
          }
        })
        
        // Set availability based on workload if not explicitly set
        if (userData.isAvailable === undefined) {
          userData.isAvailable = (userData.currentWorkload || 0) < 95
        }
        
        // Validate required fields
        if (!userData.ssoId || !userData.name || !userData.email || !userData.role) {
          errors.push(`Line ${i + 1}: Missing required field values (ssoId: ${userData.ssoId}, name: ${userData.name}, email: ${userData.email}, role: ${userData.role})`)
          continue
        }
        
        // Set default values for optional fields
        if (!userData.department) userData.department = 'Engineering'
        if (userData.experienceYears === undefined) userData.experienceYears = 0
        if (userData.currentWorkload === undefined) userData.currentWorkload = 0
        if (!userData.skills || userData.skills.length === 0) userData.skills = []
        
        console.log(`Parsed user ${i}: ${userData.name} (${userData.email})`)
        users.push(userData)
      } catch (error) {
        errors.push(`Line ${i + 1}: Parsing error - ${error}`)
      }
    }
    
    if (users.length === 0) {
      return NextResponse.json(
        { 
          success: false, 
          error: 'No valid users found in CSV',
          details: errors 
        },
        { status: 400 }
      )
    }
    
    // Insert users into database
    const insertedUsers = []
    const insertErrors = []
    
    console.log(`Attempting to insert ${users.length} users into database`)
    
    for (const userData of users) {
      try {
        console.log(`Processing user: ${userData.ssoId} - ${userData.name}`)
        
        const existingUser = await User.findOne({ 
          $or: [{ ssoId: userData.ssoId }, { email: userData.email }] 
        })
        
        if (existingUser) {
          console.log(`Updating existing user: ${userData.ssoId}`)
          // Update existing user
          const updatedUser = await User.findByIdAndUpdate(
            existingUser._id,
            userData,
            { new: true, runValidators: true }
          )
          insertedUsers.push(updatedUser)
        } else {
          console.log(`Creating new user: ${userData.ssoId}`)
          // Create new user
          const newUser = await User.create(userData)
          insertedUsers.push(newUser)
        }
      } catch (error: any) {
        console.error(`Error with user ${userData.ssoId}:`, error)
        insertErrors.push(`User ${userData.ssoId}: ${error.message}`)
      }
    }
    
    return NextResponse.json({
      success: true,
      message: `Successfully processed ${insertedUsers.length} users`,
      data: {
        imported: insertedUsers.length,
        errors: insertErrors.length,
        parseErrors: errors.length,
        details: {
          insertErrors,
          parseErrors: errors
        }
      }
    })
    
  } catch (error: any) {
    console.error('CSV import error:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to process CSV file',
        details: error.message,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
      },
      { status: 500 }
    )
  }
}

export async function GET() {
  try {
    await connectDB()
    
    const users = await User.find({}).sort({ createdAt: -1 })
    
    // Generate CSV content in Employee_Dataset format for compatibility
    const csvHeader = 'SSO ID,Full Name,Email,Role,Experience (Years),Current Workload (%),Department,Skills\n'
    
    const csvData = users.map(user => {
      const skillsStr = (user.skills || []).map((s: any) => s.skill).join(';')
      return `${user.ssoId},"${user.name}","${user.email}","${user.role}",${user.experienceYears},${user.currentWorkload},"${user.department || ''}","${skillsStr}"`
    }).join('\n')
    
    const csvContent = csvHeader + csvData
    
    return new Response(csvContent, {
      headers: {
        'Content-Type': 'text/csv',
        'Content-Disposition': `attachment; filename="users-export-${new Date().toISOString().split('T')[0]}.csv"`
      }
    })
    
  } catch (error: any) {
    console.error('User export error:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to export users', details: error.message },
      { status: 500 }
    )
  }
}