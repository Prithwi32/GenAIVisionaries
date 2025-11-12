import { NextRequest, NextResponse } from 'next/server'
import connectDB from '@/lib/mongodb'
import Task from '@/models/Task'
import User from '@/models/User'

// Parse CSV line with proper quote handling
function parseCSVLine(line: string): string[] {
  const result = []
  let current = ''
  let inQuotes = false
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    const nextChar = line[i + 1]
    
    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        current += '"'
        i++ // Skip next quote
      } else {
        inQuotes = !inQuotes
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim())
      current = ''
    } else {
      current += char
    }
  }
  result.push(current.trim())
  return result
}

export async function POST(request: NextRequest) {
  try {
    await connectDB()
    
    const formData = await request.formData()
    const file = formData.get('file') as File
    
    if (!file) {
      return NextResponse.json(
        { success: false, error: 'No file uploaded' },
        { status: 400 }
      )
    }
    
    const text = await file.text()
    const lines = text.split(/\r?\n/)
    
    if (lines.length < 2) {
      return NextResponse.json(
        { success: false, error: 'CSV file must have header and at least one data row' },
        { status: 400 }
      )
    }
    
    const header = parseCSVLine(lines[0])
    
    // Validate required columns - support both formats
    const requiredColumns = ['title', 'description']
    const altColumns = ['Task Title', 'Description (User story/ Problem description)'] // UserStory_Dataset.csv format
    
    // Check if we have either format
    const hasStandardFormat = requiredColumns.every(col => header.includes(col))
    const hasUserStoryFormat = altColumns.every(col => header.includes(col))
    
    if (!hasStandardFormat && !hasUserStoryFormat) {
      return NextResponse.json(
        { 
          success: false, 
          error: `CSV must have either format: [${requiredColumns.join(', ')}] or [${altColumns.join(', ')}]` 
        },
        { status: 400 }
      )
    }
    
    const tasks = []
    const errors = []
    
    // Get a default user for createdBy field
    const defaultUser = await User.findOne({})
    if (!defaultUser) {
      return NextResponse.json(
        { success: false, error: 'No users found in system. Please import users first.' },
        { status: 400 }
      )
    }
    
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim()
      if (!line) continue
      
      try {
        // Parse CSV line using the robust parser
        const values = parseCSVLine(line)
        
        if (values.length !== header.length) {
          errors.push(`Line ${i + 1}: Column count mismatch`)
          continue
        }
        
        const taskData: any = {
          createdBy: defaultUser._id
        }
        
        header.forEach((col, index) => {
          let value = values[index]?.replace(/"/g, '')
          
          // Handle both standard and UserStory_Dataset.csv formats
          switch (col) {
            case 'title':
            case 'Task Title':
              taskData.title = value
              break
            case 'description':
            case 'Description (User story/ Problem description)':
              taskData.description = value
              break
            case 'project':
              taskData.project = value || 'NetApp Project'
              break
            case 'requiredSkills':
            case 'Required Skills':
              try {
                if (value && value.startsWith('[')) {
                  taskData.requiredSkills = JSON.parse(value)
                } else if (value) {
                  // Handle semicolon or comma separated skills
                  const separator = value.includes(';') ? ';' : ','
                  taskData.requiredSkills = value.split(separator).map(s => s.trim()).filter(s => s)
                } else {
                  taskData.requiredSkills = []
                }
              } catch {
                taskData.requiredSkills = value ? value.split(';').map(s => s.trim()).filter(s => s) : []
              }
              break
            case 'tags':
              try {
                taskData.tags = value ? JSON.parse(value) : []
              } catch {
                taskData.tags = value ? value.split(';').map(s => s.trim()).filter(s => s) : []
              }
              break
            case 'storyPoints':
            case 'Story Points':
              taskData.storyPoints = parseInt(value) || 5
              break
            case 'difficultyLevel':
            case 'Difficulty Level':
              // Handle text difficulty levels
              if (value === 'Easy') {
                taskData.difficultyLevel = 3
              } else if (value === 'Medium') {
                taskData.difficultyLevel = 5
              } else if (value === 'Hard') {
                taskData.difficultyLevel = 8
              } else {
                taskData.difficultyLevel = parseInt(value) || 5
              }
              break
            case 'estimatedHours':
            case 'Estimated Hours':
              taskData.estimatedHours = parseInt(value) || 0
              break
            case 'priority':
            case 'Priority':
              const priorityMap: {[key: string]: string} = {
                'low': 'low',
                'medium': 'medium', 
                'high': 'high',
                'critical': 'critical'
              }
              taskData.priority = priorityMap[value?.toLowerCase()] || 'medium'
              break
            case 'status':
            case 'Status':
              const statusMap: {[key: string]: string} = {
                'to do': 'pending',
                'pending': 'pending',
                'in progress': 'in-progress',
                'in-progress': 'in-progress',
                'review': 'in-progress',
                'done': 'completed',
                'completed': 'completed'
              }
              taskData.status = statusMap[value?.toLowerCase()] || 'pending'
              break
            case 'deadline':
              if (value) {
                try {
                  taskData.deadline = new Date(value)
                } catch {
                  // Invalid date, ignore
                }
              }
              break
          }
        })
        
        // Validate required fields
        if (!taskData.title || !taskData.description) {
          errors.push(`Line ${i + 1}: Missing required field values`)
          continue
        }
        
        tasks.push(taskData)
      } catch (error) {
        errors.push(`Line ${i + 1}: Parsing error - ${error}`)
      }
    }
    
    if (tasks.length === 0) {
      return NextResponse.json(
        { 
          success: false, 
          error: 'No valid tasks found in CSV',
          details: errors 
        },
        { status: 400 }
      )
    }
    
    // Insert tasks into database
    const insertedTasks = []
    const insertErrors = []
    
    for (const taskData of tasks) {
      try {
        const newTask = await Task.create(taskData)
        insertedTasks.push(newTask)
      } catch (error: any) {
        insertErrors.push(`Task "${taskData.title}": ${error.message}`)
      }
    }
    
    return NextResponse.json({
      success: true,
      message: `Successfully processed ${insertedTasks.length} tasks`,
      data: {
        imported: insertedTasks.length,
        errors: insertErrors.length,
        parseErrors: errors.length,
        details: {
          insertErrors,
          parseErrors: errors
        }
      }
    })
    
  } catch (error) {
    console.error('CSV import error:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to process CSV file' },
      { status: 500 }
    )
  }
}

export async function GET() {
  try {
    await connectDB()
    
    const tasks = await Task.find({})
      .populate('assignedTo', 'name ssoId email')
      .populate('createdBy', 'name ssoId')
      .sort({ createdAt: -1 })
    
    // Generate CSV content in UserStory_Dataset format for compatibility
    const csvHeader = 'Task Title,Description (User story/ Problem description),Estimated Hours,Story Points,Difficulty Level,Priority,Status,Required Skills\n'
    
    const csvData = tasks.map(task => {
      const skillsStr = (task.requiredSkills || []).join(';')
      
      // Map difficulty numbers back to text
      let difficultyText = 'Medium'
      if (task.difficultyLevel <= 3) difficultyText = 'Easy'
      else if (task.difficultyLevel >= 8) difficultyText = 'Hard'
      
      // Map status back to UserStory format
      let statusText = task.status
      if (task.status === 'pending') statusText = 'To Do'
      else if (task.status === 'in-progress') statusText = 'In Progress'
      else if (task.status === 'completed') statusText = 'Done'
      
      // Capitalize priority
      const priorityText = task.priority.charAt(0).toUpperCase() + task.priority.slice(1)
      
      return `"${task.title}","${task.description}",${task.estimatedHours || 0},${task.storyPoints},"${difficultyText}","${priorityText}","${statusText}","${skillsStr}"`
    }).join('\n')
    
    const csvContent = csvHeader + csvData
    
    return new Response(csvContent, {
      headers: {
        'Content-Type': 'text/csv',
        'Content-Disposition': `attachment; filename="tasks-export-${new Date().toISOString().split('T')[0]}.csv"`
      }
    })
    
  } catch (error) {
    return NextResponse.json(
      { success: false, error: 'Failed to export tasks' },
      { status: 500 }
    )
  }
}