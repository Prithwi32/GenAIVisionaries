import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import path from 'path'
import fs from 'fs'
import connectDB from '@/lib/mongodb'
import User from '@/models/User'
import Task from '@/models/Task'
import Assignment from '@/models/Assignment'
import { sendTaskAssignmentEmail } from '@/lib/email'

const execAsync = promisify(exec)

export async function POST(request: NextRequest) {
  try {
    await connectDB()
    
    // Get all users and unassigned tasks
    const users = await User.find({ isAvailable: true })
    const unassignedTasks = await Task.find({ assignedTo: null, status: 'pending' })
    
    if (users.length === 0) {
      return NextResponse.json(
        { success: false, error: 'No available users found' },
        { status: 400 }
      )
    }
    
    if (unassignedTasks.length === 0) {
      return NextResponse.json(
        { success: false, error: 'No unassigned tasks found' },
        { status: 400 }
      )
    }
    
    // Create temporary CSV files
    const tempDir = path.join(process.cwd(), 'temp')
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir)
    }
    
    const usersPath = path.join(tempDir, `users_${Date.now()}.csv`)
    const tasksPath = path.join(tempDir, `tasks_${Date.now()}.csv`)
    const outputPath = path.join(tempDir, `assignments_${Date.now()}.csv`)
    
    // Convert users to CSV format
    const usersCsvHeader = 'ssoId,name,email,role,skills,experienceYears,currentWorkload,isAvailable,department\n'
    const usersCsvData = users.map(user => {
      const skillsJson = JSON.stringify(user.skills || []).replace(/"/g, '""')
      return `${user.ssoId},"${user.name}","${user.email}","${user.role}","${skillsJson}",${user.experienceYears},${user.currentWorkload},${user.isAvailable},"${user.department || ''}"`
    }).join('\n')
    
    fs.writeFileSync(usersPath, usersCsvHeader + usersCsvData)
    
    // Convert tasks to CSV format
    const tasksCsvHeader = 'title,description,storyPoints,difficultyLevel,requiredSkills,project\n'
    const tasksCsvData = unassignedTasks.map(task => {
      const skillsJson = JSON.stringify(task.requiredSkills || []).replace(/"/g, '""')
      return `"${task.title}","${task.description}",${task.storyPoints || 5},${task.difficultyLevel || 5},"${skillsJson}","${task.project || ''}"`
    }).join('\n')
    
    fs.writeFileSync(tasksPath, tasksCsvHeader + tasksCsvData)
    
    // Run Python ML model
    const pythonScript = path.join(process.cwd(), 'ml-model', 'csv_processor.py')
    const command = `python "${pythonScript}" "${usersPath}" "${tasksPath}" "${outputPath}"`
    
    console.log('Running ML model:', command)
    
    try {
      const { stdout, stderr } = await execAsync(command, { 
        cwd: path.join(process.cwd(), 'ml-model'),
        timeout: 30000 // 30 second timeout
      })
      
      if (stderr) {
        console.warn('Python stderr:', stderr)
      }
      
      console.log('Python stdout:', stdout)
      
      // Check if output file was created
      if (!fs.existsSync(outputPath)) {
        throw new Error('ML model did not generate output file')
      }
      
      // Read the JSON output (if available)
      const jsonOutputPath = outputPath.replace('.csv', '.json')
      let assignments = []
      
      if (fs.existsSync(jsonOutputPath)) {
        const assignmentsData = fs.readFileSync(jsonOutputPath, 'utf-8')
        assignments = JSON.parse(assignmentsData)
      }
      
      // Process assignments and update database
      let assignmentCount = 0
      let emailsSent = 0
      
      for (const assignment of assignments) {
        if (assignment['Assigned User SSO ID'] !== 'Unassigned') {
          // Find the user and task in database
          const user = users.find(u => u.ssoId === assignment['Assigned User SSO ID'])
          const task = unassignedTasks.find(t => t.title === assignment['Task Title'])
          
          if (user && task) {
            // Update task with assignment
            await Task.findByIdAndUpdate(task._id, {
              assignedTo: user._id,
              priority: assignment['Predicted Priority'],
              deadline: assignment['Estimated Deadline'] ? new Date(assignment['Estimated Deadline']) : null,
              status: 'in-progress'
            })
            
            // Create assignment record
            await Assignment.create({
              taskId: task._id,
              userId: user._id,
              confidenceScore: assignment['Confidence Score'],
              reasonForAssignment: `AI assigned based on skill match and workload analysis`
            })
            
            // Send email notification
            try {
              await sendTaskAssignmentEmail({
                userEmail: user.email,
                userName: user.name,
                taskTitle: task.title,
                taskDescription: task.description,
                deadline: assignment['Estimated Deadline'],
                priority: assignment['Predicted Priority']
              })
              emailsSent++
            } catch (emailError) {
              console.error('Email sending failed:', emailError)
            }
            
            assignmentCount++
          }
        }
      }
      
      // Clean up temporary files
      try {
        fs.unlinkSync(usersPath)
        fs.unlinkSync(tasksPath)
        fs.unlinkSync(outputPath)
        if (fs.existsSync(jsonOutputPath)) {
          fs.unlinkSync(jsonOutputPath)
        }
      } catch (cleanupError) {
        console.warn('Failed to clean up temporary files:', cleanupError)
      }
      
      return NextResponse.json({
        success: true,
        message: 'Task assignment completed successfully',
        data: {
          totalTasks: unassignedTasks.length,
          assignedTasks: assignmentCount,
          unassignedTasks: unassignedTasks.length - assignmentCount,
          emailsSent: emailsSent,
          assignments: assignments
        }
      })
      
    } catch (pythonError) {
      console.error('Python execution error:', pythonError)
      
      // Clean up files on error
      try {
        if (fs.existsSync(usersPath)) fs.unlinkSync(usersPath)
        if (fs.existsSync(tasksPath)) fs.unlinkSync(tasksPath)
        if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath)
      } catch {}
      
      return NextResponse.json(
        { 
          success: false, 
          error: 'ML model execution failed',
          details: pythonError instanceof Error ? pythonError.message : 'Unknown error'
        },
        { status: 500 }
      )
    }
    
  } catch (error: any) {
    console.error('AI assignment error:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to run AI task assignment' },
      { status: 500 }
    )
  }
}