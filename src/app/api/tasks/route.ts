import { NextRequest, NextResponse } from 'next/server'
import mongoose from 'mongoose'
import connectDB from '@/lib/mongodb'
import Task from '@/models/Task'

export async function GET(request: NextRequest) {
  try {
    await connectDB()
    
    const { searchParams } = new URL(request.url)
    const status = searchParams.get('status')
    
    let query = {}
    if (status) {
      query = { status }
    }
    
    const tasks = await Task.find(query)
      .populate('assignedTo', 'name ssoId email')
      .populate('createdBy', 'name ssoId')
      .sort({ createdAt: -1 })
    
    return NextResponse.json({ success: true, data: tasks })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: 'Failed to fetch tasks' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    await connectDB()
    const body = await request.json()
    
    console.log('Received task data:', JSON.stringify(body, null, 2))
    
    // Validate required fields
    if (!body.title || !body.description) {
      return NextResponse.json(
        { success: false, error: 'Title and description are required' },
        { status: 400 }
      )
    }
    
    // Validate ObjectId format for assignedTo if provided
    if (body.assignedTo && !mongoose.Types.ObjectId.isValid(body.assignedTo)) {
      return NextResponse.json(
        { success: false, error: 'Invalid assignedTo user ID' },
        { status: 400 }
      )
    }
    
    // Handle createdBy field - create a default ObjectId if not provided or invalid
    if (!body.createdBy || !mongoose.Types.ObjectId.isValid(body.createdBy)) {
      console.log('Creating default ObjectId for createdBy field')
      body.createdBy = new mongoose.Types.ObjectId()
    }
    
    const task = await Task.create(body)
    const populatedTask = await Task.findById(task._id)
      .populate('assignedTo', 'name ssoId email')
      .populate('createdBy', 'name ssoId')
    
    return NextResponse.json({ success: true, data: populatedTask }, { status: 201 })
  } catch (error) {
    console.error('Task creation error:', error)
    const errorMessage = error instanceof Error ? error.message : 'Failed to create task'
    return NextResponse.json(
      { success: false, error: errorMessage },
      { status: 500 }
    )
  }
}