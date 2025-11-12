import { NextRequest, NextResponse } from 'next/server'
import connectDB from '@/lib/mongodb'
import Task from '@/models/Task'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    await connectDB()
    const task = await Task.findById(params.id)
      .populate('assignedTo', 'name ssoId email')
      .populate('createdBy', 'name ssoId')
    
    if (!task) {
      return NextResponse.json(
        { success: false, error: 'Task not found' },
        { status: 404 }
      )
    }
    
    return NextResponse.json({ success: true, data: task })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: 'Failed to fetch task' },
      { status: 500 }
    )
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    await connectDB()
    const body = await request.json()
    
    // If status is being changed to completed, set completedAt
    if (body.status === 'completed' && !body.completedAt) {
      body.completedAt = new Date()
    }
    
    const task = await Task.findByIdAndUpdate(
      params.id,
      body,
      { new: true, runValidators: true }
    ).populate('assignedTo', 'name ssoId email')
     .populate('createdBy', 'name ssoId')
    
    if (!task) {
      return NextResponse.json(
        { success: false, error: 'Task not found' },
        { status: 404 }
      )
    }
    
    return NextResponse.json({ success: true, data: task })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: 'Failed to update task' },
      { status: 500 }
    )
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    await connectDB()
    const task = await Task.findByIdAndDelete(params.id)
    
    if (!task) {
      return NextResponse.json(
        { success: false, error: 'Task not found' },
        { status: 404 }
      )
    }
    
    return NextResponse.json({ success: true, message: 'Task deleted successfully' })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: 'Failed to delete task' },
      { status: 500 }
    )
  }
}