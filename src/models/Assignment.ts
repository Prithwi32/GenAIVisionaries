import mongoose from 'mongoose'

const AssignmentSchema = new mongoose.Schema({
  taskId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Task',
    required: true,
  },
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  assignedAt: {
    type: Date,
    default: Date.now,
  },
  assignedBy: {
    type: String,
    default: 'AI System',
  },
  confidenceScore: {
    type: Number,
    min: 0,
    max: 1,
  },
  notificationSent: {
    type: Boolean,
    default: false,
  },
  reasonForAssignment: String,
}, {
  timestamps: true,
})

export default mongoose.models.Assignment || mongoose.model('Assignment', AssignmentSchema)