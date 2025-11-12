import mongoose from 'mongoose'

const TaskSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    required: true,
  },
  storyPoints: {
    type: Number,
    min: 1,
    max: 100,
  },
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'critical'],
    default: 'medium',
  },
  status: {
    type: String,
    enum: ['pending', 'in-progress', 'completed', 'cancelled'],
    default: 'pending',
  },
  assignedTo: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
  },
  requiredSkills: [String],
  estimatedHours: Number,
  deadline: Date,
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  completedAt: Date,
  difficultyLevel: {
    type: Number,
    min: 1,
    max: 10,
  },
  project: String,
  tags: [String],
}, {
  timestamps: true,
})

export default mongoose.models.Task || mongoose.model('Task', TaskSchema)