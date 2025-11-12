import mongoose from 'mongoose'

const UserSchema = new mongoose.Schema({
  ssoId: {
    type: String,
    required: true,
    unique: true,
  },
  name: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
  },
  role: {
    type: String,
    enum: ['Software Engineer', 'DevOps Engineer', 'Data Scientist', 'QA Engineer', 'Project Manager', 'System Analyst'],
    required: true,
  },
  skills: [{
    skill: String,
    level: {
      type: Number,
      min: 1,
      max: 10,
    },
  }],
  experienceYears: {
    type: Number,
    required: true,
    min: 0,
    max: 50,
    validate: {
      validator: Number.isInteger,
      message: 'Experience years must be a whole number'
    }
  },
  currentWorkload: {
    type: Number,
    required: true,
    min: 0,
    max: 100,
    validate: {
      validator: function(value: number) {
        return value >= 0 && value <= 100;
      },
      message: 'Current workload must be between 0 and 100 percent'
    }
  },
  isAvailable: {
    type: Boolean,
    default: true,
  },
  department: {
    type: String,
    enum: ['Engineering', 'Data Science', 'Quality Assurance', 'Operations', 'Management'],
    required: true,
  },
}, {
  timestamps: true,
})

export default mongoose.models.User || mongoose.model('User', UserSchema)