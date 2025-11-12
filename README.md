# GenAI Visionaries - Intelligent Task Assignment System

🎯 **NetApp's AI-Powered Task Assignment System**

An intelligent system that automates the assignment of project tasks (user stories) to the best-suited employees using machine learning, with seamless integration capabilities for JIRA and Azure Boards.

## 🌟 Features

### 🤖 AI-Powered Task Assignment
- **Smart Matching**: ML algorithm analyzes skills, experience, and workload
- **Priority Prediction**: Automatic priority assignment based on complexity
- **Deadline Estimation**: Realistic completion time calculation
- **Workload Balancing**: Optimal resource distribution across team

### 🏢 Enterprise-Ready Interface
- **NetApp Branding**: Professional UI with corporate identity
- **Dashboard Analytics**: Real-time statistics and progress tracking
- **User Management**: Complete CRUD operations with skills tracking
- **Task Management**: Kanban-style interface with status segregation

### 📊 Data Management
- **CSV Import/Export**: Bulk operations for users and tasks
- **MongoDB Integration**: Robust data persistence
- **RESTful APIs**: Complete backend API structure

### 📧 Notification System
- **Outlook Integration**: Professional email notifications
- **Assignment Alerts**: Automatic notifications on task assignment
- **HTML Templates**: Branded email communications

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Python 3.8+
- MongoDB (local or cloud)

### 1. Clone and Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd GenAIVisionaries

# Install Node.js dependencies
npm install

# Install Python ML dependencies
cd ml-model
pip install -r requirements.txt
cd ..
```

### 2. Environment Configuration

Create `.env.local` file in the root directory:

```env
# Database Configuration
MONGODB_URI=mongodb://localhost:27017/genaivisionaries
# For MongoDB Atlas: mongodb+srv://username:password@cluster.mongodb.net/database

# Email Configuration (Optional - for notifications)
EMAIL_HOST=smtp.office365.com
EMAIL_PORT=587
EMAIL_USER=your-email@company.com
EMAIL_PASS=your-app-password

# OAuth2 Alternative (Enterprise)
# EMAIL_CLIENT_ID=your-azure-client-id
# EMAIL_CLIENT_SECRET=your-azure-client-secret
# EMAIL_REFRESH_TOKEN=your-refresh-token

# Application Settings
NEXT_PUBLIC_APP_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here
```

**Note:** Email configuration is optional. The system works without email notifications.

### 3. Start the Application

```bash
# Start Next.js development server
npm run dev

# Application will be available at http://localhost:3001
```

## 📁 Project Structure

```
GenAIVisionaries/
├── src/
│   ├── app/                    # Next.js 14 App Router
│   │   ├── api/               # REST API endpoints
│   │   │   ├── users/         # User CRUD operations
│   │   │   ├── tasks/         # Task CRUD operations
│   │   │   └── ai/            # ML model integration
│   │   ├── dashboard/         # Dashboard pages
│   │   └── page.tsx           # Landing page
│   ├── components/            # Reusable React components
│   ├── lib/                   # Utilities and configurations
│   └── models/                # MongoDB schemas
├── ml-model/                  # Python ML system
│   ├── main.py               # Core ML model
│   ├── csv_processor.py      # CSV handling
│   ├── requirements.txt      # Python dependencies
│   └── sample_*.csv          # Sample data files
└── public/                   # Static assets
```

## 🎯 How to Use

### 1. Access the Dashboard
- Navigate to `http://localhost:3001`
- Click "Launch Dashboard" to access the management interface

### 2. Manage Users
- Go to **Users** section in the sidebar
- Add team members with their skills and experience
- Use "Import CSV" for bulk user creation
- Edit user profiles and skill levels as needed

### 3. Manage Tasks
- Go to **Tasks** section in the sidebar
- Create tasks with descriptions, requirements, and priorities
- Use "Import CSV" for bulk task creation
- View tasks in Kanban-style layout (Pending, In Progress, Completed)

### 4. Run AI Assignment
- Go to **AI Assignment** section
- Review system statistics and readiness
- Click "Start AI Assignment" to automatically assign tasks
- Monitor progress and view detailed results
- Download assignment results as CSV

### 5. Monitor Progress
- Dashboard provides real-time analytics
- Track assignment success rates and team workload
- View recent AI assignments and their confidence scores

## 🔧 API Endpoints

### Users API
- `GET /api/users` - List all users
- `POST /api/users` - Create new user
- `GET /api/users/[id]` - Get user by ID
- `PUT /api/users/[id]` - Update user
- `DELETE /api/users/[id]` - Delete user
- `POST /api/users/import` - Import users from CSV
- `GET /api/users/import` - Export users to CSV

### Tasks API
- `GET /api/tasks` - List all tasks
- `POST /api/tasks` - Create new task
- `GET /api/tasks/[id]` - Get task by ID
- `PUT /api/tasks/[id]` - Update task
- `DELETE /api/tasks/[id]` - Delete task
- `POST /api/tasks/import` - Import tasks from CSV
- `GET /api/tasks/import` - Export tasks to CSV

### AI Assignment API
- `POST /api/ai/assign-tasks` - Run AI task assignment

## 🤖 Machine Learning Model

The ML system uses a sophisticated algorithm that considers:

### Input Factors
- **User Skills**: Skill-requirement matching with proficiency levels
- **Experience**: Years of experience in relevant domains
- **Current Workload**: Existing task assignments and capacity
- **Availability**: User availability status
- **Task Complexity**: Story points and difficulty ratings

### ML Algorithms
- **Random Forest Classifier**: For priority prediction
- **Random Forest Regressor**: For deadline estimation
- **Custom Optimization**: For skill-based matching and workload balancing

### Output
- **Task Assignments**: User-task pairings with confidence scores
- **Priority Levels**: Automatic priority classification (Low/Medium/High/Critical)
- **Deadlines**: Realistic completion time estimates
- **Email Notifications**: Automatic assignment notifications

## 📊 CSV Import Format

### Users CSV Format
```csv
ssoId,name,email,role,skills,experienceYears,currentWorkload,isAvailable,department
emp001,John Doe,john@netapp.com,developer,"[{""skill"":""python"",""level"":8}]",5,30,true,Engineering
```

### Tasks CSV Format
```csv
title,description,storyPoints,difficultyLevel,requiredSkills,project
User Auth,Implement login system,8,7,"[""python"",""security""]",Web Platform
```

## 🔗 Integration Capabilities

### JIRA Integration (Ready)
- Compatible data models for JIRA import/export
- User story format alignment
- Priority and status mapping

### Azure Boards Integration (Ready)
- Work item compatibility
- Team member synchronization
- Sprint planning support

## 🎨 UI/UX Features

### NetApp Corporate Branding
- Official NetApp color scheme (#0067C5, #00A1C9)
- Professional typography and spacing
- Consistent visual identity

### Responsive Design
- Mobile-friendly interface
- Tablet and desktop optimization
- Accessible design principles

### User Experience
- Intuitive navigation
- Real-time feedback
- Progressive enhancement
- Error handling and validation

## 🛠️ Technology Stack

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Heroicons**: Professional icon library

### Backend
- **Node.js**: Runtime environment
- **MongoDB**: NoSQL database
- **Mongoose**: ODM for MongoDB
- **Nodemailer**: Email service integration

### Machine Learning
- **Python 3.8+**: ML runtime
- **Pandas**: Data manipulation
- **Scikit-learn**: ML algorithms
- **NumPy**: Numerical computing

### DevOps
- **RESTful APIs**: Standard HTTP protocols
- **JSON**: Data interchange format
- **CSV**: Bulk data operations
- **Environment Variables**: Configuration management

## 📈 Performance & Scalability

### Optimization Features
- **Database Indexing**: Optimized queries
- **Efficient Algorithms**: O(n log n) complexity for assignments
- **Batch Processing**: Bulk operations support
- **Caching**: Response optimization

### Scalability Considerations
- **Horizontal Scaling**: Microservices-ready architecture
- **Database Sharding**: MongoDB scaling capabilities
- **Load Balancing**: Multi-instance deployment ready
- **API Rate Limiting**: Protection against abuse

## 🔐 Security Features

### Data Protection
- **Input Validation**: Comprehensive data sanitization
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Output encoding
- **CSRF Protection**: Token-based security

### Access Control
- **Authentication Ready**: User session management
- **Authorization Framework**: Role-based permissions
- **Audit Logging**: Activity tracking
- **Data Encryption**: Sensitive data protection

## 🧪 Testing

### Manual Testing Checklist
1. **User Management**: Create, edit, delete users with skills
2. **Task Management**: Create, assign, update task statuses
3. **CSV Operations**: Import/export users and tasks
4. **AI Assignment**: Run automated assignment process
5. **Email Notifications**: Verify assignment emails
6. **Dashboard Analytics**: Check real-time statistics

### Sample Data
Use the provided sample CSV files in `ml-model/` directory:
- `sample_users.csv`: 8 sample users with various skills
- `sample_tasks.csv`: 12 sample tasks with different complexities

## 🚀 Deployment

### Production Deployment
1. Set up MongoDB Atlas or production database
2. Configure email service (Outlook/Office 365)
3. Set production environment variables
4. Deploy to Vercel, Netlify, or custom server
5. Configure domain and SSL certificates

### Docker Deployment (Optional)
```dockerfile
# Create Dockerfile for containerized deployment
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## �️ Development

### Code Quality Standards
- **TypeScript**: Strict type checking enabled
- **ESLint**: Next.js recommended configuration
- **Code Formatting**: Consistent style throughout
- **Error Handling**: Proper error boundaries and logging

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Create production build  
- `npm run start` - Start production server
- `npm run lint` - Run ESLint code analysis
- `npm run lint:fix` - Fix auto-fixable ESLint issues
- `npm run type-check` - Run TypeScript type checking
- `npm run ml-model` - Execute ML model training/prediction

### Environment Setup
1. Copy `.env.example` to `.env.local`
2. Update database and email configurations
3. Install dependencies: `npm install`
4. Start development: `npm run dev`

## �📞 Support

For technical support and questions:
- **Internal**: Contact the GenAI Visionaries team
- **NetApp IT**: Submit ticket through standard channels
- **Documentation**: Refer to this README and inline code comments

## 📄 License

This project is proprietary to NetApp, Inc. All rights reserved.

---

## 🏆 Project Achievements

✅ **Complete Full-Stack Application**
✅ **AI/ML Integration with Python**
✅ **Professional NetApp Branding**
✅ **MongoDB Database Integration**
✅ **Email Notification System**
✅ **CSV Import/Export Functionality**
✅ **Responsive Dashboard Interface**
✅ **RESTful API Architecture**
✅ **Real-time Analytics**
✅ **JIRA/Azure Boards Ready**

**Built with ❤️ by GenAI Visionaries Team**
**Powered by NetApp Innovation**