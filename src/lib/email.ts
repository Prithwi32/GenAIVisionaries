import nodemailer from 'nodemailer'

interface TaskAssignmentEmailData {
  userEmail: string
  userName: string
  taskTitle: string
  taskDescription: string
  deadline?: string
  priority: string
}

/**
 * Creates email transporter with OAuth2 or SMTP fallback
 * Supports both enterprise OAuth2 and basic SMTP authentication
 */
const createTransporter = () => {
  // Check if OAuth2 is configured
  if (process.env.EMAIL_CLIENT_ID && process.env.EMAIL_CLIENT_SECRET && process.env.EMAIL_REFRESH_TOKEN) {
    return nodemailer.createTransport({
      service: 'outlook',
      auth: {
        type: 'OAuth2',
        user: process.env.EMAIL_USER,
        clientId: process.env.EMAIL_CLIENT_ID,
        clientSecret: process.env.EMAIL_CLIENT_SECRET,
        refreshToken: process.env.EMAIL_REFRESH_TOKEN,
      },
    })
  }
  
  // Fallback to SMTP authentication
  return nodemailer.createTransport({
    host: process.env.EMAIL_HOST || 'smtp.office365.com', // Updated to Office365 SMTP
    port: parseInt(process.env.EMAIL_PORT || '587'),
    secure: false, // true for 465, false for other ports
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS,
    },
    tls: {
      ciphers: 'SSLv3',
      rejectUnauthorized: false, // Allow self-signed certificates in dev
    },
    // Additional options for Outlook compatibility
    requireTLS: true,
    connectionTimeout: 60000, // 60 seconds
    greetingTimeout: 30000, // 30 seconds
    socketTimeout: 60000, // 60 seconds
  })
}

/**
 * Sends task assignment email to specified user
 * @param data Task assignment email data
 * @returns Promise with success status and message ID or error
 */
export async function sendTaskAssignmentEmail(data: TaskAssignmentEmailData) {
  // Validate email configuration
  if (!process.env.EMAIL_USER) {
    return { success: false, error: 'Email user not configured' }
  }

  // Check authentication methods
  const hasOAuth2 = process.env.EMAIL_CLIENT_ID && process.env.EMAIL_CLIENT_SECRET && process.env.EMAIL_REFRESH_TOKEN
  const hasPassword = process.env.EMAIL_PASS
  
  if (!hasOAuth2 && !hasPassword) {
    return { success: false, error: 'Email authentication not configured' }
  }

  try {
    const transporter = createTransporter()
    
    // Verify connection before sending
    await transporter.verify()
    
    const deadlineText = data.deadline 
      ? new Date(data.deadline).toLocaleDateString('en-US', {
          weekday: 'long',
          year: 'numeric',
          month: 'long',
          day: 'numeric'
        })
      : 'Not specified'
    
    const priorityColor = {
      'critical': '#dc2626',
      'high': '#ea580c', 
      'medium': '#ca8a04',
      'low': '#16a34a'
    }[data.priority.toLowerCase()] || '#6b7280'
    
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>New Task Assignment</title>
        <style>
          body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f8fafc;
          }
          .container {
            max-width: 600px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
          }
          .header {
            background: linear-gradient(135deg, #0067C5 0%, #00A1C9 100%);
            color: white;
            padding: 20px;
            text-align: center;
          }
          .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
          }
          .content {
            padding: 30px;
          }
          .greeting {
            font-size: 18px;
            margin-bottom: 20px;
          }
          .task-card {
            background-color: #f8fafc;
            border-left: 4px solid #0067C5;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
          }
          .task-title {
            font-size: 20px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 10px;
          }
          .task-description {
            color: #6b7280;
            margin-bottom: 15px;
            line-height: 1.5;
          }
          .task-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 15px;
          }
          .meta-item {
            display: flex;
            align-items: center;
            font-size: 14px;
          }
          .priority-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: white;
            text-transform: uppercase;
            background-color: ${priorityColor};
          }
          .deadline {
            color: #374151;
            font-weight: 500;
          }
          .cta-section {
            text-align: center;
            margin: 30px 0;
          }
          .cta-button {
            display: inline-block;
            background: linear-gradient(135deg, #0067C5 0%, #00A1C9 100%);
            color: white;
            text-decoration: none;
            padding: 12px 30px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 16px;
          }
          .footer {
            background-color: #f1f5f9;
            padding: 20px;
            text-align: center;
            color: #64748b;
            font-size: 14px;
          }
          .netapp-logo {
            color: #0067C5;
            font-weight: 700;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🎯 New Task Assignment</h1>
            <p>GenAI Visionaries - Intelligent Task Assignment</p>
          </div>
          
          <div class="content">
            <div class="greeting">
              Hi <strong>${data.userName}</strong>,
            </div>
            
            <p>You have been assigned a new task by our AI-powered task assignment system. The system has analyzed your skills, experience, and current workload to determine that you're the best fit for this assignment.</p>
            
            <div class="task-card">
              <div class="task-title">${data.taskTitle}</div>
              <div class="task-description">${data.taskDescription}</div>
              
              <div class="task-meta">
                <div class="meta-item">
                  <strong>Priority:</strong>&nbsp;<span class="priority-badge">${data.priority}</span>
                </div>
                <div class="meta-item">
                  <strong>Deadline:</strong>&nbsp;<span class="deadline">${deadlineText}</span>
                </div>
              </div>
            </div>
            
            <p>Please log in to the GenAI Visionaries dashboard to view more details, update the task status, and collaborate with your team.</p>
            
            <div class="cta-section">
              <a href="${process.env.NEXT_PUBLIC_APP_URL}/dashboard/tasks" class="cta-button">
                View Task Details
              </a>
            </div>
            
            <p><em>This assignment was made automatically by our intelligent task assignment system based on skill matching, workload analysis, and optimal resource allocation.</em></p>
          </div>
          
          <div class="footer">
            <p>This email was sent by <span class="netapp-logo">GenAI Visionaries</span> - Powered by NetApp</p>
            <p>Intelligent Task Assignment System</p>
          </div>
        </div>
      </body>
      </html>
    `
    
    const textContent = `
New Task Assignment - GenAI Visionaries

Hi ${data.userName},

You have been assigned a new task by our AI-powered task assignment system.

Task: ${data.taskTitle}
Description: ${data.taskDescription}
Priority: ${data.priority.toUpperCase()}
Deadline: ${deadlineText}

Please log in to the GenAI Visionaries dashboard to view more details and update the task status.

Dashboard: ${process.env.NEXT_PUBLIC_APP_URL}/dashboard/tasks

This assignment was made automatically by our intelligent task assignment system based on skill matching, workload analysis, and optimal resource allocation.

---
GenAI Visionaries - Powered by NetApp
Intelligent Task Assignment System
    `
    
    const mailOptions = {
      from: `"GenAI Visionaries" <${process.env.EMAIL_USER}>`,
      to: data.userEmail,
      subject: `🎯 New Task Assignment: ${data.taskTitle}`,
      text: textContent,
      html: htmlContent,
    }
    
    const info = await transporter.sendMail(mailOptions)
    
    return { success: true, messageId: info.messageId }
    
  } catch (error) {
    // Handle email sending errors with specific guidance
    let errorMessage = error instanceof Error ? error.message : 'Unknown error'
    
    if (errorMessage.includes('SmtpClientAuthentication is disabled')) {
      errorMessage = 'SMTP authentication disabled by organization. Contact IT for configuration.'
    } else if (errorMessage.includes('535 5.7.3 Authentication unsuccessful')) {
      errorMessage = 'Email authentication failed. Verify credentials or use App Password.'
    } else if (errorMessage.includes('EAUTH')) {
      errorMessage = 'Email authentication error. Check SMTP settings and credentials.'
    }
    
    return { success: false, error: errorMessage }
  }
}

export async function sendBulkTaskAssignmentEmails(assignments: TaskAssignmentEmailData[]) {
  const results = []
  
  for (const assignment of assignments) {
    const result = await sendTaskAssignmentEmail(assignment)
    results.push({ ...assignment, emailResult: result })
    
    // Add delay between emails to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 1000))
  }
  
  return results
}