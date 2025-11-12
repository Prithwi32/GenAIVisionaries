import pandas as pd
import json
import re

def parse_skills_from_csv(skills_str):
    """Parse skills string from CSV format."""
    if pd.isna(skills_str) or skills_str == '':
        return []
    
    try:
        # Handle different skill formats
        if skills_str.startswith('[') and skills_str.endswith(']'):
            # JSON format: [{"skill":"python","level":8}]
            return json.loads(skills_str)
        else:
            # Simple format: "Linux;Docker;Kubernetes" or "python,javascript,sql"
            separator = ';' if ';' in skills_str else ','
            skills = [skill.strip() for skill in skills_str.split(separator)]
            # Convert to skill objects with levels (assign default level of 7)
            return [{"skill": skill, "level": 7} for skill in skills if skill]
    except:
        # Fallback parsing
        separator = ';' if ';' in skills_str else ','
        skills = [skill.strip() for skill in skills_str.split(separator)]
        return [{"skill": skill, "level": 7} for skill in skills if skill]

def parse_required_skills_from_csv(skills_str):
    """Parse required skills from task CSV format."""
    if pd.isna(skills_str) or skills_str == '':
        return []
    
    try:
        # Handle different formats
        if skills_str.startswith('[') and skills_str.endswith(']'):
            # JSON array format: ["python","security"]
            return json.loads(skills_str)
        else:
            # Simple format: "Quality Assurance;Bug Tracking;Testing"
            separator = ';' if ';' in skills_str else ','
            skills = [skill.strip() for skill in skills_str.split(separator)]
            return [skill for skill in skills if skill]
    except:
        # Fallback parsing
        separator = ';' if ';' in skills_str else ','
        skills = [skill.strip() for skill in skills_str.split(separator)]
        return [skill for skill in skills if skill]

def process_users_csv(csv_file):
    """Process Employee_Dataset.csv and return standardized data."""
    df = pd.read_csv(csv_file)
    
    users = []
    for _, row in df.iterrows():
        # Handle workload field - remove % if it exists
        workload_value = row['Current Workload (%)']
        if isinstance(workload_value, str):
            workload_value = float(workload_value.replace('%', ''))
        else:
            workload_value = float(workload_value)
        
        user = {
            'ssoId': str(row['SSO ID']),
            'name': row['Full Name'],
            'email': row['Email'],
            'role': row['Role'],
            'skills': parse_skills_from_csv(row['Skills']),
            'experienceYears': int(row['Experience (Years)']),
            'currentWorkload': workload_value,
            'isAvailable': workload_value < 95,  # Consider available if workload < 95%
            'department': row['Department']
        }
        users.append(user)
    
    return users

def process_tasks_csv(csv_file):
    """Process UserStory_Dataset.csv and return standardized data."""
    df = pd.read_csv(csv_file)
    
    tasks = []
    for _, row in df.iterrows():
        # Map difficulty level text to numbers
        difficulty_map = {'Easy': 3, 'Medium': 5, 'Hard': 8}
        difficulty = difficulty_map.get(row.get('Difficulty Level', 'Medium'), 5)
        
        # Use the Story Points directly from the dataset
        story_points = int(row.get('Story Points', 5))
        estimated_hours = float(row.get('Estimated Hours', 20))
        
        task = {
            'title': row['Task Title'],
            'description': row['Description (User story/ Problem description)'],
            'storyPoints': story_points,
            'difficultyLevel': difficulty,
            'requiredSkills': parse_required_skills_from_csv(row['Required Skills']),
            'priority': row.get('Priority', 'Medium'),
            'status': row.get('Status', 'To Do'),
            'estimatedHours': estimated_hours,
            'project': 'NetApp Project'  # Default project name
        }
        tasks.append(task)
    
    return tasks

def export_users_to_csv(users, output_file):
    """Export users to CSV format matching the import structure."""
    df_data = []
    for user in users:
        # Convert skills array back to semicolon-separated string
        skills_str = ';'.join([skill['skill'] for skill in user.get('skills', [])])
        
        df_data.append({
            'SSO ID': user['ssoId'],
            'Full Name': user['name'],
            'Email': user['email'],
            'Role': user['role'],
            'Experience (Years)': user['experienceYears'],
            'Current Workload (%)': user['currentWorkload'],
            'Department': user['department'],
            'Skills': skills_str
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_file, index=False)
    return output_file

def export_tasks_to_csv(tasks, output_file):
    """Export tasks to CSV format matching the import structure."""
    df_data = []
    for task in tasks:
        # Map difficulty numbers back to text
        difficulty_map = {3: 'Easy', 5: 'Medium', 8: 'Hard'}
        difficulty_text = difficulty_map.get(task.get('difficultyLevel', 5), 'Medium')
        
        # Convert required skills array to semicolon-separated string
        skills_str = ';'.join(task.get('requiredSkills', []))
        
        # Use existing estimated hours or calculate from story points
        estimated_hours = task.get('estimatedHours', task.get('storyPoints', 5) * 8)
        
        df_data.append({
            'Task Title': task['title'],
            'Description (User story/ Problem description)': task['description'],
            'Estimated Hours': estimated_hours,
            'Story Points': task.get('storyPoints', 5),
            'Difficulty Level': difficulty_text,
            'Priority': task.get('priority', 'Medium'),
            'Status': task.get('status', 'To Do'),
            'Required Skills': skills_str
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_file, index=False)
    return output_file

# Legacy function for backward compatibility
def process_csv_files(users_csv_path, tasks_csv_path, output_path):
    """Process CSV files and generate task assignments"""
    
    try:
        from main import TaskAssignmentModel
        
        # Read CSV files using our new processors
        users = process_users_csv(users_csv_path)
        tasks = process_tasks_csv(tasks_csv_path)
        
        print(f"Loaded {len(users)} users and {len(tasks)} tasks")
        
        # Convert to DataFrames for model compatibility
        users_df = pd.DataFrame(users)
        tasks_df = pd.DataFrame(tasks)
        
        # Initialize model
        model = TaskAssignmentModel()
        
        # Preprocess data
        users_clean, tasks_clean = model.preprocess_data(users_df, tasks_df)
        
        # Train models
        print("Training priority model...")
        model.train_priority_model(tasks_clean)
        
        print("Training deadline model...")
        model.train_deadline_model(tasks_clean)
        
        # Make predictions
        print("Predicting priorities...")
        priorities = model.predict_priorities(tasks_clean)
        
        print("Assigning tasks...")
        assignments = model.assign_tasks(users_clean, tasks_clean)
        
        # Create output DataFrame
        output_data = []
        for i, (_, task) in enumerate(tasks_clean.iterrows()):
            assignment = next((a for a in assignments if str(a['taskId']) == str(task.name)), None)
            
            output_row = {
                'Task ID': task.name,
                'Task Title': task['title'],
                'Task Description': task['description'],
                'Story Points': task['storyPoints'],
                'Difficulty Level': task['difficultyLevel'],
                'Predicted Priority': priorities[i] if i < len(priorities) else 'medium',
                'Assigned User SSO ID': assignment['userSsoId'] if assignment else 'Unassigned',
                'Assigned User Name': assignment['userName'] if assignment else 'Unassigned',
                'Assigned User Email': assignment['userEmail'] if assignment else '',
                'Confidence Score': assignment['confidenceScore'] if assignment else 0,
                'Estimated Deadline': assignment['estimatedDeadline'] if assignment else '',
                'Estimated Days': assignment['estimatedDays'] if assignment else ''
            }
            output_data.append(output_row)
        
        # Save to CSV
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_path, index=False)
        
        print(f"Results saved to {output_path}")
        
        # Also save as JSON for API consumption
        json_output_path = output_path.replace('.csv', '.json')
        output_df.to_json(json_output_path, orient='records', indent=2)
        
        return {
            'success': True,
            'message': f'Processed successfully. Output saved to {output_path}',
            'assignments_count': len(assignments),
            'unassigned_count': len(tasks_clean) - len(assignments)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }