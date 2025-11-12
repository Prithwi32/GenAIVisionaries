import pandas as pd
import json
import sys
import os
from main import TaskAssignmentModel

def process_csv_files(users_csv_path, tasks_csv_path, output_path):
    """Process CSV files and generate task assignments"""
    
    try:
        # Read CSV files
        users_df = pd.read_csv(users_csv_path)
        tasks_df = pd.read_csv(tasks_csv_path)
        
        print(f"Loaded {len(users_df)} users and {len(tasks_df)} tasks")
        
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

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python csv_processor.py <users_csv> <tasks_csv> <output_csv>")
        sys.exit(1)
    
    users_csv = sys.argv[1]
    tasks_csv = sys.argv[2]
    output_csv = sys.argv[3]
    
    result = process_csv_files(users_csv, tasks_csv, output_csv)
    
    if result['success']:
        print("SUCCESS:", result['message'])
    else:
        print("ERROR:", result['error'])
        sys.exit(1)