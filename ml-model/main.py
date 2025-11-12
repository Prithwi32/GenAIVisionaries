import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime, timedelta

class TaskAssignmentModel:
    def __init__(self):
        self.priority_model = None
        self.assignment_model = None
        self.deadline_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models_trained = False
        
    def preprocess_data(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess user and task data for ML training"""
        
        # Clean users data
        users_df = users_df.copy()
        users_df['skills_text'] = users_df['skills'].apply(
            lambda x: ' '.join([skill['skill'] for skill in x]) if isinstance(x, list) else ''
        )
        users_df['avg_skill_level'] = users_df['skills'].apply(
            lambda x: np.mean([skill['level'] for skill in x]) if isinstance(x, list) and len(x) > 0 else 0
        )
        
        # Clean tasks data
        tasks_df = tasks_df.copy()
        tasks_df['required_skills_text'] = tasks_df['requiredSkills'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        
        # Fill missing values
        tasks_df['storyPoints'] = tasks_df['storyPoints'].fillna(tasks_df['storyPoints'].median())
        tasks_df['difficultyLevel'] = tasks_df['difficultyLevel'].fillna(5)
        users_df['currentWorkload'] = users_df['currentWorkload'].fillna(0)
        
        return users_df, tasks_df
    
    def extract_features(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> np.ndarray:
        """Extract features for task-user matching"""
        
        features = []
        
        for _, task in tasks_df.iterrows():
            task_skills = set(task['required_skills_text'].lower().split())
            
            for _, user in users_df.iterrows():
                user_skills = set(user['skills_text'].lower().split())
                
                # Calculate skill match score
                skill_overlap = len(task_skills.intersection(user_skills))
                skill_match_score = skill_overlap / max(len(task_skills), 1)
                
                # Experience relevance
                exp_score = min(user['experienceYears'] / 10, 1.0)
                
                # Workload factor (lower is better)
                workload_factor = 1 - (user['currentWorkload'] / 100)
                
                # Availability
                availability = 1 if user['isAvailable'] else 0
                
                # Feature vector
                feature_vector = [
                    skill_match_score,
                    exp_score,
                    workload_factor,
                    availability,
                    user['avg_skill_level'] / 10,
                    task['storyPoints'] / 100,
                    task['difficultyLevel'] / 10
                ]
                
                features.append(feature_vector)
        
        return np.array(features)
    
    def train_priority_model(self, tasks_df: pd.DataFrame):
        """Train priority assignment model"""
        
        # Prepare features for priority prediction
        text_features = self.tfidf_vectorizer.fit_transform(
            tasks_df['description'] + ' ' + tasks_df['required_skills_text']
        ).toarray()
        
        numerical_features = tasks_df[['storyPoints', 'difficultyLevel']].values
        
        X = np.hstack([text_features, numerical_features])
        
        # Create synthetic priority labels based on difficulty and story points
        # This would ideally come from historical data
        y = []
        for _, task in tasks_df.iterrows():
            score = task['storyPoints'] * 0.3 + task['difficultyLevel'] * 0.7
            if score >= 8:
                priority = 'critical'
            elif score >= 6:
                priority = 'high'
            elif score >= 4:
                priority = 'medium'
            else:
                priority = 'low'
            y.append(priority)
        
        # Encode labels
        self.label_encoders['priority'] = LabelEncoder()
        y_encoded = self.label_encoders['priority'].fit_transform(y)
        
        # Train model
        self.priority_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.priority_model.fit(X, y_encoded)
    
    def train_deadline_model(self, tasks_df: pd.DataFrame):
        """Train deadline estimation model"""
        
        # Create synthetic deadline data based on complexity
        # In real scenario, this would come from historical completion times
        completion_times = []
        for _, task in tasks_df.iterrows():
            base_time = task['storyPoints'] * task['difficultyLevel'] * 0.5
            completion_time = max(1, base_time + np.random.normal(0, base_time * 0.2))
            completion_times.append(completion_time)
        
        X = tasks_df[['storyPoints', 'difficultyLevel']].values
        y = np.array(completion_times)
        
        self.deadline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.deadline_model.fit(X, y)
    
    def predict_priorities(self, tasks_df: pd.DataFrame) -> List[str]:
        """Predict priorities for tasks"""
        
        if not self.priority_model:
            raise ValueError("Priority model not trained")
        
        text_features = self.tfidf_vectorizer.transform(
            tasks_df['description'] + ' ' + tasks_df['required_skills_text']
        ).toarray()
        
        numerical_features = tasks_df[['storyPoints', 'difficultyLevel']].values
        X = np.hstack([text_features, numerical_features])
        
        predictions = self.priority_model.predict(X)
        return self.label_encoders['priority'].inverse_transform(predictions)
    
    def assign_tasks(self, users_df: pd.DataFrame, tasks_df: pd.DataFrame) -> List[Dict]:
        """Assign tasks to users optimally"""
        
        assignments = []
        user_workloads = users_df['currentWorkload'].copy()
        
        for _, task in tasks_df.iterrows():
            best_user = None
            best_score = -1
            
            task_skills = set(str(task['required_skills_text']).lower().split())
            
            for idx, user in users_df.iterrows():
                if not user['isAvailable'] or user_workloads[idx] >= 90:
                    continue
                
                user_skills = set(str(user['skills_text']).lower().split())
                
                # Calculate matching score
                skill_overlap = len(task_skills.intersection(user_skills))
                skill_match_score = skill_overlap / max(len(task_skills), 1) if len(task_skills) > 0 else 0
                
                # Experience bonus
                exp_bonus = min(user['experienceYears'] / 10, 1.0)
                
                # Workload penalty
                workload_penalty = user_workloads[idx] / 100
                
                # Overall score
                total_score = (skill_match_score * 0.5 + 
                             exp_bonus * 0.3 + 
                             (1 - workload_penalty) * 0.2)
                
                if total_score > best_score:
                    best_score = total_score
                    best_user = user
                    best_user_idx = idx
            
            if best_user is not None:
                # Estimate deadline
                estimated_days = self.estimate_deadline(task)
                deadline = datetime.now() + timedelta(days=estimated_days)
                
                assignment = {
                    'taskId': task.get('_id', task.name),
                    'userId': best_user.get('_id', best_user.name),
                    'userSsoId': best_user['ssoId'],
                    'userName': best_user['name'],
                    'userEmail': best_user['email'],
                    'confidenceScore': best_score,
                    'estimatedDeadline': deadline.isoformat(),
                    'estimatedDays': estimated_days
                }
                
                assignments.append(assignment)
                
                # Update workload
                task_weight = task['storyPoints'] * task['difficultyLevel'] / 10
                user_workloads[best_user_idx] += task_weight
        
        return assignments
    
    def estimate_deadline(self, task) -> float:
        """Estimate completion time in days"""
        
        if self.deadline_model:
            X = np.array([[task['storyPoints'], task['difficultyLevel']]])
            estimated_days = self.deadline_model.predict(X)[0]
        else:
            # Fallback calculation
            estimated_days = task['storyPoints'] * task['difficultyLevel'] * 0.5
        
        return max(1, estimated_days)
    
    def save_models(self, model_dir: str):
        """Save trained models"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        if self.priority_model:
            joblib.dump(self.priority_model, os.path.join(model_dir, 'priority_model.pkl'))
        
        if self.deadline_model:
            joblib.dump(self.deadline_model, os.path.join(model_dir, 'deadline_model.pkl'))
        
        joblib.dump(self.tfidf_vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str):
        """Load pre-trained models"""
        
        try:
            self.priority_model = joblib.load(os.path.join(model_dir, 'priority_model.pkl'))
            self.deadline_model = joblib.load(os.path.join(model_dir, 'deadline_model.pkl'))
            self.tfidf_vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
            self.label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
            self.models_trained = True
            print(f"Models loaded from {model_dir}")
        except Exception as e:
            print(f"Error loading models: {e}")


def main():
    """Main function for testing the model with real datasets"""
    
    try:
        from dataset_processor import process_users_csv, process_tasks_csv
        
        # Load real datasets
        print("Loading Employee Dataset...")
        users_data = process_users_csv('Employee_Dataset.csv')
        users_df = pd.DataFrame(users_data)
        
        print("Loading User Story Dataset...")
        tasks_data = process_tasks_csv('UserStory_Dataset.csv')
        tasks_df = pd.DataFrame(tasks_data)
        
        print(f"Loaded {len(users_df)} employees and {len(tasks_df)} tasks from real datasets")
        
        # Initialize and train model
        model = TaskAssignmentModel()
        
        # Preprocess data
        users_clean, tasks_clean = model.preprocess_data(users_df, tasks_df)
        
        print("Training models...")
        # Train models
        model.train_priority_model(tasks_clean)
        model.train_deadline_model(tasks_clean)
        
        # Make predictions on a subset for demonstration
        test_tasks = tasks_clean.head(10)  # Test with first 10 tasks
        
        print("Making predictions...")
        priorities = model.predict_priorities(test_tasks)
        assignments = model.assign_tasks(users_clean, test_tasks)
        
        print(f"\nSample Priority Predictions:")
        for i, (_, task) in enumerate(test_tasks.iterrows()):
            print(f"- {task['title']}: {priorities[i]}")
        
        print(f"\nSample Task Assignments:")
        for assignment in assignments[:5]:  # Show first 5 assignments
            print(f"- Task: {assignment.get('taskId', 'Unknown')}")
            print(f"  Assigned to: {assignment['userName']} ({assignment['userSsoId']})")
            print(f"  Confidence: {assignment['confidenceScore']:.2f}")
            print(f"  Estimated days: {assignment['estimatedDays']:.1f}")
            print()
        
        # Save models
        model.save_models('./models')
        
        return {
            'success': True,
            'users_count': len(users_df),
            'tasks_count': len(tasks_df),
            'assignments_count': len(assignments),
            'sample_assignments': assignments[:3]
        }
        
    except Exception as e:
        print(f"Error in main: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    result = main()
    print("Final result:", result)