#!/usr/bin/env python
"""
Script to import the new fake_news_dataset.csv into the training database
"""

import os
import django
import pandas as pd
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_news_detector_project.settings')
django.setup()

from news_analyzer.models import TrainingData
from django.contrib.auth.models import User

def import_new_dataset():
    """Import the new fake_news_dataset.csv"""
    
    # Get or create admin user
    admin_user, created = User.objects.get_or_create(
        username='admin',
        defaults={'email': 'admin@example.com', 'is_staff': True, 'is_superuser': True}
    )
    
    # Path to CSV file
    csv_path = Path('data') / 'fake_news_dataset.csv'
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check label distribution
        print(f"Label distribution:")
        print(df['label'].value_counts())
        
        # Map labels - the dataset uses 'fake' and 'real' (lowercase)
        fake_label = 'fake'
        real_label = 'real'
        
        print(f"\nMapping labels:")
        print(f"FAKE news: '{fake_label}'")
        print(f"REAL news: '{real_label}'")
        
        # Clear existing training data to start fresh
        print("\nClearing existing training data...")
        TrainingData.objects.all().delete()
        
        # Import data in batches
        print(f"\nImporting training data...")
        imported_count = 0
        skipped_count = 0
        batch_size = 1000
        
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}...")
            
            batch_objects = []
            
            for index, row in batch_df.iterrows():
                try:
                    # Get values from CSV
                    title = str(row['title']) if pd.notna(row['title']) else ""
                    content = str(row['text']) if pd.notna(row['text']) else ""
                    label_value = str(row['label']).lower().strip()
                    source = str(row['source']) if pd.notna(row['source']) else ""
                    
                    # Determine if fake or real
                    if label_value == fake_label:
                        is_fake = True
                    elif label_value == real_label:
                        is_fake = False
                    else:
                        print(f"Warning: Unknown label value '{label_value}' in row {index + 1}")
                        skipped_count += 1
                        continue
                    
                    # Skip if title or content is empty
                    if not title.strip() or not content.strip():
                        print(f"Warning: Empty title or content in row {index + 1}")
                        skipped_count += 1
                        continue
                    
                    # Create training data object
                    training_obj = TrainingData(
                        title=title[:500],  # Limit title length
                        content=content,
                        source=source[:200],  # Limit source length
                        is_fake=is_fake,
                        added_by=admin_user
                    )
                    batch_objects.append(training_obj)
                    imported_count += 1
                    
                except Exception as e:
                    print(f"Error processing row {index + 1}: {e}")
                    skipped_count += 1
            
            # Bulk create the batch
            if batch_objects:
                TrainingData.objects.bulk_create(batch_objects, ignore_conflicts=True)
            
            # Print progress
            if imported_count % 5000 == 0:
                print(f"Imported {imported_count} samples so far...")
        
        # Print summary
        total_fake = TrainingData.objects.filter(is_fake=True).count()
        total_real = TrainingData.objects.filter(is_fake=False).count()
        total_samples = TrainingData.objects.count()
        
        print(f"\n=== Import Summary ===")
        print(f"Successfully imported: {imported_count}")
        print(f"Skipped rows: {skipped_count}")
        print(f"Total training samples in database: {total_samples}")
        print(f"Fake news samples: {total_fake}")
        print(f"Real news samples: {total_real}")
        print(f"Dataset balance: {total_fake/total_samples:.1%} fake, {total_real/total_samples:.1%} real")
        print(f"Ready for model training!")
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Please check that your CSV file is properly formatted.")

if __name__ == '__main__':
    print("=== New Dataset Import ===")
    print("Importing fake_news_dataset.csv into training database...")
    
    import_new_dataset()