#!/usr/bin/env python
"""
Automated script to import CSV training data for the fake news detection model.
This version automatically handles the label mapping for fake_real_news_78k.csv
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

def import_csv_data_auto(csv_filename):
    """Import training data from CSV file with automatic label mapping"""
    
    # Get or create admin user
    admin_user, created = User.objects.get_or_create(
        username='admin',
        defaults={'email': 'admin@example.com', 'is_staff': True, 'is_superuser': True}
    )
    
    # Path to CSV file
    csv_path = Path('data') / csv_filename
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please place your CSV file in the 'data/' directory")
        return
    
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Display CSV structure
        print(f"\nCSV Structure:")
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        print(f"Sample data:")
        print(df.head())
        
        # Auto-detect columns for fake_real_news_78k.csv format
        title_col = 'title'
        content_col = 'text'
        label_col = 'label'
        
        print(f"\nUsing columns:")
        print(f"Title: {title_col}")
        print(f"Content: {content_col}")
        print(f"Label: {label_col}")
        
        # Auto-detect label mapping
        unique_labels = df[label_col].unique()
        print(f"Found labels: {unique_labels}")
        
        # For fake_real_news_78k.csv: 'FAKE' = fake news, 'TRUE' = real news
        fake_label = 'FAKE'
        real_label = 'TRUE'
        
        print(f"Auto-mapping labels:")
        print(f"FAKE news: '{fake_label}'")
        print(f"REAL news: '{real_label}'")
        
        # Import data
        print(f"\nImporting training data...")
        imported_count = 0
        skipped_count = 0
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}...")
            
            for index, row in batch_df.iterrows():
                try:
                    # Get values from CSV
                    title = str(row[title_col]) if pd.notna(row[title_col]) else ""
                    content = str(row[content_col]) if pd.notna(row[content_col]) else ""
                    label_value = row[label_col]
                    
                    # Determine if fake or real
                    if label_value == fake_label:
                        is_fake = True
                    elif label_value == real_label:
                        is_fake = False
                    else:
                        print(f"Warning: Unknown label value '{label_value}' in row {index + 1}")
                        continue
                    
                    # Skip if title or content is empty
                    if not title.strip() or not content.strip():
                        print(f"Warning: Empty title or content in row {index + 1}")
                        skipped_count += 1
                        continue
                    
                    # Create training data entry
                    TrainingData.objects.get_or_create(
                        title=title[:500],  # Limit title length
                        defaults={
                            'content': content,
                            'source': '',  # No source column in this dataset
                            'is_fake': is_fake,
                            'added_by': admin_user
                        }
                    )
                    imported_count += 1
                    
                except Exception as e:
                    print(f"Error processing row {index + 1}: {e}")
                    skipped_count += 1
            
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
        print(f"Total training samples: {total_samples}")
        print(f"Fake news samples: {total_fake}")
        print(f"Real news samples: {total_real}")
        print(f"Ready for model training!")
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Please check that your CSV file is properly formatted.")

if __name__ == '__main__':
    csv_filename = "fake_real_news_78k.csv"
    
    print("=== Automated CSV Training Data Import ===")
    print(f"Importing: {csv_filename}")
    print("This script will automatically handle the label mapping.")
    
    import_csv_data_auto(csv_filename) 