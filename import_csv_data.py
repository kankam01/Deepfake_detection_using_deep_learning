#!/usr/bin/env python
"""
Script to import CSV training data for the fake news detection model.
Place your CSV file in the data/ directory and update the filename below.
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

def import_csv_data(csv_filename):
    """Import training data from CSV file"""
    
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
        
        # Ask user to map columns
        print(f"\nPlease specify which columns contain:")
        
        # Try to auto-detect common column names
        title_col = None
        content_col = None
        label_col = None
        source_col = None
        
        # Auto-detect common column names
        for col in df.columns:
            col_lower = col.lower()
            if 'title' in col_lower or 'headline' in col_lower:
                title_col = col
            elif 'content' in col_lower or 'text' in col_lower or 'body' in col_lower:
                content_col = col
            elif 'label' in col_lower or 'class' in col_lower or 'fake' in col_lower or 'real' in col_lower:
                label_col = col
            elif 'source' in col_lower or 'url' in col_lower:
                source_col = col
        
        # If auto-detection failed, ask user
        if not title_col:
            title_col = input("Column name for article title: ")
        if not content_col:
            content_col = input("Column name for article content: ")
        if not label_col:
            label_col = input("Column name for fake/real label: ")
        
        print(f"\nUsing columns:")
        print(f"Title: {title_col}")
        print(f"Content: {content_col}")
        print(f"Label: {label_col}")
        if source_col:
            print(f"Source: {source_col}")
        
        # Ask for label mapping
        print(f"\nLabel values in your CSV:")
        unique_labels = df[label_col].unique()
        print(f"Found labels: {unique_labels}")
        
        fake_label = input("Which value represents FAKE news? (or press Enter if using 1/0): ")
        real_label = input("Which value represents REAL news? (or press Enter if using 1/0): ")
        
        # Default mapping for 1/0
        if not fake_label and not real_label:
            fake_label = 1
            real_label = 0
        
        # Import data
        print(f"\nImporting training data...")
        imported_count = 0
        skipped_count = 0
        
        for index, row in df.iterrows():
            try:
                # Get values from CSV
                title = str(row[title_col]) if pd.notna(row[title_col]) else ""
                content = str(row[content_col]) if pd.notna(row[content_col]) else ""
                label_value = row[label_col]
                source = str(row[source_col]) if source_col and pd.notna(row[source_col]) else ""
                
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
                        'source': source[:200],  # Limit source length
                        'is_fake': is_fake,
                        'added_by': admin_user
                    }
                )
                imported_count += 1
                
                if imported_count % 100 == 0:
                    print(f"Imported {imported_count} samples...")
                    
            except Exception as e:
                print(f"Error processing row {index + 1}: {e}")
                skipped_count += 1
        
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
    # Update this filename to match your CSV file
    csv_filename = "fake_real_news_78k.csv"  # Change this to your filename
    
    print("=== CSV Training Data Import ===")
    print(f"Looking for CSV file: data/{csv_filename}")
    print("If your file has a different name, update the csv_filename variable in this script.")
    
    import_csv_data(csv_filename) 