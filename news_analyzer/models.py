from django.db import models
from django.contrib.auth.models import User
import json

class NewsArticle(models.Model):
    """Model to store news articles for analysis"""
    title = models.CharField(max_length=500)
    content = models.TextField()
    source = models.CharField(max_length=200, blank=True)
    url = models.URLField(blank=True)
    is_fake = models.BooleanField(null=True, blank=True)  # True for fake, False for real, None for unknown
    confidence_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title[:50]}... ({'Fake' if self.is_fake else 'Real' if self.is_fake is False else 'Unknown'})"



class TrainingData(models.Model):
    """Model to store training data for the ML model"""
    title = models.CharField(max_length=500)
    content = models.TextField()
    is_fake = models.BooleanField()  # True for fake, False for real
    source = models.CharField(max_length=200, blank=True)
    added_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title[:50]}... ({'Fake' if self.is_fake else 'Real'})"

class ModelPerformance(models.Model):
    """Model to track ML model performance metrics"""
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    training_samples = models.IntegerField()
    test_samples = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy:.3f} ({self.created_at.strftime('%Y-%m-%d')})"

class ModelFile(models.Model):
    """Model to store trained model files"""
    name = models.CharField(max_length=100)
    file_path = models.CharField(max_length=500)
    model_type = models.CharField(max_length=50)  # e.g., 'tfidf_svm', 'bert_classifier'
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"

class AnalysisFeedback(models.Model):
    """Model to track user feedback on analysis results"""
    article = models.ForeignKey(NewsArticle, on_delete=models.CASCADE)
    user_correction = models.BooleanField()  # True if user says it's fake, False if real
    feedback_comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback on {self.article.title[:30]}... - Corrected to: {'Fake' if self.user_correction else 'Real'}"
