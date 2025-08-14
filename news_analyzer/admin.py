from django.contrib import admin
from .models import NewsArticle, TrainingData, ModelPerformance, ModelFile, AnalysisFeedback

@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'source', 'is_fake', 'confidence_score', 'created_at', 'analyzed_at')
    list_filter = ('is_fake', 'source', 'created_at', 'analyzed_at')
    search_fields = ('title', 'content', 'source')
    readonly_fields = ('created_at', 'analyzed_at')
    list_per_page = 20
    
    fieldsets = (
        ('Article Information', {
            'fields': ('title', 'content', 'source', 'url')
        }),
        ('Analysis Results', {
            'fields': ('is_fake', 'confidence_score', 'analyzed_at')
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    list_display = ('title', 'is_fake', 'source', 'added_by', 'created_at')
    list_filter = ('is_fake', 'source', 'created_at')
    search_fields = ('title', 'content', 'source')
    readonly_fields = ('created_at',)
    list_per_page = 20
    
    fieldsets = (
        ('Training Data', {
            'fields': ('title', 'content', 'is_fake', 'source')
        }),
        ('User Information', {
            'fields': ('added_by', 'created_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'training_samples', 'test_samples', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('model_name',)
    readonly_fields = ('created_at',)
    list_per_page = 20
    
    fieldsets = (
        ('Model Information', {
            'fields': ('model_name',)
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score')
        }),
        ('Data Information', {
            'fields': ('training_samples', 'test_samples', 'created_at')
        }),
    )

@admin.register(ModelFile)
class ModelFileAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'is_active', 'created_at')
    list_filter = ('model_type', 'is_active', 'created_at')
    search_fields = ('name', 'model_type')
    readonly_fields = ('created_at',)
    list_per_page = 20
    
    fieldsets = (
        ('Model File Information', {
            'fields': ('name', 'file_path', 'model_type', 'is_active')
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

@admin.register(AnalysisFeedback)
class AnalysisFeedbackAdmin(admin.ModelAdmin):
    list_display = ('article', 'user_correction', 'created_at')
    list_filter = ('user_correction', 'created_at')
    search_fields = ('article__title', 'feedback_comment')
    readonly_fields = ('created_at',)
    list_per_page = 20
    
    fieldsets = (
        ('Feedback Information', {
            'fields': ('article', 'user_correction', 'feedback_comment')
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )



# Customize admin site
admin.site.site_header = "Fake News Detector Admin"
admin.site.site_title = "Fake News Detector"
admin.site.index_title = "Welcome to Fake News Detector Administration"
