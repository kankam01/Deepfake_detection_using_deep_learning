from django.urls import path
from . import views

app_name = 'news_analyzer'

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze/', views.analyze_article, name='analyze_article'),
    path('analyze-url/', views.analyze_url, name='analyze_url'),
    path('training/', views.training_dashboard, name='training_dashboard'),
    path('training/add/', views.add_training_data, name='add_training_data'),
    path('training/train/', views.train_model, name='train_model'),
    path('results/', views.results_dashboard, name='results_dashboard'),
    path('article/<int:article_id>/', views.article_detail, name='article_detail'),
    path('feedback/submit/', views.submit_feedback, name='submit_feedback'),
    path('feedback/', views.feedback_dashboard, name='feedback_dashboard'),

] 