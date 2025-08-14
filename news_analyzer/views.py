from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Count, Avg
import json
import pickle
import os
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

from .models import NewsArticle, TrainingData, ModelPerformance, ModelFile, AnalysisFeedback
from .services import NewsAPIService
from .enhanced_analysis import EnhancedAnalysisService

def home(request):
    """Home page with article analysis form"""
    recent_articles = NewsArticle.objects.all()[:5]
    stats = {
        'total_articles': NewsArticle.objects.count(),
        'fake_articles': NewsArticle.objects.filter(is_fake=True).count(),
        'real_articles': NewsArticle.objects.filter(is_fake=False).count(),
        'training_samples': TrainingData.objects.count(),
    }
    
    return render(request, 'news_analyzer/home.html', {
        'recent_articles': recent_articles,
        'stats': stats
    })

@csrf_exempt
def analyze_article(request):
    """Analyze a news article for fake news detection"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            title = data.get('title', '')
            content = data.get('content', '')
            source = data.get('source', '')
            url = data.get('url', '')
            
            if not content:
                return JsonResponse({'error': 'Article content is required'}, status=400)
            
            # Create article record
            article = NewsArticle.objects.create(
                title=title,
                content=content,
                source=source,
                url=url
            )
            
            # Analyze the article
            text_to_analyze = title + " " + content if title else content
            result = analyze_text_with_model(text_to_analyze)
            
            # Update article with results
            article.is_fake = result['is_fake']
            article.confidence_score = result['confidence']
            article.analyzed_at = timezone.now()
            article.save()
            
            return JsonResponse({
                'success': True,
                'article_id': article.id,
                'is_fake': result['is_fake'],
                'confidence': result['confidence'],
                'explanation': result['explanation']
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)

@csrf_exempt
def analyze_url(request):
    """Analyze a news article by URL with enhanced fact-checking"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            url = data.get('url', '').strip()
            
            if not url:
                return JsonResponse({'error': 'URL is required'}, status=400)
            
            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                return JsonResponse({'error': 'Please provide a valid URL starting with http:// or https://'}, status=400)
            
            # Initialize News API service
            news_service = NewsAPIService()
            
            # Extract article from URL using enhanced extraction
            article_info = news_service.extract_article_from_url(url)
            
            if not article_info:
                print(f"Failed to extract article from URL: {url}")
                return JsonResponse({'error': 'Could not extract article content from URL. Please try a different article or use manual input.'}, status=400)
            
            if not article_info.get('content') or len(article_info.get('content', '').strip()) < 50:
                print(f"Insufficient content extracted from URL: {url}")
                return JsonResponse({'error': 'Could not extract sufficient article content from URL. Please try a different article or use manual input.'}, status=400)
            
            print(f"Successfully extracted article from URL: {url}")
            print(f"Title: {article_info.get('title', 'No title')}")
            print(f"Source: {article_info.get('source', 'No source')}")
            print(f"Content length: {len(article_info.get('content', ''))}")
            print(f"Extraction method: {article_info.get('extraction_method', 'unknown')}")
            
            # Step 1: Enhanced fact-checking analysis
            fact_check_analysis = news_service.analyze_article_with_fact_checking(article_info)
            
            # Step 2: ML model analysis (existing functionality)
            text_to_analyze = article_info.get('title', '') + " " + article_info.get('content', '')
            ml_result = analyze_text_with_model(text_to_analyze)
            
            # Create article record
            article = NewsArticle.objects.create(
                title=article_info.get('title', ''),
                content=article_info.get('content', ''),
                source=article_info.get('source', ''),
                url=url
            )
            
            # Combine ML and fact-checking results for final verdict
            final_result = _combine_analysis_results(ml_result, fact_check_analysis)
            
            # Update article with results
            article.is_fake = final_result['is_fake']
            article.confidence_score = final_result['confidence']
            article.analyzed_at = timezone.now()
            article.save()
            
            # Enhanced explanation including the final verdict and all analysis components
            final_verdict_text = 'Likely Real' if final_result['is_fake'] is False else 'Likely Fake' if final_result['is_fake'] is True else 'Uncertain'
            explanation = (
                f"Final verdict: {final_verdict_text} (confidence {final_result['confidence']:.0%}). "
                + _build_comprehensive_explanation(ml_result, fact_check_analysis, article_info)
            )
            
            return JsonResponse({
                'success': True,
                'article_id': article.id,
                'is_fake': final_result['is_fake'],
                'confidence': final_result['confidence'],
                'explanation': explanation,
                'article_info': {
                    'title': article_info.get('title', ''),
                    'source': article_info.get('source', ''),
                    'source_reliability': fact_check_analysis.get('source_reliability', 'unknown') if fact_check_analysis else 'unknown',
                    'published_at': article_info.get('published_at', ''),
                    'author': article_info.get('author', ''),
                    'extraction_method': article_info.get('extraction_method', 'unknown')
                },
                'analysis_details': {
                    'ml_result': ml_result,
                    'fact_check_analysis': fact_check_analysis,
                    'extraction_method': article_info.get('extraction_method', 'unknown')
                }
            })
            
        except Exception as e:
            print(f"Error in analyze_url: {e}")
            return JsonResponse({'error': f'Error analyzing URL: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)

def _combine_analysis_results(ml_result, fact_check_analysis):
    """Combine ML and fact-checking results for final verdict"""
    # Start with ML result
    final_result = {
        'is_fake': ml_result.get('is_fake'),
        'confidence': ml_result.get('confidence', 0.5)
    }
    
    if not fact_check_analysis:
        return final_result
    
    # Adjust based on fact-checking results
    fact_check_verdict = fact_check_analysis.get('overall_verdict', 'unknown')
    
    if fact_check_verdict == 'likely_real' and final_result['is_fake']:
        # Fact-checking suggests real, but ML says fake - move confidence toward real
        final_result['confidence'] = min(0.95, final_result['confidence'] + 0.2)
    elif fact_check_verdict == 'likely_fake' and not final_result['is_fake']:
        # Fact-checking suggests fake, but ML says real - move confidence toward fake
        final_result['confidence'] = max(0.2, final_result['confidence'] - 0.2)
    elif fact_check_verdict == 'likely_real' and not final_result['is_fake']:
        # Both suggest real - increase confidence
        final_result['confidence'] = min(0.95, final_result['confidence'] + 0.1)
    elif fact_check_verdict == 'likely_fake' and final_result['is_fake']:
        # Both suggest fake - decrease confidence (toward fake)
        final_result['confidence'] = max(0.2, final_result['confidence'] - 0.1)
    
    # Apply the new threshold: confidence > 0.45 = real, < 0.45 = likely fake
    if final_result['confidence'] > 0.45:
        final_result['is_fake'] = False
    else:
        final_result['is_fake'] = True  # Below 45% confidence = likely fake
    
    return final_result

def _build_comprehensive_explanation(ml_result, fact_check_analysis, article_info):
    """Build comprehensive explanation combining all analysis methods"""
    explanation_parts = []
    
    # ML model explanation
    if ml_result:
        explanation_parts.append(f"ML Model: {ml_result.get('explanation', 'Analysis completed')}")
    
    # Fact-checking results
    if fact_check_analysis:
        source_reliability = fact_check_analysis.get('source_reliability', 'unknown')
        if source_reliability != 'unknown':
            explanation_parts.append(f"Source reliability: {source_reliability.capitalize()}")
        
        content_analysis = fact_check_analysis.get('content_analysis', {})
        if content_analysis:
            fake_indicators = content_analysis.get('fake_indicators_found', 0)
            credible_indicators = content_analysis.get('credible_indicators_found', 0)
            if fake_indicators > 0 or credible_indicators > 0:
                explanation_parts.append(f"Content analysis: Found {fake_indicators} potential fake news indicators and {credible_indicators} credible indicators")
        
        # Google Fact Check API results
        fact_check_claims = fact_check_analysis.get('fact_check_claims', [])
        if fact_check_claims:
            verified_claims = sum(1 for claim in fact_check_claims if 'true' in claim.get('rating', '').lower())
            false_claims = sum(1 for claim in fact_check_claims if any(word in claim.get('rating', '').lower() for word in ['false', 'fake', 'hoax']))
            if verified_claims > 0 or false_claims > 0:
                explanation_parts.append(f"Google Fact Check: Found {verified_claims} verified claims and {false_claims} false claims")
        
        fact_check_verdict = fact_check_analysis.get('overall_verdict', 'unknown')
        if fact_check_verdict != 'unknown':
            explanation_parts.append(f"Fact-checking verdict: {fact_check_verdict.replace('_', ' ').title()}")
    
    # Extraction method
    extraction_method = article_info.get('extraction_method', 'unknown')
    if extraction_method != 'unknown':
        explanation_parts.append(f"Content extracted using: {extraction_method.replace('_', ' ').title()}")
    
    return ". ".join(explanation_parts) if explanation_parts else "Analysis completed using multiple methods."

def analyze_text_with_model(text):
    """Analyze text using the enhanced model with logical reasoning"""
    try:
        # Use enhanced analysis service
        enhanced_service = EnhancedAnalysisService()
        result = enhanced_service.analyze_with_enhanced_model(text)
        
        if result:
            return result
        
        # Fallback to rule-based analysis if enhanced analysis fails
        return analyze_text_with_rules(text)
        
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        # Fallback to rule-based analysis
        return analyze_text_with_rules(text)

def analyze_text_with_rules(text):
    """Fallback rule-based analysis"""
    # Clean and preprocess text
    text = text.lower()
    
    # Enhanced heuristics with more indicators
    fake_indicators = [
        'fake news', 'conspiracy', 'shocking', 'you won\'t believe',
        'doctors hate', 'one weird trick', 'miracle cure', 'secret',
        'they don\'t want you to know', 'mainstream media lies',
        'clickbait', 'viral', 'going viral', 'shocking truth',
        'what they don\'t want you to know', 'insider information',
        'exclusive', 'breaking', 'urgent', 'last chance',
        'limited time', 'act now', 'don\'t miss out',
        'government cover-up', 'suppressed', 'censored',
        'alternative facts', 'fake media', 'fake news media'
    ]
    
    real_indicators = [
        'study shows', 'research indicates', 'according to experts',
        'peer-reviewed', 'scientific evidence', 'official statement',
        'verified sources', 'fact-checked', 'journal article',
        'academic study', 'university research', 'published in',
        'according to data', 'statistics show', 'official report',
        'government report', 'verified by', 'confirmed by',
        'multiple sources', 'reliable source', 'credible source'
    ]
    
    fake_score = sum(1 for indicator in fake_indicators if indicator in text)
    real_score = sum(1 for indicator in real_indicators if indicator in text)
    
    # Calculate confidence and determine if fake
    total_indicators = fake_score + real_score
    if total_indicators == 0:
        confidence = 0.5
        is_fake = None
    else:
        # Calculate confidence based on the ratio of the dominant score
        if fake_score > real_score:
            confidence = fake_score / total_indicators
            is_fake = True
        else:
            confidence = real_score / total_indicators
            is_fake = False
    
    # Apply 45% confidence threshold: below 45% = likely fake
    if confidence < 0.45:
        is_fake = True
        confidence = 1.0 - confidence  # Invert confidence for fake news
    
    explanation = f"Rule-based analysis: Found {fake_score} potential fake news indicators and {real_score} credible indicators."
    
    return {
        'is_fake': is_fake,
        'confidence': confidence,
        'explanation': explanation
    }

@login_required
def training_dashboard(request):
    """Dashboard for managing training data and model training"""
    training_data = TrainingData.objects.all()
    model_performance = ModelPerformance.objects.all()[:5]
    
    stats = {
        'total_samples': training_data.count(),
        'fake_samples': training_data.filter(is_fake=True).count(),
        'real_samples': training_data.filter(is_fake=False).count(),
        'latest_accuracy': model_performance.first().accuracy if model_performance.exists() else 0,
    }
    
    return render(request, 'news_analyzer/training_dashboard.html', {
        'training_data': training_data,
        'model_performance': model_performance,
        'stats': stats
    })

@login_required
def add_training_data(request):
    """Add new training data"""
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        is_fake = request.POST.get('is_fake') == 'true'
        source = request.POST.get('source', '')
        
        if title and content:
            TrainingData.objects.create(
                title=title,
                content=content,
                is_fake=is_fake,
                source=source,
                added_by=request.user
            )
            messages.success(request, 'Training data added successfully!')
            return redirect('training_dashboard')
        else:
            messages.error(request, 'Title and content are required!')
    
    return render(request, 'news_analyzer/add_training_data.html')

@login_required
def train_model(request):
    """Train the ML model with current training data"""
    if request.method == 'POST':
        try:
            # Get training data
            training_data = TrainingData.objects.all()
            
            if training_data.count() < 10:
                messages.error(request, 'Need at least 10 training samples to train the model.')
                return redirect('training_dashboard')
            
            # Prepare data
            texts = []
            labels = []
            
            for item in training_data:
                texts.append(item.title + " " + item.content)
                labels.append(1 if item.is_fake else 0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_vectors = vectorizer.fit_transform(X_train)
            X_test_vectors = vectorizer.transform(X_test)
            
            # Train model
            model = LogisticRegression(random_state=42)
            model.fit(X_train_vectors, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_vectors)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Save model performance
            ModelPerformance.objects.create(
                model_name=f"LogisticRegression_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_samples=len(X_train),
                test_samples=len(X_test)
            )
            
            # Save model files
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            
            model_filename = f"fake_news_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            vectorizer_filename = f"vectorizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            model_path = os.path.join(model_dir, model_filename)
            vectorizer_path = os.path.join(model_dir, vectorizer_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            # Save to database
            ModelFile.objects.create(
                name=f"Fake News Detector {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                file_path=model_path,
                model_type='tfidf_logistic',
                is_active=True
            )
            
            messages.success(request, f'Model trained successfully! Accuracy: {accuracy:.3f}')
            
        except Exception as e:
            messages.error(request, f'Error training model: {str(e)}')
    
    return redirect('training_dashboard')

def article_detail(request, article_id):
    """View detailed analysis of a specific article"""
    article = get_object_or_404(NewsArticle, id=article_id)
    return render(request, 'news_analyzer/article_detail.html', {'article': article})

def results_dashboard(request):
    """Dashboard showing analysis results"""
    articles = NewsArticle.objects.all()
    
    # Filter options
    filter_type = request.GET.get('filter', 'all')
    if filter_type == 'fake':
        articles = articles.filter(is_fake=True)
    elif filter_type == 'real':
        articles = articles.filter(is_fake=False)
    elif filter_type == 'unknown':
        articles = articles.filter(is_fake__isnull=True)
    
    return render(request, 'news_analyzer/results_dashboard.html', {
        'articles': articles,
        'filter_type': filter_type
    })

@csrf_exempt
def submit_feedback(request):
    """Submit feedback on analysis results"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            article_id = data.get('article_id')
            user_correction = data.get('user_correction')  # True for fake, False for real
            feedback_comment = data.get('feedback_comment', '')
            
            article = get_object_or_404(NewsArticle, id=article_id)
            
            # Create feedback record
            feedback = AnalysisFeedback.objects.create(
                article=article,
                user_correction=user_correction,
                feedback_comment=feedback_comment
            )
            
            # Optionally update the article with the correction
            if data.get('update_article', False):
                article.is_fake = user_correction
                article.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Feedback submitted successfully!'
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)

def feedback_dashboard(request):
    """Dashboard showing user feedback"""
    feedbacks = AnalysisFeedback.objects.all()
    
    stats = {
        'total_feedback': feedbacks.count(),
        'corrections_to_fake': feedbacks.filter(user_correction=True).count(),
        'corrections_to_real': feedbacks.filter(user_correction=False).count(),
    }
    
    return render(request, 'news_analyzer/feedback_dashboard.html', {
        'feedbacks': feedbacks,
        'stats': stats
    })


