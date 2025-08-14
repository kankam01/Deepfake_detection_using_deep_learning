"""
Enhanced Analysis Service for Fake News Detection
Integrates logical reasoning with existing detection methods
"""

import pickle
import os
import re
import numpy as np
from datetime import datetime
from collections import Counter

from .models import ModelFile, TrainingData

class EnhancedAnalysisService:
    """Enhanced analysis service with logical reasoning capabilities"""
    
    def __init__(self):
        self.logical_fallacies = {
            'appeal_to_authority': [
                'expert says', 'scientists claim', 'doctors reveal', 'authority confirms',
                'official statement', 'government admits', 'researchers found'
            ],
            'false_causation': [
                'causes', 'leads to', 'results in', 'creates', 'generates',
                'responsible for', 'blamed for', 'linked to'
            ],
            'conspiracy_theory': [
                'cover-up', 'hidden truth', 'suppressed', 'censored', 'secret',
                'they don\'t want you to know', 'mainstream media lies',
                'government hiding', 'elite controlling'
            ],
            'miracle_claims': [
                'miracle cure', 'instant results', 'overnight success',
                'one weird trick', 'secret method', 'revolutionary breakthrough',
                'cures all', 'works for everyone', 'guaranteed results'
            ],
            'urgency_manipulation': [
                'act now', 'limited time', 'last chance', 'don\'t miss out',
                'urgent', 'breaking', 'exclusive', 'shocking revelation'
            ]
        }
        
        self.factual_indicators = {
            'specific_numbers': r'\d+(?:\.\d+)?%?',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'locations': r'\b(?:in|at|near|around)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'names': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'organizations': r'\b(?:University|Institute|Center|Foundation|Corporation|Company|Inc|LLC)\b'
        }
        
        self.credible_patterns = [
            'peer-reviewed', 'published in', 'journal article', 'academic study',
            'university research', 'scientific evidence', 'official report',
            'government data', 'verified sources', 'fact-checked',
            'according to data', 'statistics show', 'research indicates'
        ]

    def analyze_with_enhanced_model(self, text):
        """Analyze text using enhanced model with logical reasoning"""
        try:
            # Try to load enhanced model first
            enhanced_result = self._analyze_with_enhanced_model(text)
            if enhanced_result:
                return enhanced_result
            
            # Fallback to existing model
            return self._analyze_with_existing_model(text)
            
        except Exception as e:
            print(f"Error in enhanced analysis: {e}")
            return self._analyze_with_rules(text)

    def _analyze_with_enhanced_model(self, text):
        """Analyze using enhanced model if available"""
        try:
            # Try to use the improved model first
            improved_result = self._analyze_with_improved_model(text)
            if improved_result:
                return improved_result
            
            # Look for logical reasoning model
            logical_model_file = ModelFile.objects.filter(
                model_type='logical_random_forest',
                is_active=True
            ).order_by('-created_at').first()
            
            if logical_model_file and os.path.exists(logical_model_file.file_path):
                return self._analyze_with_logical_model(text, logical_model_file)
            
            # Fallback to enhanced model
            enhanced_model_file = ModelFile.objects.filter(
                model_type='enhanced_random_forest',
                is_active=True
            ).order_by('-created_at').first()
            
            if not enhanced_model_file or not os.path.exists(enhanced_model_file.file_path):
                return None
            
            # Load enhanced model components
            model_dir = os.path.dirname(enhanced_model_file.file_path)
            timestamp = enhanced_model_file.file_path.split('_')[-1].replace('.pkl', '')
            
            vectorizer_path = os.path.join(model_dir, f"enhanced_vectorizer_{timestamp}.pkl")
            features_path = os.path.join(model_dir, f"enhanced_features_{timestamp}.pkl")
            
            if not (os.path.exists(vectorizer_path) and os.path.exists(features_path)):
                return None
            
            # Load components
            with open(enhanced_model_file.file_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
            
            # Extract enhanced features
            features = self._extract_enhanced_features(text)
            feature_vector = [features.get(feature, 0) for feature in feature_names]
            
            # TF-IDF features
            text_vector = vectorizer.transform([text]).toarray()
            
            # Combine features
            combined_features = np.hstack([text_vector, np.array([feature_vector])])
            
            # Make prediction
            prediction = model.predict(combined_features)[0]
            confidence = model.predict_proba(combined_features)[0].max()
            
            is_fake = bool(prediction)
            
            # Generate enhanced explanation
            explanation = self._generate_enhanced_explanation(text, features, confidence)
            
            return {
                'is_fake': is_fake,
                'confidence': confidence,
                'explanation': explanation,
                'enhanced_features': features,
                'model_type': 'enhanced_random_forest'
            }
            
        except Exception as e:
            print(f"Error loading enhanced model: {e}")
            return None

    def _analyze_with_improved_model(self, text):
        """Analyze using the improved model trained on the local dataset"""
        try:
            import joblib
            
            # Check if improved model files exist
            model_path = 'models/improved_fake_news_classifier.pkl'
            vectorizer_path = 'models/improved_tfidf_vectorizer.pkl'
            
            if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
                return None
            
            # Load the improved model and vectorizer
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            # Better text preprocessing
            processed_text = self._preprocess_text_for_model(text)
            
            # Vectorize the text
            text_vector = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(text_vector)[0]
            confidence_scores = model.predict_proba(text_vector)[0]
            
            # Get confidence for the predicted class
            if prediction == 1:  # Fake news
                confidence = confidence_scores[1]
                is_fake = True
            else:  # Real news
                confidence = confidence_scores[0]
                is_fake = False
            
            # If model confidence is low, use logical reasoning instead
            if confidence < 0.6:
                logical_result = self._analyze_with_logical_reasoning_only(text)
                if logical_result:
                    return logical_result
            
            # Apply confidence threshold: below 45% = likely fake
            if confidence < 0.45:
                is_fake = True
                confidence = 1.0 - confidence  # Invert confidence for fake news
            
            # Generate explanation
            explanation = self._generate_improved_explanation(text, confidence, is_fake)
            
            return {
                'is_fake': is_fake,
                'confidence': confidence,
                'explanation': explanation,
                'model_type': 'improved_random_forest',
                'confidence_scores': {
                    'fake': confidence_scores[1] if len(confidence_scores) > 1 else 0,
                    'real': confidence_scores[0] if len(confidence_scores) > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"Error loading improved model: {e}")
            return None

    def _generate_improved_explanation(self, text, confidence, is_fake):
        """Generate explanation for improved model predictions"""
        try:
            # Extract key features from the text
            text_lower = text.lower()
            
            # Check for fake news indicators
            fake_indicators = []
            for fallacy_type, patterns in self.logical_fallacies.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        fake_indicators.append(f"{fallacy_type}: {pattern}")
            
            # Check for credible indicators
            credible_indicators = []
            for pattern in self.credible_patterns:
                if pattern in text_lower:
                    credible_indicators.append(pattern)
            
            # Generate explanation
            if is_fake:
                if fake_indicators:
                    explanation = f"This article appears to be fake news (confidence: {confidence:.1%}). "
                    explanation += f"Detected indicators: {', '.join(fake_indicators[:3])}"
                else:
                    explanation = f"This article is classified as fake news (confidence: {confidence:.1%}) based on the trained model's analysis of the text patterns."
            else:
                if credible_indicators:
                    explanation = f"This article appears to be real news (confidence: {confidence:.1%}). "
                    explanation += f"Detected credible indicators: {', '.join(credible_indicators[:3])}"
                else:
                    explanation = f"This article is classified as real news (confidence: {confidence:.1%}) based on the trained model's analysis."
            
            return explanation
            
        except Exception as e:
            return f"Analysis completed with {confidence:.1%} confidence."

    def _analyze_with_logical_reasoning_only(self, text):
        """Analyze using only logical reasoning when model confidence is low"""
        try:
            text_lower = text.lower()
            
            # Count fake news indicators
            fake_indicators = []
            fake_count = 0
            
            # Check for obvious fake news patterns
            obvious_fake_patterns = [
                'shocking', 'amazing', 'incredible', 'unbelievable', 'stunning',
                'miracle cure', 'instant results', 'overnight success',
                'one weird trick', 'secret method', 'revolutionary breakthrough',
                'cures all', 'works for everyone', 'guaranteed results',
                'doctors hate', 'government hiding', 'cover up', 'suppressed',
                'censored', 'mainstream media lies', 'conspiracy',
                'elite controlling', 'deep state', 'illuminati', 'new world order',
                'act now', 'limited time', 'last chance', 'don\'t miss out',
                'urgent', 'breaking', 'exclusive', 'shocking revelation',
                'they don\'t want you to know', 'what they don\'t want you to know',
                # Science fiction and fantasy indicators
                'time travel', 'traveled back', 'traveled forward', 'time machine',
                'parallel universe', 'alternate reality', 'dimension', 'portal',
                'teleport', 'teleported', 'supernatural', 'magic', 'wizard',
                'vampire', 'werewolf', 'alien', 'ufo', 'extraterrestrial',
                'flying saucer', 'space ship', 'time warp', 'wormhole',
                'accidentally traveled', 'accidentally went back', 'accidentally went forward',
                'temporal displacement', 'time displacement', 'time anomaly',
                # Impossible claims
                'impossible', 'defies physics', 'defies science', 'defies logic',
                'breaks the laws of', 'violates the laws of', 'scientifically impossible',
                'physically impossible', 'mathematically impossible',
                # Obvious fiction indicators
                'fairy tale', 'storybook', 'fantasy', 'science fiction', 'sci-fi',
                'fictional', 'made up', 'fabricated', 'hoax', 'prank', 'joke',
                'satire', 'parody', 'comedy', 'humor', 'funny', 'hilarious',
                # Ridiculous claims
                'talking animals', 'flying pigs', 'invisible man', 'invisible woman',
                'mind reading', 'mind control', 'psychic', 'psychic powers',
                'superhuman', 'super powers', 'superhero', 'superheroes',
                'mutant', 'mutants', 'x-men', 'spider-man', 'batman',
                # Absurd scenarios
                'zombie apocalypse', 'zombies', 'vampires', 'werewolves',
                'ghosts', 'haunted', 'haunting', 'poltergeist', 'exorcism',
                'demons', 'angels', 'heaven', 'hell', 'afterlife', 'reincarnation',
                'past life', 'future life', 'prophecy', 'prophet', 'fortune teller'
            ]
            
            for pattern in obvious_fake_patterns:
                if pattern in text_lower:
                    fake_count += 1
                    fake_indicators.append(pattern)
            
            # Count credible indicators
            credible_indicators = []
            credible_count = 0
            
            credible_patterns = [
                'study shows', 'research indicates', 'according to experts',
                'peer-reviewed', 'published in', 'journal article', 'academic study',
                'university research', 'scientific evidence', 'official report',
                'government data', 'verified sources', 'fact-checked',
                'according to data', 'statistics show', 'research indicates'
            ]
            
            for pattern in credible_patterns:
                if pattern in text_lower:
                    credible_count += 1
                    credible_indicators.append(pattern)
            
            # Check for impossible scenarios (time travel, supernatural, etc.)
            impossible_scenarios = [
                'time travel', 'traveled back', 'traveled forward', 'time machine',
                'parallel universe', 'alternate reality', 'dimension', 'portal',
                'teleport', 'teleported', 'supernatural', 'magic', 'wizard',
                'vampire', 'werewolf', 'alien', 'ufo', 'extraterrestrial',
                'flying saucer', 'space ship', 'time warp', 'wormhole',
                'accidentally traveled', 'accidentally went back', 'accidentally went forward',
                'temporal displacement', 'time displacement', 'time anomaly'
            ]
            
            impossible_count = sum(1 for scenario in impossible_scenarios if scenario in text_lower)
            
            # Determine if fake based on logical reasoning
            if impossible_count >= 1:
                # If any impossible scenario is mentioned, it's definitely fake
                is_fake = True
                confidence = 0.95
                explanation = f"This article appears to be fake news (confidence: {confidence:.1%}). "
                explanation += f"Contains impossible scenarios: {', '.join([s for s in impossible_scenarios if s in text_lower][:3])}"
            elif fake_count > credible_count and fake_count >= 2:
                is_fake = True
                confidence = min(0.95, 0.5 + (fake_count * 0.1))
                explanation = f"This article appears to be fake news (confidence: {confidence:.1%}). "
                explanation += f"Detected {fake_count} fake news indicators: {', '.join(fake_indicators[:3])}"
            elif credible_count > fake_count and credible_count >= 2:
                is_fake = False
                confidence = min(0.95, 0.5 + (credible_count * 0.1))
                explanation = f"This article appears to be real news (confidence: {confidence:.1%}). "
                explanation += f"Detected {credible_count} credible indicators: {', '.join(credible_indicators[:3])}"
            else:
                # If no clear indicators, check for sensationalism
                sensational_words = ['shocking', 'amazing', 'incredible', 'miracle', 'secret', 'exclusive']
                sensational_count = sum(1 for word in sensational_words if word in text_lower)
                
                if sensational_count >= 2:
                    is_fake = True
                    confidence = 0.7
                    explanation = f"This article appears to be fake news (confidence: {confidence:.1%}). "
                    explanation += f"Detected {sensational_count} sensational indicators suggesting clickbait or fake news."
                else:
                    is_fake = False
                    confidence = 0.6
                    explanation = f"This article appears to be real news (confidence: {confidence:.1%}). "
                    explanation += f"No obvious fake news indicators detected."
            
            return {
                'is_fake': is_fake,
                'confidence': confidence,
                'explanation': explanation,
                'model_type': 'logical_reasoning',
                'fake_indicators': fake_indicators,
                'credible_indicators': credible_indicators
            }
            
        except Exception as e:
            print(f"Error in logical reasoning analysis: {e}")
            return None

    def _preprocess_text_for_model(self, text):
        """Preprocess text for model input"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove very short words (likely noise)
            words = text.split()
            words = [word for word in words if len(word) > 2]
            
            return ' '.join(words).strip()
            
        except Exception as e:
            return text.lower().strip()

    def _analyze_with_logical_model(self, text, logical_model_file):
        """Analyze using logical reasoning model"""
        try:
            # Load logical model components
            model_dir = os.path.dirname(logical_model_file.file_path)
            # Extract timestamp from filename: logical_fake_news_model_20250807_183931.pkl
            filename = os.path.basename(logical_model_file.file_path)
            timestamp = filename.replace('logical_fake_news_model_', '').replace('.pkl', '')
            
            vectorizer_path = os.path.join(model_dir, f"logical_vectorizer_{timestamp}.pkl")
            features_path = os.path.join(model_dir, f"logical_features_{timestamp}.pkl")
            
            if not (os.path.exists(vectorizer_path) and os.path.exists(features_path)):
                return None
            
            # Load components
            with open(logical_model_file.file_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
            
            # Extract logical features using the same method as training
            features = self._extract_logical_features_for_model(text)
            feature_vector = [features.get(feature, 0) for feature in feature_names]
            
            # TF-IDF features
            text_vector = vectorizer.transform([text]).toarray()
            
            # Combine features
            combined_features = np.hstack([text_vector, np.array([feature_vector])])
            
            # Make prediction
            prediction = model.predict(combined_features)[0]
            confidence = model.predict_proba(combined_features)[0].max()
            
            is_fake = bool(prediction)
            
            # Generate logical reasoning explanation
            explanation = self._generate_logical_explanation(text, features, confidence)
            
            return {
                'is_fake': is_fake,
                'confidence': confidence,
                'explanation': explanation,
                'enhanced_features': features,
                'model_type': 'logical_random_forest'
            }
            
        except Exception as e:
            print(f"Error loading logical model: {e}")
            return None

    def _extract_logical_features_for_model(self, text):
        """Extract logical features using the same method as training"""
        text_lower = text.lower()
        features = {}
        
        # Logical fallacies (same as in training)
        logical_fallacies = {
            'appeal_to_authority': [
                'expert says', 'scientists claim', 'doctors reveal', 'authority confirms',
                'official statement', 'government admits', 'researchers found'
            ],
            'false_causation': [
                'causes', 'leads to', 'results in', 'creates', 'generates',
                'responsible for', 'blamed for', 'linked to'
            ],
            'conspiracy_theory': [
                'cover-up', 'hidden truth', 'suppressed', 'censored', 'secret',
                'they don\'t want you to know', 'mainstream media lies',
                'government hiding', 'elite controlling'
            ],
            'miracle_claims': [
                'miracle cure', 'instant results', 'overnight success',
                'one weird trick', 'secret method', 'revolutionary breakthrough',
                'cures all', 'works for everyone', 'guaranteed results'
            ],
            'urgency_manipulation': [
                'act now', 'limited time', 'last chance', 'don\'t miss out',
                'urgent', 'breaking', 'exclusive', 'shocking revelation'
            ]
        }
        
        for fallacy_type, patterns in logical_fallacies.items():
            fallacy_count = sum(1 for pattern in patterns if pattern in text_lower)
            features[f'fallacy_{fallacy_type}'] = fallacy_count
        
        # Factual indicators (same as in training)
        factual_indicators = {
            'specific_numbers': r'\d+(?:\.\d+)?%?',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'locations': r'\b(?:in|at|near|around)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'names': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'organizations': r'\b(?:University|Institute|Center|Foundation|Corporation|Company|Inc|LLC)\b'
        }
        
        for indicator_name, pattern in factual_indicators.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'factual_{indicator_name}'] = len(matches)
        
        # Credible patterns (same as in training)
        credible_patterns = [
            'peer-reviewed', 'published in', 'journal article', 'academic study',
            'university research', 'scientific evidence', 'official report',
            'government data', 'verified sources', 'fact-checked',
            'according to data', 'statistics show', 'research indicates'
        ]
        
        credible_count = sum(1 for pattern in credible_patterns if pattern in text_lower)
        features['credible_patterns'] = credible_count
        
        # Sensationalism (same as in training)
        sensational_words = [
            'shocking', 'amazing', 'incredible', 'unbelievable', 'stunning',
            'revolutionary', 'breakthrough', 'miracle', 'secret', 'exclusive'
        ]
        sensational_count = sum(1 for word in sensational_words if word in text_lower)
        features['sensationalism'] = sensational_count
        
        return features

    def _analyze_with_existing_model(self, text):
        """Analyze using existing model"""
        try:
            # Look for existing model (but not logical model)
            model_file = ModelFile.objects.filter(
                is_active=True,
                model_type__in=['tfidf_logistic', 'enhanced_random_forest']
            ).order_by('-created_at').first()
            
            if not model_file or not os.path.exists(model_file.file_path):
                return None
            
            # Load model and vectorizer
            with open(model_file.file_path, 'rb') as f:
                model = pickle.load(f)
            
            # Find vectorizer
            model_dir = os.path.dirname(model_file.file_path)
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            
            vectorizer_file = None
            for file in model_files:
                if file.startswith('vectorizer_'):
                    vectorizer_file = os.path.join(model_dir, file)
                    break
            
            if not vectorizer_file or not os.path.exists(vectorizer_file):
                return None
            
            with open(vectorizer_file, 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Make prediction
            text_vector = vectorizer.transform([text])
            prediction = model.predict(text_vector)[0]
            confidence = model.predict_proba(text_vector)[0].max()
            
            is_fake = bool(prediction)
            
            return {
                'is_fake': is_fake,
                'confidence': confidence,
                'explanation': f"ML model prediction with {confidence:.2%} confidence.",
                'model_type': 'existing_model'
            }
            
        except Exception as e:
            print(f"Error loading existing model: {e}")
            return None

    def _analyze_with_rules(self, text):
        """Fallback rule-based analysis with enhanced logic"""
        text_lower = text.lower()
        
        # Enhanced logical reasoning analysis
        logical_score = self._analyze_logical_reasoning(text_lower)
        factual_score = self._analyze_factual_accuracy(text)
        sensationalism_score = self._analyze_sensationalism(text_lower)
        
        # Calculate overall score
        total_score = logical_score + factual_score + sensationalism_score
        
        # Determine if fake based on logical reasoning
        if logical_score < -2 or sensationalism_score > 3:
            is_fake = True
            confidence = min(0.9, abs(total_score) / 10)
        elif factual_score > 2 and logical_score > 0:
            is_fake = False
            confidence = min(0.9, factual_score / 5)
        else:
            is_fake = None
            confidence = 0.5
        
        # Apply 45% confidence threshold: below 45% = likely fake
        if confidence < 0.45:
            is_fake = True
            confidence = 1.0 - confidence  # Invert confidence for fake news
        
        explanation = self._generate_rule_based_explanation(logical_score, factual_score, sensationalism_score)
        
        return {
            'is_fake': is_fake,
            'confidence': confidence,
            'explanation': explanation,
            'model_type': 'enhanced_rules'
        }

    def _extract_enhanced_features(self, text):
        """Extract enhanced features for analysis"""
        text_lower = text.lower()
        features = {}
        
        # Logical fallacies
        for fallacy_type, patterns in self.logical_fallacies.items():
            fallacy_count = sum(1 for pattern in patterns if pattern in text_lower)
            features[f'fallacy_{fallacy_type}'] = fallacy_count
        
        # Factual indicators
        for indicator_name, pattern in self.factual_indicators.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'factual_{indicator_name}'] = len(matches)
        
        # Credible patterns
        credible_count = sum(1 for pattern in self.credible_patterns if pattern in text_lower)
        features['credible_patterns'] = credible_count
        
        # Sensationalism
        sensational_words = [
            'shocking', 'amazing', 'incredible', 'unbelievable', 'stunning',
            'revolutionary', 'breakthrough', 'miracle', 'secret', 'exclusive'
        ]
        sensational_count = sum(1 for word in sensational_words if word in text_lower)
        features['sensationalism'] = sensational_count
        
        # Emotional manipulation
        emotional_words = [
            'fear', 'anger', 'hate', 'love', 'hope', 'despair', 'joy', 'sadness',
            'outrage', 'shock', 'disgust', 'amazement', 'wonder', 'terror'
        ]
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        features['emotional_manipulation'] = emotional_count
        
        return features

    def _analyze_logical_reasoning(self, text):
        """Analyze logical reasoning patterns"""
        score = 0
        
        # Check for logical fallacies (negative score)
        for fallacy_type, patterns in self.logical_fallacies.items():
            fallacy_count = sum(1 for pattern in patterns if pattern in text)
            score -= fallacy_count * 2  # Heavy penalty for logical fallacies
        
        # Check for balanced reporting (positive score)
        balanced_indicators = ['however', 'but', 'although', 'while', 'despite']
        balanced_count = sum(1 for indicator in balanced_indicators if indicator in text)
        score += balanced_count
        
        # Check for hedging language (indicates uncertainty)
        hedging_words = ['might', 'could', 'possibly', 'perhaps', 'maybe', 'seems', 'appears']
        hedging_count = sum(1 for word in hedging_words if word in text)
        score -= hedging_count * 0.5
        
        return score

    def _analyze_factual_accuracy(self, text):
        """Analyze factual accuracy indicators"""
        score = 0
        
        # Check for specific numbers and dates
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        score += len(numbers) * 0.5
        
        # Check for dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text)
        score += len(dates) * 0.5
        
        # Check for specific locations
        locations = re.findall(r'\b(?:in|at|near|around)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        score += len(locations) * 0.3
        
        # Check for specific names
        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
        score += len(names) * 0.3
        
        # Check for organizations
        organizations = re.findall(r'\b(?:University|Institute|Center|Foundation|Corporation|Company|Inc|LLC)\b', text)
        score += len(organizations) * 0.5
        
        # Check for credible patterns
        credible_count = sum(1 for pattern in self.credible_patterns if pattern in text.lower())
        score += credible_count * 1.0
        
        return score

    def _analyze_sensationalism(self, text):
        """Analyze sensationalism indicators"""
        sensational_words = [
            'shocking', 'amazing', 'incredible', 'unbelievable', 'stunning',
            'revolutionary', 'breakthrough', 'miracle', 'secret', 'exclusive',
            'urgent', 'breaking', 'last chance', 'act now', 'don\'t miss out'
        ]
        
        sensational_count = sum(1 for word in sensational_words if word in text)
        return sensational_count

    def _generate_enhanced_explanation(self, text, features, confidence):
        """Generate enhanced explanation based on features"""
        explanation_parts = []
        
        # Analyze logical fallacies
        fallacy_counts = {k: v for k, v in features.items() if k.startswith('fallacy_')}
        total_fallacies = sum(fallacy_counts.values())
        
        if total_fallacies > 0:
            fallacy_types = [k.replace('fallacy_', '').replace('_', ' ') for k, v in fallacy_counts.items() if v > 0]
            explanation_parts.append(f"Detected {total_fallacies} logical fallacy indicators: {', '.join(fallacy_types)}")
        
        # Analyze factual accuracy
        factual_counts = {k: v for k, v in features.items() if k.startswith('factual_')}
        total_factual = sum(factual_counts.values())
        
        if total_factual > 0:
            explanation_parts.append(f"Found {total_factual} factual accuracy indicators (numbers, dates, names, organizations)")
        
        # Analyze sensationalism
        sensationalism = features.get('sensationalism', 0)
        if sensationalism > 0:
            explanation_parts.append(f"Detected {sensationalism} sensationalism indicators")
        
        # Analyze credible patterns
        credible_patterns = features.get('credible_patterns', 0)
        if credible_patterns > 0:
            explanation_parts.append(f"Found {credible_patterns} credible source indicators")
        
        # Emotional manipulation
        emotional_manipulation = features.get('emotional_manipulation', 0)
        if emotional_manipulation > 0:
            explanation_parts.append(f"Detected {emotional_manipulation} emotional manipulation indicators")
        
        if explanation_parts:
            explanation = ". ".join(explanation_parts) + f" Overall confidence: {confidence:.1%}"
        else:
            explanation = f"Enhanced analysis completed with {confidence:.1%} confidence."
        
        return explanation

    def _generate_rule_based_explanation(self, logical_score, factual_score, sensationalism_score):
        """Generate explanation for rule-based analysis"""
        explanation_parts = []
        
        if logical_score < 0:
            explanation_parts.append(f"Logical reasoning analysis: {abs(logical_score)} potential logical fallacies detected")
        
        if factual_score > 0:
            explanation_parts.append(f"Factual accuracy: {factual_score} factual indicators found")
        
        if sensationalism_score > 0:
            explanation_parts.append(f"Sensationalism: {sensationalism_score} sensational indicators detected")
        
        if not explanation_parts:
            explanation_parts.append("Limited indicators found for analysis")
        
        return ". ".join(explanation_parts)

    def get_analysis_summary(self, text):
        """Get comprehensive analysis summary"""
        result = self.analyze_with_enhanced_model(text)
        
        if not result:
            return {
                'error': 'Unable to perform analysis',
                'is_fake': None,
                'confidence': 0.0,
                'explanation': 'Analysis failed'
            }
        
        # Add additional analysis details
        text_lower = text.lower()
        
        # Count specific indicators
        miracle_claims = sum(1 for word in ['miracle', 'cure', 'instant', 'overnight', 'secret'] if word in text_lower)
        conspiracy_claims = sum(1 for word in ['conspiracy', 'cover-up', 'hidden', 'suppressed', 'secret'] if word in text_lower)
        authority_appeals = sum(1 for word in ['expert', 'scientist', 'doctor', 'official', 'authority'] if word in text_lower)
        
        analysis_details = {
            'miracle_claims': miracle_claims,
            'conspiracy_claims': conspiracy_claims,
            'authority_appeals': authority_appeals,
            'model_type': result.get('model_type', 'unknown'),
            'enhanced_features': result.get('enhanced_features', {})
        }
        
        result['analysis_details'] = analysis_details
        return result

    def _generate_logical_explanation(self, text, features, confidence):
        """Generate logical reasoning explanation"""
        explanation_parts = []
        
        # Analyze logical fallacies
        fallacy_counts = {k: v for k, v in features.items() if k.startswith('fallacy_')}
        total_fallacies = sum(fallacy_counts.values())
        
        if total_fallacies > 0:
            fallacy_types = [k.replace('fallacy_', '').replace('_', ' ') for k, v in fallacy_counts.items() if v > 0]
            explanation_parts.append(f"Detected {total_fallacies} logical fallacy indicators: {', '.join(fallacy_types)}")
        
        # Analyze factual accuracy
        factual_counts = {k: v for k, v in features.items() if k.startswith('factual_')}
        total_factual = sum(factual_counts.values())
        
        if total_factual > 0:
            explanation_parts.append(f"Found {total_factual} factual accuracy indicators (numbers, dates, names, organizations)")
        
        # Analyze sensationalism
        sensationalism = features.get('sensationalism', 0)
        if sensationalism > 0:
            explanation_parts.append(f"Detected {sensationalism} sensationalism indicators")
        
        # Analyze credible patterns
        credible_patterns = features.get('credible_patterns', 0)
        if credible_patterns > 0:
            explanation_parts.append(f"Found {credible_patterns} credible source indicators")
        
        # Emotional manipulation
        emotional_manipulation = features.get('emotional_manipulation', 0)
        if emotional_manipulation > 0:
            explanation_parts.append(f"Detected {emotional_manipulation} emotional manipulation indicators")
        
        if explanation_parts:
            explanation = ". ".join(explanation_parts) + f" Logical reasoning analysis with {confidence:.1%} confidence."
        else:
            explanation = f"Logical reasoning analysis completed with {confidence:.1%} confidence."
        
        return explanation
