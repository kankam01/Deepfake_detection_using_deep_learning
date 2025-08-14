import requests
from newspaper import Article
from bs4 import BeautifulSoup
import json
from urllib.parse import urlparse
import re
from django.conf import settings
import trafilatura
from readability import Document

class NewsAPIService:
    """Service for handling News API integration and web scraping"""
    
    def __init__(self):
        self.api_key = "aa7ce526c3c34252ad8a8af006eba7f6"
        self.base_url = "https://newsapi.org/v2"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def extract_article_from_url(self, url):
        """Extract article content from a given URL"""
        try:
            # Validate URL
            if not url or not url.startswith(('http://', 'https://')):
                return None
            
            print(f"Attempting to extract content from: {url}")
            
            # Skip News API for now as it returns unrelated articles
            # First try to get article info from News API
            # article_info = self._get_article_from_newsapi(url)
            # if article_info and article_info.get('content'):
            #     print(f"Successfully extracted via News API")
            #     return article_info

            # Try extracting with newspaper3k
            try:
                print(f"Trying newspaper3k extraction...")
                news_article = Article(url)
                news_article.download()
                news_article.parse()
                title = news_article.title
                content = news_article.text
                
                # Clean up content
                if content:
                    content = re.sub(r'\s+', ' ', content).strip()
                
                if content and len(content) > 100:
                    parsed_url = urlparse(url)
                    source = parsed_url.netloc.replace('www.', '')
                    print(f"Successfully extracted via newspaper3k: {len(content)} characters")
                    return {
                        'title': title,
                        'content': content,
                        'source': source,
                        'url': url,
                        'published_at': news_article.publish_date.strftime('%Y-%m-%d') if news_article.publish_date else '',
                        'author': ', '.join(news_article.authors) if news_article.authors else '',
                        'extraction_method': 'newspaper3k'
                    }
                else:
                    print(f"Newspaper3k extracted content too short: {len(content) if content else 0} characters")
            except Exception as e:
                print(f"Error extracting with newspaper3k: {e}")

            # Try website-specific extraction first
            website_specific = self._extract_from_specific_websites(url)
            if website_specific:
                print(f"Successfully extracted using website-specific method")
                return website_specific
            
            # Try advanced content extraction libraries
            advanced_extraction = self._extract_with_advanced_libraries(url)
            if advanced_extraction:
                print(f"Successfully extracted using advanced libraries")
                return advanced_extraction
            
            # Fallback to enhanced web scraping
            print(f"Trying enhanced web scraping...")
            return self._scrape_article_content(url)
            
        except Exception as e:
            print(f"Error extracting article from URL: {e}")
            return None
    
    def _get_article_from_newsapi(self, url):
        """Try to get article information from News API"""
        try:
            # Extract domain from URL for better search
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace('www.', '')
            
            # Search for the article in News API using domain
            search_url = f"{self.base_url}/everything"
            params = {
                'q': domain,
                'apiKey': self.api_key,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 5
            }
            
            response = requests.get(search_url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('articles') and len(data['articles']) > 0:
                    # Find the most relevant article
                    for article in data['articles']:
                        article_url = article.get('url', '')
                        if url in article_url or article_url in url:
                            return {
                                'title': article.get('title', ''),
                                'content': (article.get('description', '') + ' ' + (article.get('content', '') or '')).strip(),
                                'source': article.get('source', {}).get('name', ''),
                                'url': article.get('url', url),
                                'published_at': article.get('publishedAt', ''),
                                'author': article.get('author', '')
                            }
                    
                    # If no exact match, return the first article
                    article = data['articles'][0]
                    return {
                        'title': article.get('title', ''),
                        'content': (article.get('description', '') + ' ' + (article.get('content', '') or '')).strip(),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', url),
                        'published_at': article.get('publishedAt', ''),
                        'author': article.get('author', '')
                    }
        except Exception as e:
            print(f"Error getting article from News API: {e}")
        
        return None
    
    def _extract_with_advanced_libraries(self, url):
        """Extract content using advanced libraries like trafilatura and readability"""
        try:
            print(f"Trying advanced extraction with trafilatura...")
            
            # Try trafilatura first (most reliable)
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    result = trafilatura.extract(downloaded, include_formatting=True, include_links=True, include_images=True)
                    if result and len(result) > 200:
                        # Get metadata
                        metadata = trafilatura.extract_metadata(downloaded)
                        title = metadata.title if metadata and metadata.title else ''
                        
                        # Clean up the content
                        content = result.strip()
                        content = re.sub(r'\s+', ' ', content)
                        
                        parsed_url = urlparse(url)
                        source = parsed_url.netloc.replace('www.', '')
                        
                        print(f"Successfully extracted with trafilatura: {len(content)} characters")
                        return {
                            'title': title,
                            'content': content,
                            'source': source,
                            'url': url,
                            'published_at': '',
                            'author': metadata.author if metadata else '',
                            'extraction_method': 'trafilatura'
                        }
            except Exception as e:
                print(f"Trafilatura extraction failed: {e}")
            
            # Try readability-lxml as fallback
            try:
                print(f"Trying readability-lxml extraction...")
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                doc = Document(response.text)
                title = doc.title()
                content = doc.summary()
                
                if content and len(content) > 200:
                    # Clean up the content
                    content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    parsed_url = urlparse(url)
                    source = parsed_url.netloc.replace('www.', '')
                    
                    print(f"Successfully extracted with readability: {len(content)} characters")
                    return {
                        'title': title,
                        'content': content,
                        'source': source,
                        'url': url,
                        'published_at': '',
                        'author': '',
                        'extraction_method': 'readability'
                    }
            except Exception as e:
                print(f"Readability extraction failed: {e}")
            
            return None
            
        except Exception as e:
            print(f"Error in advanced extraction: {e}")
            return None
    
    def _extract_from_specific_websites(self, url):
        """Extract content from specific well-known websites"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Handle specific websites with known structures
            if 'medium.com' in domain:
                return self._extract_from_medium(url)
            elif 'substack.com' in domain:
                return self._extract_from_substack(url)
            elif 'wordpress.com' in domain:
                return self._extract_from_wordpress(url)
            elif 'blogspot.com' in domain or 'blogger.com' in domain:
                return self._extract_from_blogger(url)
            elif 'tumblr.com' in domain:
                return self._extract_from_tumblr(url)
            
            return None
            
        except Exception as e:
            print(f"Error in website-specific extraction: {e}")
            return None
    
    def _extract_from_medium(self, url):
        """Extract content from Medium articles"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Medium specific selectors
            title = ''
            title_elem = soup.select_one('h1') or soup.select_one('[data-testid="storyTitle"]')
            if title_elem:
                title = title_elem.get_text().strip()
            
            content = ''
            # Medium article content
            article_body = soup.select_one('article') or soup.select_one('[data-testid="storyBody"]')
            if article_body:
                paragraphs = article_body.find_all('p')
                content_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                content = ' '.join(content_parts)
            
            if content and len(content) > 200:
                return {
                    'title': title,
                    'content': content,
                    'source': 'medium.com',
                    'url': url,
                    'published_at': '',
                    'author': '',
                    'extraction_method': 'medium_specific'
                }
        except Exception as e:
            print(f"Error extracting from Medium: {e}")
        return None
    
    def _extract_from_substack(self, url):
        """Extract content from Substack articles"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = ''
            title_elem = soup.select_one('h1') or soup.select_one('.post-title')
            if title_elem:
                title = title_elem.get_text().strip()
            
            content = ''
            # Substack article content
            article_body = soup.select_one('.body markup') or soup.select_one('.post-content')
            if article_body:
                paragraphs = article_body.find_all('p')
                content_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                content = ' '.join(content_parts)
            
            if content and len(content) > 200:
                return {
                    'title': title,
                    'content': content,
                    'source': 'substack.com',
                    'url': url,
                    'published_at': '',
                    'author': '',
                    'extraction_method': 'substack_specific'
                }
        except Exception as e:
            print(f"Error extracting from Substack: {e}")
        return None
    
    def _extract_from_wordpress(self, url):
        """Extract content from WordPress blogs"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = ''
            title_elem = soup.select_one('.entry-title') or soup.select_one('h1')
            if title_elem:
                title = title_elem.get_text().strip()
            
            content = ''
            # WordPress article content
            article_body = soup.select_one('.entry-content') or soup.select_one('.post-content')
            if article_body:
                paragraphs = article_body.find_all('p')
                content_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                content = ' '.join(content_parts)
            
            if content and len(content) > 200:
                return {
                    'title': title,
                    'content': content,
                    'source': 'wordpress.com',
                    'url': url,
                    'published_at': '',
                    'author': '',
                    'extraction_method': 'wordpress_specific'
                }
        except Exception as e:
            print(f"Error extracting from WordPress: {e}")
        return None
    
    def _extract_from_blogger(self, url):
        """Extract content from Blogger/Blogspot blogs"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = ''
            title_elem = soup.select_one('.post-title') or soup.select_one('h1')
            if title_elem:
                title = title_elem.get_text().strip()
            
            content = ''
            # Blogger article content
            article_body = soup.select_one('.post-body') or soup.select_one('.entry-content')
            if article_body:
                paragraphs = article_body.find_all('p')
                content_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                content = ' '.join(content_parts)
            
            if content and len(content) > 200:
                return {
                    'title': title,
                    'content': content,
                    'source': 'blogspot.com',
                    'url': url,
                    'published_at': '',
                    'author': '',
                    'extraction_method': 'blogger_specific'
                }
        except Exception as e:
            print(f"Error extracting from Blogger: {e}")
        return None
    
    def _extract_from_tumblr(self, url):
        """Extract content from Tumblr posts"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = ''
            title_elem = soup.select_one('.post-title') or soup.select_one('h1')
            if title_elem:
                title = title_elem.get_text().strip()
            
            content = ''
            # Tumblr post content
            article_body = soup.select_one('.post-content') or soup.select_one('.entry-content')
            if article_body:
                paragraphs = article_body.find_all('p')
                content_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                content = ' '.join(content_parts)
            
            if content and len(content) > 200:
                return {
                    'title': title,
                    'content': content,
                    'source': 'tumblr.com',
                    'url': url,
                    'published_at': '',
                    'author': '',
                    'extraction_method': 'tumblr_specific'
                }
        except Exception as e:
            print(f"Error extracting from Tumblr: {e}")
        return None
    
    def test_url_extraction(self, url):
        """Test URL extraction and return detailed results"""
        try:
            print(f"\n=== Testing URL Extraction for: {url} ===")
            
            # Test each method
            methods = [
                ('News API', self._get_article_from_newsapi),
                ('Newspaper3k', lambda u: self._extract_with_newspaper3k(u)),
                ('Trafilatura', lambda u: self._extract_with_trafilatura(u)),
                ('Readability', lambda u: self._extract_with_readability(u)),
                ('Enhanced Scraping', self._scrape_article_content)
            ]
            
            results = {}
            for method_name, method_func in methods:
                try:
                    print(f"\n--- Testing {method_name} ---")
                    result = method_func(url)
                    if result:
                        content_length = len(result.get('content', ''))
                        print(f"✓ {method_name} SUCCESS: {content_length} characters")
                        results[method_name] = {
                            'success': True,
                            'content_length': content_length,
                            'title': result.get('title', ''),
                            'source': result.get('source', ''),
                            'method': result.get('extraction_method', method_name.lower())
                        }
                    else:
                        print(f"✗ {method_name} FAILED: No content extracted")
                        results[method_name] = {'success': False}
                except Exception as e:
                    print(f"✗ {method_name} ERROR: {e}")
                    results[method_name] = {'success': False, 'error': str(e)}
            
            return results
            
        except Exception as e:
            print(f"Error in test_url_extraction: {e}")
            return None
    
    def _extract_with_newspaper3k(self, url):
        """Extract content using newspaper3k"""
        try:
            news_article = Article(url)
            news_article.download()
            news_article.parse()
            
            title = news_article.title
            content = news_article.text
            
            if content:
                content = re.sub(r'\s+', ' ', content).strip()
            
            if content and len(content) > 100:
                parsed_url = urlparse(url)
                source = parsed_url.netloc.replace('www.', '')
                return {
                    'title': title,
                    'content': content,
                    'source': source,
                    'url': url,
                    'published_at': news_article.publish_date.strftime('%Y-%m-%d') if news_article.publish_date else '',
                    'author': ', '.join(news_article.authors) if news_article.authors else '',
                    'extraction_method': 'newspaper3k'
                }
        except Exception as e:
            print(f"Newspaper3k error: {e}")
        return None
    
    def _extract_with_trafilatura(self, url):
        """Extract content using trafilatura"""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                result = trafilatura.extract(downloaded, include_formatting=True, include_links=True, include_images=True)
                if result and len(result) > 200:
                    metadata = trafilatura.extract_metadata(downloaded)
                    title = metadata.title if metadata and metadata.title else ''
                    
                    content = result.strip()
                    content = re.sub(r'\s+', ' ', content)
                    
                    parsed_url = urlparse(url)
                    source = parsed_url.netloc.replace('www.', '')
                    
                    return {
                        'title': title,
                        'content': content,
                        'source': source,
                        'url': url,
                        'published_at': '',
                        'author': metadata.author if metadata else '',
                        'extraction_method': 'trafilatura'
                    }
        except Exception as e:
            print(f"Trafilatura error: {e}")
        return None
    
    def _extract_with_readability(self, url):
        """Extract content using readability-lxml"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            doc = Document(response.text)
            title = doc.title()
            content = doc.summary()
            
            if content and len(content) > 200:
                content = re.sub(r'<[^>]+>', '', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                parsed_url = urlparse(url)
                source = parsed_url.netloc.replace('www.', '')
                
                return {
                    'title': title,
                    'content': content,
                    'source': source,
                    'url': url,
                    'published_at': '',
                    'author': '',
                    'extraction_method': 'readability'
                }
        except Exception as e:
            print(f"Readability error: {e}")
        return None
    
    def _scrape_article_content(self, url):
        """Scrape article content from the URL with enhanced extraction"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "sidebar", "advertisement", "ads", "social", "share", "comment", "comments"]):
                element.decompose()
            
            # Extract title with enhanced selectors
            title = ''
            title_selectors = [
                'h1',
                'title',
                '[property="og:title"]',
                '[name="twitter:title"]',
                '.article-title',
                '.post-title',
                '.entry-title',
                '.headline',
                '.story-title',
                '.content-title',
                'h1.article-title',
                'h1.post-title',
                'h1.entry-title',
                '[class*="title"]',
                '[class*="headline"]'
            ]
            
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    if title and len(title) > 10:
                        break
            
            if not title:
                title_elem = soup.find('title')
                if title_elem:
                    title = title_elem.get_text().strip()
            
            # Enhanced content extraction
            content = ''
            content_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.story-body',
                '.article-body',
                '[role="main"]',
                'main',
                '.content',
                '.post-body',
                '.story-content',
                '.article-text',
                '.post-text',
                '.entry-text',
                '.content-body',
                '.article-main',
                '.post-main',
                '.entry-main',
                '[class*="content"]',
                '[class*="article"]',
                '[class*="post"]',
                '[class*="entry"]',
                '[class*="story"]'
            ]
            
            # Try structured content areas first
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Get all text from the content element
                    content = ' '.join(content_elem.get_text().split())
                    if len(content) > 200:  # Ensure we have substantial content
                        print(f"Found content using selector: {selector}")
                        break
            
            # If no structured content found, try paragraph-based extraction
            if not content or len(content) < 200:
                print("Trying paragraph-based extraction...")
                paragraphs = soup.find_all('p')
                content_parts = []
                
                # Filter and collect substantial paragraphs
                for p in paragraphs:
                    text = p.get_text().strip()
                    # Skip short paragraphs, navigation, ads, etc.
                    if len(text) > 30 and not any(skip_word in text.lower() for skip_word in [
                        'cookie', 'privacy', 'subscribe', 'newsletter', 'advertisement', 
                        'sponsored', 'advert', 'click here', 'read more', 'share this',
                        'follow us', 'like us', 'comment', 'comments', 'login', 'sign up'
                    ]):
                        content_parts.append(text)
                
                content = ' '.join(content_parts)
            
            # If still no content, try div-based extraction
            if not content or len(content) < 200:
                print("Trying div-based extraction...")
                # Find the largest text-containing div
                divs = soup.find_all('div')
                largest_div = None
                max_text_length = 0
                
                for div in divs:
                    text = div.get_text().strip()
                    if len(text) > max_text_length and len(text) > 100:
                        # Skip divs that are likely navigation or ads
                        if not any(skip_word in div.get('class', []) for skip_word in [
                            'nav', 'menu', 'sidebar', 'ad', 'advertisement', 'social', 'share'
                        ]):
                            largest_div = div
                            max_text_length = len(text)
                
                if largest_div:
                    content = ' '.join(largest_div.get_text().split())
            
            # Extract source from domain
            parsed_url = urlparse(url)
            source = parsed_url.netloc.replace('www.', '')
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Remove common unwanted text patterns
            unwanted_patterns = [
                r'cookie\s+policy',
                r'privacy\s+policy',
                r'subscribe\s+to\s+our\s+newsletter',
                r'follow\s+us\s+on',
                r'share\s+this\s+article',
                r'read\s+more',
                r'click\s+here',
                r'advertisement',
                r'sponsored\s+content'
            ]
            
            for pattern in unwanted_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
            # Final cleanup
            content = re.sub(r'\s+', ' ', content).strip()
            
            print(f"Extracted content length: {len(content)} characters")
            
            # Ensure we have meaningful content
            if len(content) < 100:
                print(f"Content too short ({len(content)} chars), returning None")
                return None
            
            return {
                'title': title,
                'content': content,
                'source': source,
                'url': url,
                'published_at': '',
                'author': '',
                'extraction_method': 'enhanced_scraping'
            }
            
        except Exception as e:
            print(f"Error scraping article content: {e}")
            return None
    
    def get_news_sources(self):
        """Get list of news sources from News API"""
        try:
            url = f"{self.base_url}/sources"
            params = {
                'apiKey': self.api_key,
                'language': 'en',
                'country': 'us'
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('sources', [])
        except Exception as e:
            print(f"Error getting news sources: {e}")
        
        return []
    
    def analyze_source_reliability(self, source_name):
        """Analyze the reliability of a news source"""
        if not source_name:
            return 'unknown'
            
        reliable_sources = [
            'reuters', 'associated press', 'ap', 'bbc', 'cnn', 'nbc', 'abc', 'cbs',
            'the new york times', 'washington post', 'wall street journal', 'usa today',
            'npr', 'pbs', 'time', 'newsweek', 'the atlantic', 'the economist',
            'scientific american', 'nature', 'science', 'national geographic',
            'forbes', 'bloomberg', 'cnbc', 'fox news', 'msnbc', 'al jazeera'
        ]
        
        unreliable_sources = [
            'infowars', 'breitbart', 'daily stormer', 'stormfront', 'the blaze',
            'natural news', 'health impact news', 'collective evolution',
            'before it\'s news', 'your news wire', 'activist post',
            'conspiracy', 'truth', 'real news', 'alternative news'
        ]
        
        source_lower = source_name.lower()
        
        for reliable in reliable_sources:
            if reliable in source_lower:
                return 'reliable'
        
        for unreliable in unreliable_sources:
            if unreliable in source_lower:
                return 'unreliable'
        
        return 'unknown'
    
    def test_api_connection(self):
        """Test the News API connection"""
        try:
            url = f"{self.base_url}/sources"
            params = {
                'apiKey': self.api_key,
                'language': 'en',
                'country': 'us'
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return True, "API connection successful"
            else:
                return False, f"API error: {response.status_code}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def analyze_article_with_fact_checking(self, article_info):
        """Analyze article using Google Fact Check API and other fact-checking methods"""
        try:
            fact_check_results = {
                'source_reliability': 'unknown',
                'content_analysis': {},
                'fact_check_claims': [],
                'overall_verdict': 'unknown'
            }
            
            # Step 1: Analyze source reliability
            source = article_info.get('source', '')
            fact_check_results['source_reliability'] = self.analyze_source_reliability(source)
            
            # Step 2: Analyze content for fake news indicators
            content = article_info.get('content', '')
            title = article_info.get('title', '')
            full_text = f"{title} {content}".lower()
            
            fact_check_results['content_analysis'] = self._analyze_content_indicators(full_text)
            
            # Step 3: Use Google Fact Check API
            fact_check_claims = self._check_with_google_fact_check(title, content)
            fact_check_results['fact_check_claims'] = fact_check_claims
            
            # Step 4: Determine overall verdict
            fact_check_results['overall_verdict'] = self._determine_overall_verdict(
                fact_check_results['source_reliability'],
                fact_check_results['content_analysis'],
                fact_check_claims
            )
            
            return fact_check_results
            
        except Exception as e:
            print(f"Error in fact-checking analysis: {e}")
            return None
    
    def _analyze_content_indicators(self, text):
        """Analyze text for fake news indicators"""
        fake_indicators = [
            'fake news', 'conspiracy', 'shocking', 'you won\'t believe',
            'doctors hate', 'one weird trick', 'miracle cure', 'secret',
            'they don\'t want you to know', 'mainstream media lies',
            'clickbait', 'viral', 'going viral', 'shocking truth',
            'what they don\'t want you to know', 'insider information',
            'exclusive', 'breaking', 'urgent', 'last chance',
            'limited time', 'act now', 'don\'t miss out',
            'government cover-up', 'suppressed', 'censored',
            'alternative facts', 'fake media', 'fake news media',
            'hoax', 'scam', 'fraud', 'deception', 'manipulation'
        ]
        
        credible_indicators = [
            'study shows', 'research indicates', 'according to experts',
            'peer-reviewed', 'scientific evidence', 'official statement',
            'verified sources', 'fact-checked', 'journal article',
            'academic study', 'university research', 'published in',
            'according to data', 'statistics show', 'official report',
            'government report', 'verified by', 'confirmed by',
            'multiple sources', 'reliable source', 'credible source',
            'evidence-based', 'data-driven', 'research-based'
        ]
        
        fake_count = sum(1 for indicator in fake_indicators if indicator in text)
        credible_count = sum(1 for indicator in credible_indicators if indicator in text)
        
        return {
            'fake_indicators_found': fake_count,
            'credible_indicators_found': credible_count,
            'total_indicators': fake_count + credible_count,
            'fake_ratio': fake_count / (fake_count + credible_count) if (fake_count + credible_count) > 0 else 0
        }
    
    def _check_with_google_fact_check(self, title, content):
        """Check claims using Google Fact Check API (RapidAPI implementation)"""
        try:
            import http.client
            import urllib.parse
            
            # RapidAPI Fact Checker API key
            from .config import GOOGLE_FACT_CHECK_API_KEY
            api_key = GOOGLE_FACT_CHECK_API_KEY
            
            # Extract key phrases for fact-checking
            search_queries = self._extract_fact_check_queries(title, content)
            
            fact_check_results = []
            
            for query in search_queries[:3]:  # Limit to 3 queries to avoid rate limits
                try:
                    # Setup connection
                    conn = http.client.HTTPSConnection("fact-checker.p.rapidapi.com")
                    
                    # Prepare headers
                    headers = {
                        'x-rapidapi-key': api_key,
                        'x-rapidapi-host': "fact-checker.p.rapidapi.com"
                    }
                    
                    # URL encode the query
                    encoded_query = urllib.parse.quote(query)
                    
                    # Make request
                    conn.request("GET", f"/search?query={encoded_query}&limit=10&offset=0&language=en", headers=headers)
                    
                    response = conn.getresponse()
                    data = response.read()
                    
                    if response.status == 200:
                        try:
                            json_data = json.loads(data.decode("utf-8"))
                            
                            # Process the results based on RapidAPI response structure
                            if 'data' in json_data:
                                claims = json_data['data']
                                
                                for claim in claims:
                                    # Extract claim information from RapidAPI response
                                    claim_text = claim.get('claim', '')
                                    claimant = claim.get('claimant', 'Unknown')
                                    claim_date = claim.get('date', '')
                                    publisher = claim.get('publisher', 'Unknown')
                                    review_rating = claim.get('rating', 'Unknown')
                                    review_url = claim.get('url', '')
                                    
                                    fact_check_results.append({
                                        'query': query,
                                        'claim_text': claim_text,
                                        'claimant': claimant,
                                        'claim_date': claim_date,
                                        'publisher': publisher,
                                        'rating': review_rating,
                                        'review_url': review_url
                                    })
                            
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON response for query '{query}': {e}")
                            continue
                    
                    conn.close()
                    
                    # Add small delay to avoid rate limiting
                    import time
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error checking fact for query '{query}': {e}")
                    continue
            
            return fact_check_results
            
        except Exception as e:
            print(f"Error in Google Fact Check API: {e}")
            return []
    
    def _extract_fact_check_queries(self, title, content):
        """Extract key phrases for fact-checking"""
        # Combine title and content
        full_text = f"{title} {content}"
        
        # Extract potential factual claims
        queries = []
        
        # Look for specific patterns that might be factual claims
        import re
        
        # Extract sentences with numbers, dates, or specific claims
        sentences = re.split(r'[.!?]+', full_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length for fact-checking
                # Look for sentences with numbers, dates, or specific factual language
                if re.search(r'\d+', sentence) or any(word in sentence.lower() for word in [
                    'study', 'research', 'found', 'discovered', 'revealed', 'announced',
                    'confirmed', 'verified', 'proven', 'evidence', 'data', 'statistics',
                    'report', 'survey', 'poll', 'election', 'vote', 'percent', '%'
                ]):
                    queries.append(sentence)
        
        # If no specific claims found, use the title and first few sentences
        if not queries:
            if title:
                queries.append(title)
            sentences = content.split('.')[:2]
            queries.extend([s.strip() for s in sentences if len(s.strip()) > 20])
        
        return queries[:5]  # Limit to 5 queries
    
    def _determine_overall_verdict(self, source_reliability, content_analysis, fact_check_claims):
        """Determine overall fact-checking verdict"""
        score = 0
        
        # Source reliability scoring
        if source_reliability == 'reliable':
            score += 2
        elif source_reliability == 'unreliable':
            score -= 2
        
        # Content analysis scoring
        fake_ratio = content_analysis.get('fake_ratio', 0)
        if fake_ratio > 0.6:
            score -= 2
        elif fake_ratio < 0.3:
            score += 1
        
        # Fact check claims scoring
        if fact_check_claims:
            for claim in fact_check_claims:
                rating = claim.get('rating', '').lower()
                if 'false' in rating or 'fake' in rating or 'hoax' in rating:
                    score -= 1
                elif 'true' in rating or 'accurate' in rating or 'correct' in rating:
                    score += 1
        
        # Determine verdict based on score
        if score >= 2:
            return 'likely_real'
        elif score <= -2:
            return 'likely_fake'
        else:
            return 'uncertain' 