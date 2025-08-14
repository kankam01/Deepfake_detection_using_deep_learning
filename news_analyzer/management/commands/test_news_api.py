from django.core.management.base import BaseCommand
from news_analyzer.services import NewsAPIService

class Command(BaseCommand):
    help = 'Test News API connection'

    def handle(self, *args, **options):
        service = NewsAPIService()
        success, message = service.test_api_connection()
        
        if success:
            self.stdout.write(
                self.style.SUCCESS(f'✓ {message}')
            )
        else:
            self.stdout.write(
                self.style.ERROR(f'✗ {message}')
            ) 