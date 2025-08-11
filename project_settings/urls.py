from django.contrib import admin
from django.urls import path, include
from ml_app.views import login_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', login_view, name='login'),  # Root URL loads login page
    path('ml_app/', include('ml_app.urls', namespace='ml_app')),  # ml_app routes under /ml_app/
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)