from django.urls import path
from . import views
from .views import about, index, predict_page, cuda_full, login_view, selection_view

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('selection/', views.selection_view, name='selection'),
    path('', views.index, name='home'),  # Deepfake detection homepage
    path('about/', views.about, name='about'),
    path('predict/', views.predict_page, name='predict'),
    path('cuda_full/', views.cuda_full, name='cuda_full'),
]