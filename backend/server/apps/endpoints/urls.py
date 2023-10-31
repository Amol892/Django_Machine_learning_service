
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import *

router = DefaultRouter(trailing_slash=False)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlalgorithms", MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")
router.register(r"abtests", ABTestViewset, basename="abtests" )

urlpatterns = [
    
    path("api/v1/", include(router.urls)),
    path("api/v1/<str:endpoint_name>/predict", predictionAPIVIew.as_view(), name="predict"),
    path("api/v1/stop_ab_test/<int:ab_test_id>/",StopABTestView.as_view(), name="stop_ab")
    
]