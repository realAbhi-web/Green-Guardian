from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from .views import weather_info, aqi_info, historical_weather, forecast, geocoding, crop_price_tracker, soil_data, water_data, crop_recommendation_view, crop_yield_prediction, get_markets, get_states, fertilizer_recommendation, chat_with_bot
urlpatterns = [
    path('weather/<str:city>/', weather_info,name='weather_info'),
    path('aqi/<str:lat>/<str:lon>/', aqi_info,name='aqi_info'),
    path('history/<str:lat>/<str:lon>/<str:start>/<str:end>/', historical_weather,name='history_info'),
    path('forecast/<str:lat>/<str:lon>/', forecast,name='forecast_info'),
    path('geocode/<str:city>/', geocoding,name='geocode_info'),
    path('crop_price_tracker/', crop_price_tracker, name='crop_price_tracker'),
    path('get_states/', get_states, name='get_states'),
    path('get_markets/', get_markets, name='get_markets'),
    path('soil/<str:lat>/<str:lon>/', soil_data,name='soil_data'),
    path('water/<str:lat>/<str:lon>/', water_data,name='water_data'),
    path('crop-recommendation/', csrf_exempt(crop_recommendation_view), name='crop_recommendation'),  # New endpoint
    path("crop-yield/", crop_yield_prediction, name="crop_yield_prediction"),
    # path('fertilizer-recommendation/', csrf_exempt(crop_recommendation_view), name='fertilizer_recommendation'),  # New endpoint
    path('fertilizer-recommendation/', csrf_exempt(fertilizer_recommendation), name='fertilizer_recommendation'),
    path("chat/", chat_with_bot, name="chat_with_bot"),

]
