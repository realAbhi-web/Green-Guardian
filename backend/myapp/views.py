from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ml_model.crop_recommendation import predict_crop
from ml_model.crop_yield_predictor import predict_yield
from ml_model.fertiser_recommendation import predict_fertilizer
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import json
import joblib
import pandas as pd
import os

# Create your views here.

from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import requests

OPEN_WEATHER_API_KEY = 'REMOVED_KEY'

GROQ_API_KEY = 'REMOVED_KEY'

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

CEDA_API_KEY = 'REMOVED_KEY'  #valid till Fri Sep 12 2025 03:54:46

CEDA_URL = 'https://api.ceda.ashoka.edu.in/api/v1/agmarknet/'

# Valid till 15 days from now probably end on 20 Sep 2025
MEERSENS_API_KEY = 'REMOVED_KEY' #site https://eaas.meersens.com/api/me

MEERSENS_URL = 'https://api.meersens.com/environment/public'

# Global API config
API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_PARAMS = {
    "api-key": "REMOVED_KEY",
    "format": "json",
    "limit": 1000
}

import re

def load_data():
    try:
        response = requests.get(API_URL, params=API_PARAMS, timeout=10)
        response.raise_for_status()
        return response.json().get("records", [])
    except requests.RequestException as e:
        print(f"API request failed: {str(e)}")
        return []

# Load the data once at module import
DATA = load_data()

def sanitize_input(text, max_length=255):
    """Sanitize string input for safety."""
    if not isinstance(text, str):
        return ""
    # Remove potentially dangerous characters
    cleaned = re.sub(r'[<>"\']', '', text.strip())
    return cleaned[:max_length]


@api_view(['GET'])
def geocoding(request, city):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"

    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200 or not data:
            return Response({"error": "Failed to fetch geocoding data"}, status=response.status_code)
        
        # Return the first result
        result = data["results"]
        return Response(result)
    
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def weather_info(request, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPEN_WEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            return Response({"error": data.get("message", "Failed to fetch weather")}, status=response.status_code)
        
        # Pick the fields you want to return
        result = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "weather": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }
        
        return Response(data)
    
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def aqi_info(request, lat, lon):
    """
    Fetch AQI info from OpenWeather API for given latitude and longitude
    """
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPEN_WEATHER_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return Response({"error": data.get("message", "Failed to fetch AQI")}, status=response.status_code)

        # Air Pollution API returns something like:
        # { "coord": {...}, "list": [ { "main": {"aqi": 1}, "components": {...}, "dt": 1693872000 } ] }
        forecast_list = []
        for entry in data.get('list', []):
            forecast_list.append({
                "aqi": entry['main']['aqi'],               # 1 = Good, 5 = Very Poor
                "components": entry['components'],        # CO, NO2, PM2.5, etc.
                "timestamp": entry['dt']
            })
        
        return Response(data)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def historical_weather(request, lat, lon, start, end):
    """
    Fetch historical weather data from Open-Meteo API
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&hourly=temperature_2m,relative_humidity_2m,soil_moisture_0_to_7cm"
        f"&timezone=auto"
    )

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200 or "error" in data:
            return Response({"error": data.get("reason", "Failed to fetch historical weather")},
                            status=response.status_code)

        # ✅ Use correct keys: "latitude", "longitude"
        historical_data = {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "hourly": data.get("hourly", {})
        }

        return Response(historical_data)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def forecast(request, lat, lon):

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&forecast_days=15&timezone=auto"
    try:
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200 or "error" in data:
            return Response({"error": data.get("reason", "Failed to fetch forecast")}, status=response.status_code)
        
        # Pick the fields you want to return
        forecast_data = {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "daily": data.get("daily", {})
        }
        
        return Response(data)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
# Main crop price tracker
@api_view(['GET', 'POST'])
def crop_price_tracker(request):

    '''
    curl -X POST http://127.0.0.1:8000/api/crop_price_tracker/ \
-H "Content-Type: application/json" \
-d '{
  "crop": "Wheat",
  "state": "Maharashtra",
  "market": "Ulhasnagar"
}'

{"crops":["Alsandikai","Amaranthus","Amla(Nelli Kai)","Amphophalus","Apple","Arecanut(Betelnut/Supari)","Arhar (Tur/Red Gram)(Whole)","Ashgourd","Bajra(Pearl Millet/Cumbu)","Banana","Banana - Green","Beans","Beetroot","Bengal Gram Dal (Chana Dal)","Bengal Gram(Gram)(Whole)","Bhindi(Ladies Finger)","Bitter gourd","Black Gram (Urd Beans)(Whole)","Bottle gourd","Brinjal","Cabbage","Capsicum","Carrot","Castor Seed","Cauliflower","Chikoos(Sapota)","Chili Red","Chilly Capsicum","Cluster beans","Coconut","Coconut Oil","Coconut Seed","Colacasia","Coriander(Leaves)","Corriander seed","Cotton","Cowpea(Veg)","Cucumbar(Kheera)","Cummin Seed(Jeera)","Drumstick","Dry Fodder","Elephant Yam (Suran)","French Beans (Frasbean)","Garlic","Ginger(Dry)","Ginger(Green)","Grapes","Green Chilli","Green Gram (Moong)(Whole)","Green Peas","Ground Nut Seed","Groundnut","Groundnut (Split)","Guar","Guar Seed(Cluster Beans Seed)","Guava","Gur(Jaggery)","Indian Beans (Seam)","Isabgul (Psyllium)","Jowar(Sorghum)","Kabuli Chana(Chickpeas-White)","Knool Khol","Kulthi(Horse Gram)","Lak(Teora)","Lemon","Lentil (Masur)(Whole)","Lime","Linseed","Little gourd (Kundru)","Mahua","Maize","Mango (Raw-Ripe)","Mataki","Methi(Leaves)","Mint(Pudina)","Mousambi(Sweet Lime)","Mustard","Neem Seed","Onion","Onion Green","Orange","Paddy(Dhan)(Basmati)","Paddy(Dhan)(Common)","Papaya","Peas Wet","Peas cod","Pineapple","Pointed gourd (Parval)","Pomegranate","Potato","Pumpkin","Raddish","Rice","Ridgeguard(Tori)","Round gourd","Safflower","Seetapal","Sesamum(Sesame,Gingelly,Til)","Snakeguard","Soanf","Soyabean","Spinach","Sponge gourd","Sweet Potato","Sweet Pumpkin","Tapioca","Thondekai","Tinda","Tomato","Turmeric","Turmeric (raw)","Water Melon","Wheat","buttery"],"result":[],"error":"No data found for the given crop, state, and market."}%                                    


    '''


    crops = sorted({record['commodity'] for record in DATA if record.get('commodity')})
    result = []
    error = None

    if request.method == 'POST':
        crop = sanitize_input(request.data.get('crop', ''), 100)
        state = sanitize_input(request.data.get('state', ''), 100)
        market = sanitize_input(request.data.get('market', ''), 100)

        if not crop or not state or not market:
            error = "All fields (crop, state, market) are required."
        else:
            result = [
                r for r in DATA
                if r.get('commodity', '').lower() == crop.lower()
                and r.get('state', '').lower() == state.lower()
                and r.get('market', '').lower() == market.lower()
            ]
            if not result:
                error = "No data found for the given crop, state, and market."

    return Response({
        'crops': crops,
        'result': result,
        'error': error
    })

@api_view(['GET'])
def get_states(request):
    crop = sanitize_input(request.GET.get('crop', ''), 100).lower()
    if not crop:
        return Response([])
    states = sorted({r['state'] for r in DATA if r.get('commodity', '').lower() == crop})
    return Response(states)

@api_view(['GET'])
def get_markets(request):
    crop = sanitize_input(request.GET.get('crop', ''), 100).lower()
    state = sanitize_input(request.GET.get('state', ''), 100).lower()
    
    if not crop or not state:
        return Response([])
    
    markets = sorted({
        r['market'] for r in DATA
        if r.get('commodity', '').lower() == crop and r.get('state', '').lower() == state
    })
    return Response(markets)

@api_view(['GET'])
def soil_data(request, lat, lon):
    """
    Fetch soil data from Open-Meteo API
    """
    url = f"http://127.0.0.1:8000/soil/report?lat={lat}&lon={lon}"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200 or "error" in data:
            return Response({"error": data.get("reason", "Failed to fetch soil data")}, status=response.status_code)

        soil_data = {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "soil": data.get("soil", {})
        }

        return Response(data)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def water_data(request, lat, lon):
    url = f"{MEERSENS_URL}/water/current"
    headers = {"apikey": MEERSENS_API_KEY}
    params = {
        "lat": lat,
        "lng": lon,
        "index_type": "meersens",
        "health_recommendations": "true"
    }
    r = requests.get(url, headers=headers, params=params)
    data = r.json()

    if not data.get("found", True):
        return JsonResponse({"message": "No water data available for this location."}, status=404)

    return JsonResponse(data, safe=False, status=r.status_code)

@csrf_exempt   # (disable CSRF for testing curl/postman; later use proper auth)
def crop_recommendation_view(request):

    '''
    
curl -X POST http://127.0.0.1:8000/api/crop-recommendation/ -d "N=90&P=40&K=40&temperature=25&humidity=70&ph=6.5&rainfall=200"


{"success": true, "prediction": "coffee"}%                                          

    
    '''


    if request.method == "POST":
        try:
            N = float(request.POST.get("N"))
            P = float(request.POST.get("P"))
            K = float(request.POST.get("K"))
            temperature = float(request.POST.get("temperature"))
            humidity = float(request.POST.get("humidity"))
            ph = float(request.POST.get("ph"))
            rainfall = float(request.POST.get("rainfall"))

            prediction = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            return JsonResponse({"success": True, "prediction": prediction})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=400)

    return JsonResponse({"error": "Only POST allowed"}, status=405)

@csrf_exempt
def crop_yield_prediction(request):
    '''
    
    curl -X POST http://127.0.0.1:8000/api/crop-yield/ \        
-H "Content-Type: application/json" \
-d '{
  "area": "Punjab",
  "item": "Wheat",
  "season": "Kharif",
  "crop_year": 2025,
  "average_rainfall": 200,
  "pesticides": 5,
  "annual_rainfall": 1800
}'

{"success": true, "prediction": 0.8999999761581421}%                                 

    
    '''
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST request required"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)

    result = predict_yield(data)
    status_code = 200 if result.get("success") else 400
    return JsonResponse(result, status=status_code)


@csrf_exempt
def fertilizer_recommendation(request):

    ''' 
    ENDPOINT REQUEST EXAMPLE (POSTMAN/CURL):

    $ curl -X POST http://127.0.0.1:8000/api/fertilizer-recommendation/ \
-H "Content-Type: application/json" \
-d '{
    "temperature": 30,
    "humidity": 70,
    "moisture": 25,
    "soil_type": "Loamy",
    "crop_type": "Wheat",
    "nitrogen": 50,
    "potassium": 30,
    "phosphorous": 20
}'

{"success": true, "recommendation": "Urea"}%   
~ ⌚ 14:49:06
$ curl https://api.groq.com/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer REMOVED_KEY" \
  -d '{
    "model": "llama-3.1-8b-instant",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain how AI works in a few words."}
    ],
    "max_completion_tokens": 256
  }'

{"id":"chatcmpl-ba8597bd-e20b-43a3-9393-7c0b45f34bb0","object":"chat.completion","created":1757841538,"model":"llama-3.1-8b-instant","choices":[{"index":0,"message":{"role":"assistant","content":"AI works by processing data with algorithms to generate predictions or actions."},"logprobs":null,"finish_reason":"stop"}],"usage":{"queue_time":0.052068693,"prompt_tokens":51,"prompt_time":0.006643667,"completion_tokens":14,"completion_time":0.043850921,"total_tokens":65,"total_time":0.050494588},"usage_breakdown":null,"system_fingerprint":"fp_510c177af0","x_groq":{"id":"req_01k53rafjxfvqt5qhrem6rvvae"},"service_tier":"on_demand"}

~ ⌚ 14:49:07
$ 



    
    '''
    

    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST request required"}, status=400)
    
    try:
        data = json.loads(request.body)  # read JSON payload
        print("REQUEST BODY:", request.body)
        print("PARSED DATA:", data)
        temperature = float(data.get("temperature") or 0)
        humidity = float(data.get("humidity") or 0)
        moisture = float(data.get("moisture") or 0)
        nitrogen = float(data.get("nitrogen") or 0)
        potassium = float(data.get("potassium") or 0)
        phosphorous = float(data.get("phosphorous") or 0)
        soil_type = data.get("soil_type") or ""
        crop_type = data.get("crop_type") or ""

        
        # call your model
        recommendation = predict_fertilizer(
        temperature=temperature,  # ✅ fixed
        humidity=humidity,
        moisture=moisture,
        soil_type=soil_type,
        crop_type=crop_type,
        nitrogen=nitrogen,
        potassium=potassium,
        phosphorous=phosphorous
)

        
        return JsonResponse({"success": True, "recommendation": recommendation})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)


# Initialize chatbot (do this once)
# chatbot = ChatBot("FarmerBot")
# trainer = ChatterBotCorpusTrainer(chatbot)
# trainer.train("chatterbot.corpus.english")  # You can create your own corpus later

@csrf_exempt
def chat_with_bot(request):
    '''
    $ curl -X POST http://127.0.0.1:8000/api/chat/ \       
-H "Content-Type: application/json" \
-d '{"query": "Which crops are best for rainy season in Punjab?"}'

{"success": true, "reply": "Punjab is one of the most fertile regions in India, and the rainy season presents a unique set of opportunities for farmers. Here are some of the best crops that can be grown during the rainy season in Punjab:\n\n1. **Pulses**: Pulses, such as arhar (pigeon pea), moong (green gram), and urad (black gram), are ideal for the rainy season. They can grow in waterlogged conditions and are a great source of protein for livestock and human consumption.\n2. **Hybrid Rice**: Hybrid rice varieties, such as IR-64 and IR-72, are specifically designed to thrive in wet conditions and can be grown during the rainy season. They have a shorter maturity period and can produce high yields.\n3. **Sugarcane**: Sugarcane is another lucrative crop that can be grown during the rainy season. It's a drought-tolerant crop that can grow in waterlogged conditions, and its juice can be harvested and sold throughout the year.\n4. **Maize**: While maize is a crop that typically requires well-drained soil, there are some varieties that can tolerate waterlogging to a certain extent. These varieties can be grown during the rainy season, but it's essential to ensure proper drainage"}%
    '''
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST method required."}, status=405)
        
    try:
        data = json.loads(request.body)
        user_query = data.get("query")
        if not user_query:
            return JsonResponse({"success": False, "error": "Missing 'query' field in request."}, status=400)

        # Customize system prompt for farmers
        system_prompt = (
            "You are an expert agricultural assistant. Answer farmers' questions clearly, "
            "give practical advice on crops, soil, irrigation, and weather, and avoid generic AI answers."
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "max_completion_tokens": 256
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            return JsonResponse({"success": False, "error": response_data.get("error", "Unknown error")}, status=500)

        # Extract the assistant's reply
        reply = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return JsonResponse({"success": True, "reply": reply})

    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON."}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)



