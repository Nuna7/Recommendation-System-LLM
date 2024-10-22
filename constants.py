from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('PYTRENDS_API_KEY')

BASE_URL = 'https://www.googleapis.com/youtube/v3'

# Add more if you have more than 1 API_KEY.
APIS = [API_KEY]