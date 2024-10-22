import torch
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain_core.output_parsers import StrOutputParser

import pycountry

from datetime import datetime, timedelta

import pandas as pd

import requests

from models import summarizer, generator, tokenizer
from prompt import prompts

from constants import APIS, BASE_URL

import numpy as np
import re

import calendar

def get_channel_info(channel_id):
    current_api_key_index = 0
    while current_api_key_index < len(APIS):
        try:
            youtube = build('youtube', 'v3', developerKey=APIS[current_api_key_index])
            response = youtube.channels().list(part='snippet,contentDetails,statistics',
                    id=channel_id).execute()

            return response
        except HttpError as e:
            print('An HTTP error occurred:', e)
        current_api_key_index += 1

    return response

def get_video_details(content_details):
    current_api_key_index = 0
    while current_api_key_index < len(APIS):
        try:
            youtube = build('youtube', 'v3', developerKey=APIS[current_api_key_index])
            request_ = youtube.playlistItems().list(part='snippet,contentDetails',
                    playlistId=content_details['relatedPlaylists']['uploads'],maxResults=50)

            video_details = request_.execute()
        except:
            video_details = None

        if video_details is not None:
            break

        current_api_key_index += 1
    
    if video_details is not None:
        # Taking all video id for specific information about each video
        video_ids = []
        for i in range(len(video_details['items'])):
            video_ids.append(video_details['items'][i]['contentDetails']['videoId'])

        next_page_token = video_details.get('nextPageToken', '')

    if next_page_token:
        more_pages = True
    else:
        more_pages = False

    # Due to youtube data api limit, the following logic is necessary to take more than 50 videos of user
    while more_pages:
        if next_page_token:
            request_ = youtube.playlistItems().list(part='contentDetails,snippet,status',
                    playlistId=content_details['relatedPlaylists']['uploads'],maxResults=50,
                    pageToken=next_page_token)
            try:
                video_details = request_.execute()

                for i in range(len(video_details['items'])):
                    video_ids.append(video_details['items'][i]['contentDetails']['videoId'])
                next_page_token = video_details.get('nextPageToken', "end")

                if next_page_token == "end":
                    break
            except:
                pass

    return video_ids

def get_video_statistics(video_ids):
    # Getting videos statistics
    all_video_details = []
    if video_ids is not None:
        current_api_key_index = 0
        while current_api_key_index < len(APIS):
            try:
                for i in range(0, len(video_ids), 50):
                    params = {'part': 'snippet,statistics',
                        'id': ','.join(video_ids[i:i + 50]),'key': APIS[current_api_key_index]}

                    response = requests.get(f'{BASE_URL}/videos', params=params)
                    video_details = response.json()
                    if "items" in video_details:
                        for video in video_details['items']:
                            tags = []
                            defaultLanguage = None
                            audioLanguage = None
                            likeCount = 0
                            viewCount = 0

                            if "defaultLanguage" in video['snippet']:
                                default_language = video['snippet']['defaultLanguage']

                            if "tags" in video['snippet']:
                                tags = video['snippet']['tags']

                            if "likeCount" in video['statistics']:
                                like_count = video['statistics']['likeCount']

                            if "viewCount" in video['statistics']:
                                view_count = video['statistics']['viewCount']

                            if "commentCount" in video['statistics']:
                                comment_count = video['statistics']['commentCount']

                            if 'defaultAudioLanguage' in video['snippet']:
                                audio_language = video['snippet']['defaultAudioLanguage']

                            category_id = video['snippet']['categoryId']

                            video_stats = dict(Title=video['snippet']['title'],
                                Published_date=video['snippet']['publishedAt'],Default_language=default_language,
                                Views=view_count,Likes=like_count,Comments=comment_count,
                                Category_id=category_id,Tags=tags,
                                Description=video['snippet']['description'],Audio_language=audio_language)

                            all_video_details.append(video_stats)
                break
            except:
                pass
                current_api_key_index += 1
                
        all_video_details = pd.DataFrame(all_video_details)
    else:
        all_video_details = None

    return all_video_details

def get_category(top_category_ids):
    top_category = dict()
    reverse_category = dict()
    for i in range(0, len(top_category_ids), 50):
        category_ids_batch = ','.join(top_category_ids[i:i + 50])

        # Make the request to retrieve information about multiple videos
        if category_ids_batch:
            current_api_key_index = 0
            while current_api_key_index < len(APIS):
                try:
                    current_api_key = APIS[current_api_key_index]
                    params = {
                        'part': 'snippet',
                        'id': category_ids_batch,
                        'key': current_api_key
                    }
        
                    response = requests.get(f'{BASE_URL}/videoCategories', params=params)
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as e:
                    data = None
        
                current_api_key_index += 1
        
                if current_api_key_index == len(APIS):
                    Text = "API IS OUT OF ORDER FOR TODAY"
        
            # Process and print the video category data
            if 'items' in data:
                for video in data['items']:
                    top_category[video['id']] = video['snippet']['title']
                    reverse_category[video['snippet']['title']] = video['id']
        else:
            break

    return top_category, reverse_category

def get_best_similar_video(tags, category_ids):
    """
        Make an API call to retrieve video recommendations based on specified parameters.

        Parameters:
        - categoryId (str): The category ID of the videos to search for.
        - start_date (datetime): The start date for filtering video results.
        - tag (str): The tag or keyword to include in the search query.
        - country_code (str): The country code for region-based filtering.

        Returns:
        - result (dict): A dictionary containing video search and details information.
        - Text (str): Additional information or error message.
    """
    months_back = 36
    today = datetime.now()
    start_date = today - timedelta(days=30 * months_back)

    for category_id in category_ids:
        current_api_key_index = 0
        while current_api_key_index < len(APIS):
            try:
                current_api_key = APIS[current_api_key_index]
                youtube = build('youtube', 'v3', developerKey=current_api_key)

                # Perform a video search based on the specified parameters
                search_response = youtube.search().list(part='snippet',q=tags,order='viewCount',type='video',
                    videoCategoryId=category_id,maxResults=10).execute()

                ids = []
                for item in search_response['items']:
                    video_date_str = item['snippet']['publishedAt']
                    video_date = pd.to_datetime(video_date_str, format='%Y-%m-%dT%H:%M:%SZ')

                    if video_date >= start_date:
                        ids.append(item['id']['videoId'])

                result = youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(ids)
                ).execute()

                break
            except Exception as e:
                result = None

            current_api_key_index += 1

    video = []
    for item in result['items']:
        video.append(item['snippet']['title'] + " \n " + item['snippet']['description'] )

    text = None
    # Check if the API key index exceeded the available keys
    if current_api_key_index - 1 < len(APIS):
        text = "API IS OUT OF ORDER FOR TODAY"

    return video,text

def inference(video_description):
    input_ = prompts.format(top_video=video_description)
    
    try:
        recommendations = generator(
            input_,
            max_length=2000,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        return recommendations[0]['generated_text']
    except Exception as e:
        print(f"An error occurred during generation: {str(e)}")
        return None


def get_summarized(result):
    summarized_text = []
    for text in result:
        if len(text) > 1024:
            text = text[:1024]

        if len(text) <= 20:
            summarized_text.append(text)
        else:
            summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
            summarized_text.append(summary[0]['summary_text'])

    return summarized_text


def closest_to_centroid(centroid, embeddings):
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return np.argmin(distances)

def postprocess_model_output(model_output):
    print(model_output)
    recommendation = model_output.split("Provide your recommend content below:")[1]
    output_parser = StrOutputParser()
    recommendation = output_parser.parse(recommendation)
    
    return recommendation

def process_distribution(df, group_by, value_column=None):
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    day_of_week_order = list(calendar.day_name)
        
    if value_column:
        distribution = df.groupby(group_by, as_index=False)[value_column].mean()
    else:
        distribution = df.groupby(group_by, as_index=False).size().rename(columns={'size': 'Count'})

    if group_by == 'Month':
        distribution[group_by] = pd.Categorical(distribution[group_by], categories=month_order, ordered=True)
    elif group_by == 'Day_of_Week':
        distribution[group_by] = pd.Categorical(distribution[group_by], categories=day_of_week_order, ordered=True)

    return distribution.sort_values(group_by)