from prompt import prompts

import streamlit as st
import pandas as pd
import pycountry
from collections import Counter
from sklearn.cluster import KMeans
import time

from utility import (get_category, get_channel_info, get_video_details, get_video_statistics, 
                    get_summarized, closest_to_centroid, inference, get_best_similar_video, 
                    postprocess_model_output, process_distribution)

from models import model

@st.cache_data
def fetch_channel_data(channel_id):
    channel_info = get_channel_info(channel_id)
    channel_data = channel_info['items'][0]
    snippet = channel_data.get('snippet', {})
    content_details = channel_data.get('contentDetails', {})
    statistics = channel_data.get('statistics', {})
    default_image = channel_info['items'][0]['snippet']['thumbnails']['medium']['url']
    datetime = pd.to_datetime(channel_info['items'][0]['snippet']['publishedAt'])
    publishedAt = datetime.strftime('%Y-%m-%d')
    country_code = channel_info['items'][0]['snippet']['country']
    country = pycountry.countries.get(alpha_2=country_code)
    country = country.name
    video_ids = get_video_details(content_details)
    all_video_details = get_video_statistics(video_ids)
    
    if not all_video_details.empty:
        all_video_details['Published_date'] = pd.to_datetime(all_video_details['Published_date'])
        all_video_details['Views'] = pd.to_numeric(all_video_details['Views'])
        all_video_details['Likes'] = pd.to_numeric(all_video_details['Likes'])
        all_video_details['Comments'] = pd.to_numeric(all_video_details['Comments'])
        
        all_video_details['Month'] = all_video_details['Published_date'].dt.strftime('%b')
        all_video_details['Day_of_Month'] = all_video_details['Published_date'].dt.day
        all_video_details['Day_of_Week'] = all_video_details['Published_date'].dt.day_name()
        
        all_video_details['Published_date_str'] = all_video_details['Published_date'].dt.strftime('%Y-%m-%d')
    
        # Sorted 
        sorted_video = all_video_details.sort_values(by="Views", ascending=False)
        
        # Video upload distributions
        video_uploaded_month = process_distribution(all_video_details, 'Month')
        video_uploaded_day = process_distribution(all_video_details, 'Day_of_Month')
        video_uploaded_weekday = process_distribution(all_video_details, 'Day_of_Week')

        # Likes distributions
        likes_by_month = process_distribution(all_video_details, 'Month', 'Likes')
        likes_by_day = process_distribution(all_video_details, 'Day_of_Month', 'Likes')
        likes_by_weekday = process_distribution(all_video_details, 'Day_of_Week', 'Likes')

        # Views distributions
        views_by_month = process_distribution(all_video_details, 'Month', 'Views')
        views_by_day = process_distribution(all_video_details, 'Day_of_Month', 'Views')
        views_by_weekday = process_distribution(all_video_details, 'Day_of_Week', 'Views')
    
    return {
        'channel_info': channel_info,
        'snippet': snippet,
        'statistics': statistics,
        'default_image': default_image,
        'publishedAt': publishedAt,
        'country': country,
        'all_video_details': all_video_details,
        'sorted_video': sorted_video,
        'video_uploaded_month': video_uploaded_month,
        'video_uploaded_day': video_uploaded_day,
        'video_uploaded_weekday': video_uploaded_weekday,
        'likes_by_month': likes_by_month,
        'likes_by_day': likes_by_day,
        'likes_by_weekday': likes_by_weekday,
        'views_by_month': views_by_month,
        'views_by_day': views_by_day,
        'views_by_weekday': views_by_weekday
    }

def get_recommendations(all_video_details):
    top_100_videos = all_video_details.sort_values(by="Views", ascending=False).head(100)
    all_tags = set()
    for tags in top_100_videos['Tags']:
        for tag in tags:
            all_tags.add(tag)
    all_tags_list = [tag for tags in top_100_videos['Tags'] for tag in tags]
    tag_counter = Counter(all_tags_list)
    most_common_tags = [tag for tag, count in tag_counter.most_common(50)]
    top_category_ids = list(set(top_100_videos['Category_id']))
    top_category, _ = get_category(top_category_ids)
    result, text = get_best_similar_video(most_common_tags, top_category)
    top_100_videos['content'] = top_100_videos['Title'] + top_100_videos['Description']
    temp_top = top_100_videos['content'].tolist()
    
    embeddings = model.encode(temp_top)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    centroids = kmeans.cluster_centers_
    representative_indices = [closest_to_centroid(centroid, embeddings) for centroid in centroids]
    representative_strings = [temp_top[idx] for idx in representative_indices]
    summarize_1 = get_summarized(result)
    summarize_2 = get_summarized(representative_strings)
    all_videos_description = ""
    
    for i, text in enumerate(summarize_1):
        all_videos_description += f"Video {i} : " + text + ". \n"
        
    for j, text in enumerate(summarize_2):
        all_videos_description += f"Video {i + j} : " + text + ". \n"
        
    return inference(all_videos_description)

st.set_page_config(layout="wide", page_title="YouTube Channel Analyzer")

st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: auto;
        display: block;
        margin: 0 auto;
    }
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem;
    }
    .channel-id-input {
        max-width: 600px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .channel-info {
        display: flex;
        align-items: flex-start;
    }
    .channel-text {
        flex: 2;
        padding-right: 2rem;
    }
    .channel-image {
        flex: 1;
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="title-container"><h1>YouTube Channel Analyzer</h1></div>', unsafe_allow_html=True)


col1, col2, col3 = st.columns([1,2,1])
with col2:
    channel_id = st.text_input('Enter YouTube Channel ID', key="channel_id_input")
    st.markdown(
        """
        <style>
        [data-testid="stTextInput"] {
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if channel_id:
    with st.spinner('Fetching channel data...'):
        data = fetch_channel_data(channel_id)

    col_left, col_middle, col_right = st.columns([1, 3, 1])
        
    with col_middle:
        st.markdown('<div class="channel-info">', unsafe_allow_html=True)
        col_text, col_image = st.columns([1, 1])

        with col_text:
            st.subheader("Channel Information")
            st.markdown(f"""
            **Channel Name:** {data['snippet']['title']}  
            **Subscribers:** {data['statistics']['subscriberCount']}  
            **Total Views:** {data['statistics']['viewCount']}  
            **Total Videos:** {data['statistics']['videoCount']}  
            **Published At:** {data['publishedAt']}  
            **Country:** {data['country']}
            """)

        with col_image:
            st.image(data['default_image'], width=200)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Video Statistics')
        st.dataframe(data['sorted_video'], height=300)
        
    with col2:
        st.subheader('Monthly Video Upload Distribution')
        st.bar_chart(data['video_uploaded_month'].set_index('Month'))

    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Monthly Average Likes')
        st.line_chart(data['likes_by_month'].set_index('Month'))

    with col4:
        st.subheader('Monthly Average Views')
        st.line_chart(data['views_by_month'].set_index('Month'))
        
    col5, col6 = st.columns(2)
    with col5:
        st.subheader('Daily Video Upload Distribution')
        st.line_chart(data['video_uploaded_day'].set_index('Day_of_Month'))

    with col6:
        st.subheader('Daily Average Likes')
        st.line_chart(data['likes_by_day'].set_index('Day_of_Month'))
        
    col7, col8 = st.columns(2)
    with col7:
        st.subheader('Daily Average Views')
        st.line_chart(data['views_by_day'].set_index('Day_of_Month'))

    with col8:
        st.subheader('Day of Week Average Video Upload Distribution')
        st.line_chart(data['video_uploaded_weekday'].set_index('Day_of_Week'))
        
    col9, col10 = st.columns(2)
    with col9:
        st.subheader('Day of Week Average Likes')
        st.line_chart(data['likes_by_weekday'].set_index('Day_of_Week'))

    with col10:
        st.subheader('Day of Week Average Views')
        st.line_chart(data['views_by_weekday'].set_index('Day_of_Week'))

    st.markdown("---")

    if st.button('Get Recommendations'):
        with st.spinner('Analyzing channel content...'):
            recommendations = get_recommendations(data['all_video_details'])

        if recommendations is None:
            st.write("Sorry, the model couldn't process the request right now.")
        else:
            st.subheader('Content Recommendations')
            st.write(postprocess_model_output(recommendations))
