"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib, os, pickle

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud

# Data dependencies
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)

# Vectorizer
news_vectorizer = open("resources/tfidvectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

news_vect = pickle.load(open("resources/tfidvectorizer.pkl", 'rb'))

# Load your raw data
raw = pd.read_csv("resources/train.csv")

def word_cloud(df,class_no,class_name):
  """
  This function generates word cloud visualizations across different classes.

  Parameters:
    df (obj): Data frame.
    class_no (int): Class number
    class_name (obj): Class name

   Returns:
    word cloud visual
  """

  sentiment_class = ' '.join([text for text in df['message'][df['sentiment'] == class_no]])
  from wordcloud import WordCloud
  wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,
                        background_color="white").generate(sentiment_class)

  plt.figure(figsize=(10, 7))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.title('WordCloud for' + " " + class_name)
  plt.axis('off')
  st.pyplot()

def data_viz(df):
	# getting a visual of train data and the count of sentiment
	# this gives us an unfiltered overview of the most common sentiment about climate change 
	fig = plt.subplots(figsize=(12,6))
	# sns.set_theme(style="darkgrid")
	senti = df.groupby('sentiment', as_index=False).count()
	plt.bar(senti['sentiment'].astype('str'), senti['message'])
	plt.xlabel('Sentiment')
	plt.ylabel('# of observations')
	st.pyplot()

	st.markdown(""""
		We can clearly see that there are far more tweets in the sentiment category '1', corresponding to those who do believe in climate change. We can get a better idea of distributions through a pie chart.
	""")

	fig = plt.subplots(figsize=(12,6))
	plt.pie(senti['message'], labels=senti['sentiment'].astype('str'), autopct='%1.1f%%', explode=[0.05, 0.05, 0.05, 0.05])
	plt.title('Percentage of sentiments in train_data')
	st.pyplot()
	st.markdown("""
		The pie chart shows clearly that over 50% of our dataset is comprised of those who do believe in climate change. This is indicative of unbalanced data. Imbalanced data may be problematic, as the model will have far more samples of the class present in a greater proportion to train on, which may impact the accuracy of the model in predicting the other classes. However, we will keep the ratios between the sentiment categories the same, as this may be an indication of real-life ratios that exist and so would allow our model to be more generalisable to other real-life data.
		#### Frequently used words
		In order to visualise the words used in our dataset, we create WordClouds:
	""")
	from wordcloud import WordCloud
	sentiment = {"-1": "Anti", "0": "Neutral", "1": "Pro", "2": "News"}
	for key in sentiment.keys(): 
		word_cloud(raw, int(key), sentiment[key])

	st.markdown("""
		These word clouds give us a clearer idea of what words are particularly frequent in each category. As we can see, 'climate', 'change', 'global' and 'warming' are very prominent across all four senitment classes, however this is to be expected as the dataset comprises of tweets surrounding this topic. Another frequent term we can see is 'link'. This is where any urls were replaced with the word 'link', and we can see that this is very prominent in the sentiment 2. This makes sense, as the sentiment 2 corresponds to tweets with links to news articles.

We can see other examples of relevant phrases in each word cloud such as 'man made' and 'scam' in the -1 sentiment word cloud (non-believers of climate change), 'believe' and 'real' in the 1 sentiment word cloud (believers of climate change), and 'paris agreement', 'carbon dioxide' and 'scientist' in sentiment 2 word cloud (tweets with links to news articles). In the sentiment 0 (neutral) word cloud, we can see some more vague and non-indicative words occuring often such as 'thought', 'today', or 'club penguin'.
	""")



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information","Data Visualisations"]
	selection = st.sidebar.selectbox("Choose Option", options)

	if selection == "Data Visualisations":
		st.markdown("""
			## Exploratory Data Analysis
			### Sentiment Distribution
			Exploratory data analysis allows us to have a look at our data and the relationships between aspects of the data. We can firstly look at the distribution of the sentiments.
		""")
		data_viz(raw)

		
		

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		st.markdown("""
        ### Use some of the built-in models to find out the sentiment of your tweet.
        #### **Follow Instructions Below:**
        1. Select a model to classify your tweet from the dropdown menu below
        2. Type in your tweet on the text area
        3. Click the 'Classify' button and see the results below.
    """)
		# List of available models for tweet classification
		models = ["Logistic Regression","Support Vector Machine","Random Forests"]
		chosen_model = st.selectbox("Pick a classification model: ", models)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = news_vect.transform([tweet_text])
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			# predictor = joblib.load(open(os.path.join("resources/lg_model.pkl"),"rb"))
			if chosen_model == "Logistic Regression":
				predictor = pickle.load(open("resources/lg_model.pkl", 'rb'))
			if chosen_model == "Support Vector Machine":
				predictor = pickle.load(open("resources/svm_model.pkl", 'rb'))
			if chosen_model == "Random Forests":
				predictor = pickle.load(open("resources/rf_model.pkl", 'rb'))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			sentiment = {"-1": "Anti", "0": "Neutral", "1": "Pro", "2": "News"}
			st.success("Text Categorized as: {}".format(sentiment[str(prediction[0])]))


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
