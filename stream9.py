import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()


st.markdown(
	"""
	
	<style>
     .main {
     background-color: #F5F5F5;

     }
	</style>

	""",
	unsafe_allow_html=True
)



@st.cache
def get_data():
	taxi_data = pd.read_csv('data/taxidata.csv')
	return taxi_data


with header:
	st.title('Welcome to Header')
	st.text('This is header text....')


with dataset:
	st.header('NYC Taxi dataset')
	st.text('This is dataset text....')

	taxi_data = get_data()
	# st.write = (taxi_data.head())

	st.subheader('Pickup Location ID NYC data')
	pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
	st.bar_chart(pulocation_dist)


with features:
	st.header('The features')

	st.markdown('* **first feature:** I created this feature')
	st.markdown('* **second feature:** I created this feature')



with model_training:
	st.header('Model Training')
	st.text('This is Model Traing where you can add some parameters.....')

	sel_col, disp_col = st.beta_columns(2)
	max_depth = sel_col.slider('max depth of the model?', min_value=10, max_value=100, value=20, step=10)
	# number_of_trees = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No Limit'], index=0)
	n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No Limit'], index=0)

	sel_col.text('Here is a list of fields in data')
	sel_col.write(taxi_data.columns)

	input_feature = sel_col.text_input('Which feature','PULocationID')

	regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
	X = taxi_data[[input_feature]]
	y = taxi_data[['trip_distance']]
	regr.fit(X, y)
	prediction = regr.predict(y)

	disp_col.subheader('Mean Absolute Error')
	disp_col.write(mean_absolute_error(y, prediction))

	# disp_col.subheader('Mean Squared Error')
	# disp_col.write(mean_squared_error(y, prediction))

	# disp_col.subheader('R Squared Score')
	# disp_col.write(r2_score(y, prediction))










