import streamlit as st
from src.skin_color import estimate_skin
from PIL import Image
from src.infer import predict_test, process_preds

st.set_page_config(
    page_title="MetaSkin",  # default page title
    layout="centered"
)

page = st.selectbox("Choose your page", ["Step 1", "Step 2"])

st.write('<h1 style="font-weight:400; color:red">MetaSkin</h1>', unsafe_allow_html=True)


if page == 'Step 1':
    st.write('Please upload a portrait of yourself')

    userFile = st.file_uploader('Portrait of your face', type=['jpg', 'jpeg', 'png'])

    if st.button('Process Portrait'):
        if userFile is not None:
            img = Image.open(userFile)
            img.save('skincolor.png')
            race = estimate_skin('skincolor.png')
            if race != 'African American':
                st.warning('You have a higher chance')
            seg = Image.open('segment.jpg')
            st.image(seg, width=None, caption='output segmented image')
        else:
            st.error('File not found')
elif page == 'Step 2':
    st.write('Please upload an image of your lesion')
    with st.form(key='my_form'):
        local = st.selectbox("Where is the lesion located?", ['anterior torso', 'head/neck', 'lateral torso', 'lower extremity', 'oral', 'palms/soles', 'posterior torso', 'torso', 'upper extremity', 'none of the above'])
        sex = st.selectbox("What is your sex?", ['Male', 'Female'])
        age = st.number_input("What is your age?", step=1)
        userFile = st.file_uploader('Upload an image of the lesion', type=['jpg', 'jpeg', 'png'])

        submit_button = st.form_submit_button(label='Predict')
        if submit_button:
            if userFile is not None:
                img = Image.open(userFile)
                img.save('lesion.png')
                x = predict_test('lesion.png', age, local, sex)
                st.success(x)
            else:
                st.error('File not found')
