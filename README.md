NOTE : This is a personal mini project

This is a videogame chatbot for Genshin Impact character builds.
The data is in a custom pdf and will be updated on a regular basis,
as well as when new characters are added. This project has not been
hosted yet.

STEPS TO RUN
1) Create a virtual environment using python -m venv venvname
2) Install requirements using pip install -r requirements.txt
3) Run the app using streamlit run chatbot.py

IMP : This code runs on your cpu. To use your gpu instead,
      replace model_kwargs={'device':'cpu'} with 
              model_kwargs={'device':'cuda:0'} 
              (number based on which gpu you want to use)
              
Using CUDA require cudatoolkit installation and nsight (for vscode)

https://developer.nvidia.com/cuda-downloads make sure your gpu supports cuda

![Screenshot (19)](https://github.com/ungodlike/VideoGame-chatbot-for-Genshin-Builds/assets/115410346/7cc223ef-31d6-43e0-81ed-98095c40ab0e)
