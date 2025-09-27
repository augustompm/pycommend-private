![4179993](https://github.com/user-attachments/assets/9057edbb-6d9f-4aba-8bce-1ff120f7b970)
# Virtual Try-On Prototype with Flask, Twilio, and Hugging Face's Gradio API !

This repository contains the code for a virtual try-on application built using Flask, Twilio's WhatsApp API, and Gradio's app with IDM-VTON try-on model. Users can send images via WhatsApp to Twilio to try on garments virtually, and the results are sent back to them.

## Features
- Receive images of a person and a garment via WhatsApp.
- Use Gradio’s API to generate virtual try-on results.
- Return the result image to the user via WhatsApp.
- Uses Twilio Sandbox for WhatsApp for easy prototyping and testing.

## Technologies Used
- **Flask**: Backend server to handle requests and interact with Twilio and Gradio.
- **Twilio API**: To send and receive WhatsApp messages and media.
- **Gradio API**: For interacting with the virtual try-on Gradio app, which uses IDM-VTON ([Improving Diffusion Models for Authentic Virtual Try-on](https://huggingface.co/yisol/IDM-VTON)) model under the hood to generate accurate garment try-on results.
- **Ngrok**: For exposing the local server to the internet for WhatsApp interaction.

## Prerequisites
Before running this project, ensure you have the following:
- Twilio account with WhatsApp sandbox setup.
- Hugging Face account to use the Gradio API.
- Python 3.6+ installed on your machine.

## Twilio Setup
1. Create a [Twilio account](https://www.twilio.com/try-twilio).
2. Activate the [Twilio Sandbox for WhatsApp](https://www.twilio.com/console/sms/whatsapp/sandbox):
   - In the Twilio console, navigate to the **Messaging** section and select **Try it Out** under the **WhatsApp** sandbox.
   - Follow the instructions to join the sandbox by sending a WhatsApp message to the provided Twilio number.
3. Get your **Twilio Account SID** and **Auth Token** from your Twilio console:
   - Go to **Settings** in the Twilio console to find these credentials.
4. Take note of the **Twilio Sandbox number** for sending and receiving WhatsApp messages.

Once the sandbox is set up, you can start receiving and sending messages to the WhatsApp sandbox number for testing your virtual try-on application.

## Hugging Face Setup
1. Create a [Hugging Face account](https://huggingface.co/join).
2. Use the **Nymbo Virtual Try-On** model as API available on Hugging Face Spaces: [Nymbo Virtual Try-On](https://huggingface.co/spaces/Nymbo/Virtual-Try-On).The API is built using the IDM-VTON (Image-based Virtual Try-On Network) model, which takes images of both a person and a garment to generate realistic try-on results.

## Installation
Clone the repository:
```bash
git clone https://github.com/adarshb3/Virtual-Try-On-Application-using-Flask-Twilio-and-Gradio.git
cd Virtual-Try-On-Application-using-Flask-Twilio-and-Gradio
```
Install the required Python packages:
```
pip install -r requirements.txt
```
Set up your environment variables:
```
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
```

## Running the Application
Start the Flask server:
```
python app.py
```
## Ngrok Setup for Local Development
Since the Flask server runs locally, we use **ngrok** to expose the server to the internet so that Twilio's WhatsApp Sandbox can communicate with it.

1. Download and install ngrok from [here](https://ngrok.com/download).
2. Once installed, authenticate ngrok by running:
```
ngrok authtoken your_ngrok_auth_token
```
(Get your authentication token from the ngrok dashboard after signing up).
3. Start ngrok to expose your local Flask server:
```
.\ngrok http 8080
```
Copy the ngrok forwarding URL (e.g., https://e3e3-xxxx.ngrok-free.app) and set this as your Twilio webhook under the WhatsApp Sandbox Settings:
```
https://your-ngrok-url/webhook
```
Use WhatsApp to send a message or an image to the Twilio Sandbox number, and the application will respond with the virtual try-on result.

## Code Explanation
- **app.py**: The main Flask application that handles incoming WhatsApp messages, downloads the images from Twilio, and interacts with Gradio's virtual try-on model.
- **static**: This folder stores the images temporarily that are sent by users.
- **requirements.txt**: List of dependencies required for the project.

## Key Functions
- **webhook()**: Handles incoming POST requests from Twilio, manages the session, and interacts with the Gradio API.
- **send_to_gradio()**: Sends the person and garment images to Gradio's model for processing.
- **download_image()**: Downloads media files from Twilio's API and stores them locally.

## Usage
1. Send a photo of yourself via WhatsApp to the Twilio Sandbox number.
2. You'll receive a prompt asking you to send a photo of the garment.
3. After sending the garment photo, the system will process the images and send you the result with the garment virtually applied to your photo.
   
![WhatsApp Image 2024-10-19 at 2 32 06 PM](https://github.com/user-attachments/assets/2ecb7ccb-5637-4785-8a30-55170ae8bcb8)


## Future Enhancements
- Integrate more detailed error handling for the Gradio API.
- Add functionality to handle multiple garments or different garment categories (e.g. bottoms, shoes).
- Deploy on a production-grade WSGI server like Gunicorn instead of the Flask development server.

## License
This project is licensed under the MIT License.
   





