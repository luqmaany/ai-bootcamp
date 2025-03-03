import os

from flask import Flask, request, jsonify,render_template, send_file
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from pathlib import Path
from openai import OpenAI
import json


load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    messages = [{"role": "user", "content": user_message}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the weather for"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_current_weather": get_current_weather,
        }

        messages.append(response_message)
       
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit")
            )

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        final_response_text = second_response.choices[0].message.content
    else:
        final_response_text = response_message.content

    speech_file_path = Path("static/speech.mp3")
    tts_response = client.audio.speech.create(
        model="tts-1",
        voice="sage",
        input=final_response_text
    )

    tts_response.stream_to_file(speech_file_path)

    return jsonify({
        "response": final_response_text,
        "audio_url": "/static/speech.mp3"
    })

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
   
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
   
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        with open(filepath, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        return jsonify({"transcript": transcript.text})

    else:
        return jsonify({"error": "Invalid file format"}), 400


def get_current_weather(location, unit="celsius"):
    if "melbourne" in location.lower():
        return json.dumps({"location": "Melbourne", "temperature": "20", "unit": unit})
    elif "sydney" in location.lower():
        return json.dumps({"location": "Sydney", "temperature": "25", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "san francisco", "temperature": "33", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown", "unit": unit})
   
   

   

if __name__ == '__main__':
    app.run(debug=True, port=5001)