import os
import time
from io import BytesIO

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import ffmpeg
import requests
from dotenv import load_dotenv

app = Flask(__name__)

ALLOWED_ORIGIN = os.environ.get('FRONTEND_ORIGIN', 'http://localhost:3000')
load_dotenv()

cors = CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

OPENAI_API_URL = 'https://api.openai.com/v1/audio/transcriptions'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PORT = int(os.environ.get('PORT', 3001))


class TranscriptionError(Exception):
    """Custom exception for transcription errors."""
    pass


def time_to_seconds(time_str):
    """Convert a MM:SS format string to total seconds."""
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(time_str.split(":"))))


def remove_file(filename):
    """Helper function to safely remove a file."""
    try:
        os.remove(filename)
    except OSError:
        pass


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio segment.
    
    Requires audio file, start time, and end time as input. Returns transcribed text.
    """
    try:
        audio_file = request.files.get('file')
        start_time = request.form.get('startTime')
        end_time = request.form.get('endTime')

        if not audio_file:
            raise TranscriptionError('Audio file is required.')
        
        if not start_time or not end_time:
            raise TranscriptionError('Start and end times are required.')
        
        start_seconds = time_to_seconds(start_time)
        end_seconds = time_to_seconds(end_time)
        time_duration = end_seconds - start_seconds

        if time_duration < 0:
            raise TranscriptionError('Start time cannot be greater than end time.')

        temp_filename = f"temp-{int(time.time())}.mp3"
        output_filename = f"output-{int(time.time())}.mp3"

        with open(temp_filename, 'wb') as temp_file:
            temp_file.write(audio_file.read())

        ffmpeg.input(temp_filename).output(output_filename, ss=start_seconds, t=time_duration).run()

        with open(output_filename, 'rb') as output_file:
            trimmed_audio = output_file.read()

        files = {
            'file': ('audio.mp3', trimmed_audio, audio_file.mimetype),
        }
        
        data = {'model': 'whisper-1', 'response_format': 'json'}
        headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}

        response = requests.post(OPENAI_API_URL, files=files, data=data, headers=headers)
        response.raise_for_status()

        transcription = response.json().get('text')

        # Cleanup temporary files
        remove_file(temp_filename)
        remove_file(output_filename)

        return jsonify({'transcription': transcription})

    except TranscriptionError as te:
        return jsonify({'message': str(te)}), 400
    except requests.HTTPError as http_err:
        return jsonify({'message': http_err.response.json().get('error', {}).get('message', 'Error transcribing audio')}), http_err.response.status_code
    except Exception as err:
        print(err)
        return jsonify({'message': 'Error transcribing audio'}), 500


@app.route('/api/hello', methods=['GET'])
def hello():
    """Returns a simple greeting to test the server."""
    return jsonify({'greet': 'Hello.'})


if __name__ == "__main__":
    app.run(port=PORT)
