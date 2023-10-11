'''voice processing inference'''
import os
import pathlib
import requests
import iso639
import whisper
import yt_dlp as youtube_dl
import json
import base64
cnvrg_workdir = os.getcwd()
# model = whisper.load_model(os.environ['MODEL_SIZE'])
model = whisper.load_model('small.en')

def yt_vid_to_audio(url):
    '''
    download the audio file either from youtube or s3

    Parameters
    ----------
    url : URL to the youtube video to be converted and used as an audio input

    Returns
    -------
    out_file : converted youtube vid to audio file

    '''
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': cnvrg_workdir+'/audio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([str(url)])

    out_file = os.path.join(cnvrg_workdir, "audio.wav")
    # result of success
    #print(yt_.title + " has been successfully downloaded.")
    
    return out_file

def download_test_file(url_):
    """
    Downloads the model files if they are not already present or
    pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    if not os.path.exists(current_dir + f'/{url_}') and not os.path.exists('/input/cnn/' + url_):
        print(f'Downloading file: {url_}')
        response = requests.get(url_)
        file = url_.split("/")[-1]
        path_ = os.path.join(current_dir, file)
        with open(path_, "wb") as file_:
            file_.write(response.content)

def predict(audio_file):
    '''

    Parameters
    ----------
    audio_file : audio file containing speech

    Returns
    -------
    text extracted from the sudio file
    '''
### SCALE TEST DATA ###
    print('Running Stand Alone Endpoint')
    script_dir = pathlib.Path(__file__).parent.resolve()

    audio_f = str(audio_file['file'])
    link = str(audio_file['link'])
    lang = str(audio_file['language'])
    #model_size = str(audio_file['model_size'])

    if len(audio_f) == 0:
        if 'www.youtube.com' in link or 'youtu.be' in link:
            audio_file = yt_vid_to_audio(link)
            name = 'audio.wav'
            print(name)
        else:
            download_test_file(audio_f)
            name = audio_f.rsplit("/", maxsplit=1)[-1]
    else:
        decoded_audio = base64.b64decode(audio_file['file'])
    
        # Save the decoded audio to a file
        name = 'audio.wav'
        with open(os.path.join(script_dir,name), 'wb') as f:
            f.write(decoded_audio)

    file_name = os.path.join(script_dir,name)
    print(file_name)
    dic = {}
    # print('model_size:', os.environ['MODEL_SIZE'])
    result = model.transcribe(file_name, task='transcribe', fp16=False,
                              language=iso639.to_iso639_1(lang))
    # print the recognized text
    dic = result['text']
    return dic

#with open('audio.wav', 'rb') as f:
#    content = f.read()
#    encoded = base64.b64encode(content).decode("utf-8")

print(predict({'file':'',
              'link':'https://youtu.be/jZOywn1qArI?si=iWnyY5asQhQdCcjm',
               'language':'english'}))
