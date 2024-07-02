import cv2
from pytube import YouTube

def download_youtube_video(video_url, output_path='.'):
    try:
        # Créer une instance YouTube avec l'URL de la vidéo
        yt = YouTube(video_url)

        # Obtenir le flux de la meilleure qualité
        stream = yt.streams.get_highest_resolution()

        # Télécharger la vidéo
        print(f"Téléchargement de '{yt.title}' en cours...")
        stream.download(output_path)
        print(f"'{yt.title}' a été téléchargé avec succès dans '{output_path}'")
    
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")

# URL de la vidéo YouTube que vous souhaitez télécharger
video_url = 'https://www.youtube.com/watch?v=c0FtiZUO9Kg&pp=ygUec2FmYXJ5IHdpbGQgYW5pbWFscyBwbGFuZSB2aWV3'

# Chemin de téléchargement (par défaut, le script téléchargera la vidéo dans le répertoire actuel)
output_path = 'videos/safari.mp4'

# Appeler la fonction pour télécharger la vidéo
download_youtube_video(video_url, output_path)
