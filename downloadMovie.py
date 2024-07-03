import argparse
from pytube import YouTube

class YouTubeDownloader:
    def __init__(self, video_url, output_path='.'):
        self.video_url = video_url
        self.output_path = output_path

    def download_video(self):
        try:
            # Créer une instance YouTube avec l'URL de la vidéo
            yt = YouTube(self.video_url)

            # Obtenir le flux de la meilleure qualité
            stream = yt.streams.get_highest_resolution()

            # Télécharger la vidéo
            print(f"Téléchargement de '{yt.title}' en cours...")
            stream.download(self.output_path)
            print(f"'{yt.title}' a été téléchargé avec succès dans '{self.output_path}'")
        
        except Exception as e:
            print(f"Erreur lors du téléchargement : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Télécharger une vidéo YouTube.')
    parser.add_argument('video_url', type=str, help='URL de la vidéo YouTube à télécharger')
    parser.add_argument('--output_path', type=str, default='/videos/elephant.mp4')

    args = parser.parse_args()

    # Créer une instance de la classe YouTubeDownloader
    downloader = YouTubeDownloader(args.video_url, args.output_path)

    # Appeler la méthode pour télécharger la vidéo
    downloader.download_video()
