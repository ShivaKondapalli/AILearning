from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

search_queries = ["Badminton racket.", "Squash racket.", "Tennis racket"]


def download(query):

    argumnets= {"keywords": query, "format": "jpg", "limit": 50, "print_urls": True,
                "size": "medium", "aspect_ratio": "panoramic"
                }
    try:
        response.download(argumnets)

    except FileNotFoundError:
        arguments = {"keywords": query, "format": "jpg", "limit": 50, "print_urls": True,
                     "size": "medium"
                     }

        try:
            response.download(arguments)
        except:
            pass
for query in search_queries:
    download(query)
    print()

