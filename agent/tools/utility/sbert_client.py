import numpy as np
import requests
import logging
import time


class SentenceTransformerClient:
    def __init__(self, api_url: str, batch_size: int = 32):
        self.api_url = api_url
        self.batch_size = batch_size
    
    def embed(self, documents: list[str], verbose: bool = False):
        """
        Embed documents using the deployed embedding API
        """
        embeddings = []
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            if verbose:
                logging.info(f"Processing batch {i//self.batch_size + 1}/{(len(documents)-1)//self.batch_size + 1}")
            
            retry = 5
            while retry:
                try:
                    response = requests.post(
                        f"{self.api_url}/encode",
                        json={"texts": batch, "normalize": True},
                        headers={"Content-Type": "application/json"},
                        timeout=60
                    )
                    response.raise_for_status()
                    batch_embeddings = response.json()
                    embeddings.extend(batch_embeddings['embeddings'])
                    break
                except Exception as e:
                    logging.error(f"Error embedding batch: {e}")
                    time.sleep(5)
                    retry -= 1
        
        return np.array(embeddings)