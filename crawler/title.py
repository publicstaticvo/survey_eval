import os
import re
import json
import time
import tqdm
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Union
from constants import *


class ArxivCrawlerEngine:
    def __init__(self, api_key: str = ""):
        self.base_url = "https://arxiv.org"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        if api_key: self.headers['Authorization'] = f"Bearer {api_key}"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def _download_paper(self, url: str, save_path: str = None) -> bool:
        """
        Download PDF or TeX source file of an arXiv paper.
        
        Args:
            arxiv_id: arXiv paper ID (e.g., '2301.12345' or '2301.12345v1')
            format: 'pdf' for PDF file or 'tex' for TeX source files
            save_path: Path to save the file. If None, saves to current directory
            
        Returns:
            bool: True if download successful, False otherwise
        """
        import os
        
        # Clean arxiv_id (remove 'arXiv:' prefix if present)
        retry = 3
        while retry > 0:        
            try:                
                # Download the file
                response = self.session.get(url, timeout=60, stream=True)
                response.raise_for_status()
                
                # Save to file
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = os.path.getsize(save_path) / (1024 * 1024)  # Size in MB
                print(f"Successfully downloaded {url} to: {save_path} ({file_size:.2f} MB)", end=" ")
                return True
                
            except requests.RequestException as e:
                retry -= 1
                print(f"{e}. Retry: {retry}")
                if retry > 0: time.sleep(4 ** (3 - retry))
            except Exception as e:
                retry -= 1
                print(f"Unknown error downloading paper {url}: {e}. Retry: {retry}")
                if retry > 0: time.sleep(1)
        
        return False
        
    def download_arxiv_paper(self, arxiv_id: str, format: str = 'pdf', save_path: str = None) -> bool:
        """
        Download PDF or TeX source file of an arXiv paper.
        
        Args:
            arxiv_id: arXiv paper ID (e.g., '2301.12345' or '2301.12345v1')
            format: 'pdf' for PDF file or 'tex' for TeX source files
            save_path: Path to save the file. If None, saves to current directory
            
        Returns:
            bool: True if download successful, False otherwise
        """
        import os
        
        # Clean arxiv_id (remove 'arXiv:' prefix if present)
        arxiv_id = arxiv_id.replace('arXiv:', '').strip()
        
        # Create save_path if not provided
        if format == 'pdf':
            save_path = os.path.join(save_path, f"{arxiv_id}.pdf")
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" 
        elif format == "tex":
            save_path = os.path.join(save_path, f"{arxiv_id}.tar.gz")    
            url = f"https://arxiv.org/src/{arxiv_id}"         
        else:
            print(f"Error: Invalid format '{format}'. Use 'pdf' or 'tex'.")
            return False

        return self._download_paper(url, save_path)

    def get_citations_semantic_scholar(self, arxiv_id: str, title: Optional[str] = None) -> Optional[Dict]:
        """Get citation count from Semantic Scholar API."""
        retry = 3
        while retry > 0:
            try:
                time.sleep(3 ** (3 - retry))
                # Try with arXiv ID first if available
                url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
                params = {'fields': 'title,citationCount,year,authors'}
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                return {
                    'citations': data.get('citationCount', -1),
                    'source': 'Semantic Scholar',
                    'year': data.get('year'),
                    'found': True
                }
                
            except Exception as e:
                retry -= 1
                if retry > 0: 
                    print(f"Error fetching citations for '{arxiv_id}': {e}, Retry: {retry}")
                
        # Fallback: search by title
        if title:
            retry = 3
            while retry > 0:
                print(f"Error fetching citations for '{arxiv_id}' by arxiv_id, Falling back to title.")
                try:
                    time.sleep(3 ** (3 - retry))
                    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                    params = {
                        'query': title,
                        'fields': 'title,citationCount,year',
                        'limit': 1
                    }
                    response = self.session.get(search_url, params=params, timeout=10)
                    
                    response.raise_for_status()
                    data = response.json()
                    if data.get('data') and len(data['data']) > 0:
                        paper = data['data'][0]
                        return {
                            'citations': paper.get('citationCount', -1),
                            'source': 'Semantic Scholar',
                            'year': paper.get('year'),
                            'found': True
                        }
                    else:
                        print(f"Title {title} Not found.")
                        return {'citations': -1, "found": False, "source": "Semantic Scholar"}
                    
                except Exception as e:
                    retry -= 1
                    print(f"Error fetching citations for '{title}': {e}, Retry: {retry}") 
        
        return {'citations': -1, 'source': 'Error', 'found': False}
        
    def crawl_month(self, category: str, year: int, month: int, max_papers: int = 2000) -> List[Dict[str, str]]:
        """Crawl papers for a specific category and month."""
        assert max_papers in [25, 50, 100, 250, 500, 1000, 2000], f"Invalid max_papers: {max_papers}"

        download_complete = False
        all_papers = []
        while not download_complete:
                
            # Be respectful to the server
            time.sleep(1)            
            if all_papers:
                url = f"{self.base_url}/list/{category}/{year}-{month:02d}?skip={len(all_papers)}&show={max_papers}"
            else:
                url = f"{self.base_url}/list/{category}/{year}-{month:02d}?show={max_papers}"

            retry = 3
            while retry > 0:
            
                try:
                    time.sleep(3 ** (3 - retry) / 2)
                    print(f"Crawling: {url}")
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    papers = []
                    
                    # Find all paper entries in the list
                    paper_items = soup.find_all('dt')
                    
                    for dt in paper_items:
                        # Get the paper ID from the dt tag
                        arxiv_link_tag = dt.find('a', title='Abstract')
                        if not arxiv_link_tag: continue                            
                        paper_id = arxiv_link_tag.text.strip().replace('arXiv:', '')

                        # Get the corresponding dd tag with paper details
                        dd = dt.find_next_sibling('dd')
                        if not dd: continue
                        
                        # Extract title
                        title_div = dd.find('div', class_='list-title')
                        if title_div:
                            title = title_div.text.replace('Title:', '').strip()
                            # Clean up title (remove extra whitespace)
                            title = re.sub(r'\s+', ' ', title)
                            
                            papers.append({
                                'title': title,
                                'arxiv_url': paper_id,
                                'category': category,
                            })
                    
                    all_papers.extend(papers)
                    download_complete = (len(papers) < max_papers)
                    break
                    
                except requests.RequestException as e:
                    retry -= 1
                    print(f"Error crawling {url}: {e}, Retry: {retry} ")

            else: 
                download_complete = True
                
        print(f"Found {len(all_papers)} papers in {category} {year}-{month:02d}")
        return all_papers
    
    def save_to_file(self, papers: Dict, filename: str):
        """Save results to a text file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f)
        
        print(f"\nResults saved to {filename}")


def download_title_and_save(
        engine: ArxivCrawlerEngine, 
        subjects_to_crawl: List[str] = ["cs", "econ", "eess", "math", "phy", "q-bio", "q-fin", "stat"],
        months: List[int] = list(range(1, 11)),
        year: int = 2025,
        save_fn: str = "paper2025.json"
    ):
    print("Starting arXiv Papers Crawler")
    print(f"Categories: {subjects_to_crawl}")
    print(f"Months: {months}")
    print("=" * 80)
    
    # Crawl all papers
    final_results = {s: [] for s in subjects_to_crawl}
    for subject in subjects_to_crawl:
        for category in CATEGORIES[subject]:
            for month in months:
                papers = engine.crawl_month(category, year, month)
                print(f"\nTotal papers crawled for {subject}: {len(papers)}")
                final_results[subject].extend(papers)
    
    # Display results
    print("\n" + "=" * 80)
    
    # Save to file
    engine.save_to_file(final_results, save_fn)
    return final_results


def get_citations(engine: ArxivCrawlerEngine, path: str, papers: Dict[str, Dict[str, str]]):
    papers_with_citation = {s: [] for s in papers}
    count = 0

    for s in papers:
        if os.path.exists(os.path.join(path, s)): continue
        for x in tqdm.tqdm(papers[s], desc=f"Fetching category {s}"):
            assert x['arxiv_url'].startswith("https://arxiv.org/abs/")
            arxiv_id = x['arxiv_url'][-10:]
            citation_details = engine.get_citations_semantic_scholar(arxiv_id, x['title'])
            if citation_details['citations'] >= 0:
                x['cited'] = citation_details['citations']
                papers_with_citation[s].append(x)
                count += 1
        print(f"Fetch category {s} finished. Get {len(papers_with_citation[s])}/{len(papers[s])} citation information.")
        os.makedirs(os.path.join(path, s), exist_ok=True)
        engine.save_to_file(papers_with_citation[s], os.path.join(path, s, "papers.json"))
    
    print(f"Get {count} citation results")


if __name__ == "__main__":    
    crawler = ArxivCrawlerEngine()
    download_title_and_save(crawler, ['cs'])
