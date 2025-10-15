import requests
from bs4 import BeautifulSoup
import argparse
import time
import tqdm
import json
import re
import os
from typing import List, Dict, Optional, Union
from urllib.parse import urljoin, quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import tarfile
import zipfile
import gzip, shutil

class ArxivCrawlerEngine:
    def __init__(self, api_key: str = ""):
        self.base_url = "https://arxiv.org"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        if api_key: self.headers['Authorization'] = f"Bearer {api_key}"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def download_paper(self, arxiv_id: str, format: str = 'pdf', save_path: str = None) -> bool:
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
                print(f"Successfully downloaded {arxiv_id}.{format} to: {save_path} ({file_size:.2f} MB)")
                return True
                
            except requests.RequestException as e:
                retry -= 1
                print(f"Error downloading paper {arxiv_id}: {e}. Retry: {retry}")
                if retry > 0: time.sleep(4 ** (3 - retry))
            except Exception as e:
                retry -= 1
                print(f"Unknown error downloading paper {arxiv_id}: {e}. Retry: {retry}")
                if retry > 0: time.sleep(1)
        
        return False

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
    
    def get_citations_google_scholar(self, title: str) -> Optional[Dict]:
        """Get citation count from Google Scholar (via web scraping)."""
        try:
            time.sleep(1)
            search_url = f"https://scholar.google.com/scholar?q={quote(title)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the first result
            result = soup.find('div', class_='gs_ri')
            if result:
                # Look for "Cited by" link
                cited_by = result.find('a', string=re.compile(r'Cited by \d+'))
                if cited_by:
                    match = re.search(r'Cited by (\d+)', cited_by.text)
                    if match:
                        return {
                            'citations': int(match.group(1)),
                            'source': 'Google Scholar',
                            'found': True
                        }
            
            return {'citations': -1, 'source': 'Google Scholar', 'found': False}
            
        except Exception as e:
            print(f"Error fetching from Google Scholar: {e}")
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
                    time.sleep(3 ** (3 - retry))
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
                        if not arxiv_link_tag:
                            continue
                            
                        paper_id = arxiv_link_tag.text.strip().replace('arXiv:', '')
                        # arxiv_url = urljoin(self.base_url, f"/abs/{paper_id}")
                        
                        # Get the corresponding dd tag with paper details
                        dd = dt.find_next_sibling('dd')
                        if not dd:
                            continue
                        
                        # Extract title
                        title_div = dd.find('div', class_='list-title')
                        if title_div:
                            title = title_div.text.replace('Title:', '').strip()
                            # Clean up title (remove extra whitespace)
                            title = re.sub(r'\s+', ' ', title)
                            
                            papers.append({
                                'title': title,
                                'arxiv_url': paper_id,  # arxiv_url,
                                'category': category,
                                'date': f"{year}-{month:02d}"
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
    
    def filter_survey_papers(self, papers: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter papers containing 'survey' or 'summary' in title."""
        pattern = re.compile(r"((survey|summary)(?= on)|(?<=a )(survey|summary))", re.IGNORECASE)
        filtered = []
        
        for paper in papers:
            title_lower = paper['title'].lower()
            if re.findall(pattern, title_lower):
                filtered.append(paper)
        
        return filtered

    def select_representative_papers(
            self, 
            papers: List[Dict[str, str]], 
            n_papers: int = 100,
            n_clusters: int = 20,
            citation_weight: float = 0.3
        ) -> List[Dict[str, str]]:
        """
        Select representative papers using clustering for diversity and citation-based ranking.
        
        Args:
            papers: List of papers with titles and citation counts
            n_papers: Number of papers to select (default: 100)
            n_clusters: Number of clusters for diversity (default: 20)
            citation_weight: Weight for citation count in selection (0-1, default: 0.3)
            
        Returns:
            List of selected representative papers
        """
        if len(papers) <= n_papers:
            print(f"Total papers ({len(papers)}) <= requested ({n_papers}), returning all papers")
            return papers
        elif len(papers) <= 2 * n_papers:
            print(f"Total papers ({len(papers)}) neer requested ({n_papers}), returning paper with most cited")
            papers = sorted(papers, key=lambda x: x['cited'], reverse=True)[:n_papers]
            return papers
        
        print(f"\n{'='*80}")
        print(f"Selecting {n_papers} representative papers from {len(papers)} total papers")
        print(f"Strategy: {n_clusters} clusters for diversity + citation-weighted ranking")
        print(f"{'='*80}\n")
        
        # Step 1: Extract titles and create TF-IDF vectors
        print("Step 1: Creating TF-IDF vectors from paper titles...")
        titles = [p['title'] for p in papers]
        
        # Use TF-IDF to convert titles to vectors
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            title_vectors = vectorizer.fit_transform(titles)
            print(f"  Created {title_vectors.shape[0]} vectors with {title_vectors.shape[1]} features")
        except ValueError as e:
            print(f"  Error in vectorization: {e}")
            print("  Falling back to simpler approach...")
            # Fallback: just sort by citations
            sorted_papers = sorted(papers, key=lambda x: x['cited'], reverse=True)
            return sorted_papers[:n_papers]
        
        # Step 2: Cluster papers for diversity
        print(f"\nStep 2: Clustering papers into {n_clusters} groups...")
        n_clusters_actual = min(n_clusters, len(papers))
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(title_vectors)
        
        # Add cluster labels to papers
        for i, paper in enumerate(papers):
            paper['cluster'] = int(cluster_labels[i])
        
        # Count papers per cluster
        cluster_counts = Counter(cluster_labels)
        print(f"  Cluster distribution: {dict(cluster_counts)}")
        
        # Step 3: Select papers from each cluster
        print(f"\nStep 3: Selecting papers from each cluster...")
        print(f"  Citation weight: {citation_weight:.2f}, Diversity weight: {1-citation_weight:.2f}")
        
        selected_papers = []
        
        # Calculate papers per cluster (proportional allocation)
        papers_per_cluster = {}
        for cluster_id in range(n_clusters_actual):
            proportion = cluster_counts[cluster_id] / len(papers)
            papers_per_cluster[cluster_id] = max(1, int(n_papers * proportion))
        
        # Adjust to ensure we get exactly n_papers
        total_allocated = sum(papers_per_cluster.values())
        if total_allocated < n_papers:
            # Add remaining papers to largest clusters
            remaining = n_papers - total_allocated
            largest_clusters = sorted(papers_per_cluster.keys(), key=lambda x: cluster_counts[x], reverse=True)
            for i in range(remaining):
                papers_per_cluster[largest_clusters[i % len(largest_clusters)]] += 1
        elif total_allocated > n_papers:
            # Remove papers from largest clusters
            excess = total_allocated - n_papers
            largest_clusters = sorted(papers_per_cluster.keys(), key=lambda x: papers_per_cluster[x], reverse=True)
            for i in range(excess):
                cluster_id = largest_clusters[i % len(largest_clusters)]
                if papers_per_cluster[cluster_id] > 1:
                    papers_per_cluster[cluster_id] -= 1
        
        print(f"  Papers per cluster: {papers_per_cluster}")
        
        # Select top papers from each cluster based on citations
        for cluster_id in range(n_clusters_actual):
            cluster_papers = [p for p in papers if p['cluster'] == cluster_id]
            n_select = papers_per_cluster[cluster_id]
            
            # Sort by citations within cluster
            cluster_papers_sorted = sorted(cluster_papers, key=lambda x: x.get('cited', 0), reverse=True)
            
            # Select top N from this cluster
            selected_from_cluster = cluster_papers_sorted[:n_select]
            selected_papers.extend(selected_from_cluster)
            
            if selected_from_cluster:
                print(f"  Cluster {cluster_id}: Selected {len(selected_from_cluster)} papers "
                      f"(top citations: {selected_from_cluster[0].get('cited', 0)})")
        
        # Step 4: Final ranking by combined score
        print(f"\nStep 4: Final ranking by combined score...")
        
        # Normalize citation counts to 0-1 range
        max_citations = max([p.get('cited', 0) for p in selected_papers]) or 1
        
        for paper in selected_papers:
            citation_score = paper.get('cited', 0) / max_citations
            diversity_score = 1.0  # Already selected for diversity via clustering
            paper['combined_score'] = (citation_weight * citation_score + 
                                      (1 - citation_weight) * diversity_score)
        
        # Sort by combined score
        selected_papers_sorted = sorted(selected_papers, key=lambda x: x['combined_score'], reverse=True)
        
        print(f"\nSelected {len(selected_papers_sorted)} papers!")
        print(f"  Citation range: {min([p.get('cited', 0) for p in selected_papers_sorted])} - "
              f"{max([p.get('cited', 0) for p in selected_papers_sorted])}")
        
        return selected_papers_sorted
    
    def save_to_file(self, papers: Dict, filename: str):
        """Save results to a text file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f)
        
        print(f"\nResults saved to {filename}")


def download_title_and_save(
        engine: ArxivCrawlerEngine, 
        subjects_to_crawl: List[str] = ["cs", "econ", "eess", "math", "phy", "q-bio", "q-fin", "stat"],
        months: List[int] = list(range(1, 10)),
        year: int = 2025,
        save_fn: str = "survey2025.json"
    ):
    # Define categories
    categories = {
        "cs": [
            "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY", 
            "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR", 
            "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", 
            "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC", 
            "cs.SD", "cs.SE", "cs.SI", "cs.SY"
        ], 
        "eess": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"],
        "econ": ["econ.EM", "econ.GN", "econ.TH"],
        "math": [
            "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO", "math.CT", 
            "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GR", 
            "math.GT", "math.HO", "math.IT", "math.KT", "math.LO", "math.MG", "math.MP", 
            "math.NA", "math.NT", "math.OA", "math.OC", "math.PR", "math.QA", "math.RA", 
            "math.RT", "math.SG", "math.SP", "math.ST"
        ],
        "physics": [
            "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE", "astro-ph.IM", 
            "astro-ph.SR", "cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci", 
            "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft", "cond-mat.stat-mech", 
            "cond-mat.str-el", "cond-mat.supr-con", "gr-qc", "hep-ex", "hep-lat", "hep-ph", 
            "hep-th", "math-ph", "nlin.AO", "nlin.CD", "nlin.CG", "nlin.PS", "nlin.SI", 
            "nucl-ex", "nucl-th", "physics.acc-ph", "physics.ao-ph", "physics.app-ph", 
            "physics.atm-clus", "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", 
            "physics.class-ph", "physics.comp-ph", "physics.data-an", "physics.ed-ph", 
            "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", 
            "physics.ins-det", "physics.med-ph", "physics.optics", "physics.pop-ph", 
            "physics.soc-ph", "physics.space-ph", "quant-ph"
        ],
        "q-bio": [
            "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT", "q-bio.PE", "q-bio.QM", 
            "q-bio.SC", "q-bio.TO"
        ],
        "q-fin": [
            "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR", 
            "q-fin.RM", "q-fin.ST", "q-fin.TR"
        ],
        "stat": ["stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH"]
    }
    
    print("Starting arXiv Survey Papers Crawler")
    print(f"Categories: {subjects_to_crawl}")
    print(f"Months: {months}")
    print("=" * 80)
    
    # Crawl all papers
    final_results = {s: [] for s in subjects_to_crawl}
    for subject in subjects_to_crawl:
        for category in categories[subject]:
            for month in months:
                # Phase 1: Crawl All Papers
                papers = engine.crawl_month(category, year, month)
                print(f"\nTotal papers crawled for {subject}: {len(papers)}")
                # Phase 2: Filter Survey Papers
                survey_papers = engine.filter_survey_papers(papers)
                print(f"Survey/Summary papers found: {len(survey_papers)}")
                # Phase 3: Get Cited By Number
                survey_papers_with_cited = []
                for paper in survey_papers:
                    citation_details = engine.get_citations_semantic_scholar(paper['arxiv_url'], paper['title'])
                    if citation_details['citations'] >= 0:
                        paper['cited'] = citation_details['citations']
                        survey_papers_with_cited.append(paper)
                print(f"Survey/Summary papers cited info found: {len(survey_papers_with_cited)}")
                # Save
                final_results[subject].extend(survey_papers_with_cited)
    
    # Display results
    print("\n" + "=" * 80)
    
    # Save to file
    engine.save_to_file(final_results, save_fn)
    return final_results


def get_survey_citations(engine: ArxivCrawlerEngine, papers: Dict[str, Dict[str, str]]):
    papers_with_citation = {s: [] for s in papers}
    count = 0

    for s in papers:
        if os.path.exists(f"P:\\AI4S\\survey_eval\\crawled_papers\\{s}"): continue
        for x in tqdm.tqdm(papers[s], desc=f"Fetching category {s}"):
            assert x['arxiv_url'].startswith("https://arxiv.org/abs/")
            arxiv_id = x['arxiv_url'][-10:]
            citation_details = engine.get_citations_semantic_scholar(arxiv_id, x['title'])
            if citation_details['citations'] >= 0:
                x['cited'] = citation_details['citations']
                papers_with_citation[s].append(x)
                count += 1
        print(f"Fetch category {s} finished. Get {len(papers_with_citation[s])}/{len(papers[s])} citation information.")
        os.makedirs(f"P:\\AI4S\\survey_eval\\crawled_papers\\{s}", exist_ok=True)
        engine.save_to_file(papers_with_citation[s], f"P:\\AI4S\\survey_eval\\crawled_papers\\{s}\\papers.json")
    
    print(f"Get {count} citation results")
    engine.save_to_file(papers_with_citation, "survey2025_.json")


def cluster_and_download_papers(
        engine: ArxivCrawlerEngine,
        subjects_to_crawl: str = ["cs", "econ", "eess", "math", "phy", "q-bio", "q-fin", "stat"],
        path: str = "P:\\AI4S\\survey_eval\\crawled_papers",
        num_paper_download: int = 100
    ):
    for s in subjects_to_crawl:
        with open(f"{path}\\{s}\\papers.json", "r+", encoding='utf-8') as f:
            papers = json.load(f)
        
        # Filter most-cited summaries
        selected_papers = engine.select_representative_papers(papers, num_paper_download)
        for paper in selected_papers:
            save_path = os.path.join(path, s)
            if paper['arxiv_url'].startswith("https://"): 
                paper['arxiv_url'] = paper['arxiv_url'][-10:]
            if os.path.exists(os.path.join(save_path, paper['arxiv_url'])): continue
            engine.download_paper(paper['arxiv_url'], save_path=save_path)
            status = engine.download_paper(paper['arxiv_url'], "tex", save_path=save_path)
            if status:
                # unzip paper
                path_to_unzip = os.path.join(save_path, f"{paper['arxiv_url']}.tar.gz")
                path_target = os.path.join(save_path, f"{paper['arxiv_url']}")
                try:
                    os.makedirs(path_target, exist_ok=True)
                    with tarfile.open(path_to_unzip, "r:gz") as f:
                        f.extractall(path_target)
                except:
                    try:
                        with zipfile.ZipFile(path_to_unzip, 'r') as f:
                            f.extractall(path_target)
                    except:
                        try:
                            main_tex = os.path.join(path_target, "main.tex")
                            with gzip.open(path_to_unzip, 'rb') as fin, open(main_tex, "wb") as fout:
                                shutil.copyfileobj(fin, fout)
                        except:
                            print(f"fail to unzip {paper['arxiv_url']}")
                finally:
                    os.remove(path_to_unzip)


def argparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--mode", type=str, default="get_papers", 
                        choices=["get_papers", "download_papers", "call_cited"])
    parser.add_argument("--months", type=str, required=True)
    parser.add_argument("--years", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":    
    # ["cs", "econ", "eess", "math", "phy", "q-bio", "q-fin", "stat"]
    # api_key=sk-6t2r2UAqWiwrk6hm7d499e1bFfE14399A3D7C947137891Eb
    crawler = ArxivCrawlerEngine()
    # download_title_and_save(crawler)
    # with open("survey2025.json", "r+", encoding='utf-8') as f:
    #     survey_papers = json.load(f)
    # get_survey_citations(crawler, survey_papers)
    # subjects = ['econ', 'q-bio', 'q-fin', 'stat', 'eess']
    # survey_2024 = download_title_and_save(crawler, subjects, list(range(1, 13)), 2024, "survey2024.json")
    # for s in subjects:
    #     with open(f"P:\\AI4S\\survey_eval\\crawled_papers\\{s}\\papersf.json", "r+", encoding='utf-8') as f:
    #         papers = json.load(f)
    #     papers.extend(survey_2024[s])
    #     with open(f"P:\\AI4S\\survey_eval\\crawled_papers\\{s}\\papers.json", "w+", encoding='utf-8') as f:
    #         json.dump(papers, f)
    cluster_and_download_papers(crawler, ['cs'])
