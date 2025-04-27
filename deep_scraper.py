#!/usr/bin/env python3
"""
SRM University-AP Deep Web Scraper

This module implements an asynchronous web scraper for the SRM University-AP website.
It performs deep scraping of all available URLs without limits, extracting text from
both HTML pages and PDF files, skipping unwanted file types.
The scraped data is saved in a pickle file for later processing.
"""

import asyncio
import logging
import os
import pickle
import re
import time
import sys
from typing import Dict, List, Set, Tuple, Optional
from urllib.parse import urljoin, urlparse

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/deep_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SRMAPDeepScraper")

# Check for required packages and install if missing
required_packages = ["aiohttp", "beautifulsoup4", "pymupdf", "tqdm"]
missing_packages = []

for package in required_packages:
    try:
        __import__(package.replace("-", "_").split(">=")[0].split("==")[0])
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        logger.info("Package installation completed")
    except Exception as e:
        logger.error(f"Failed to install packages: {str(e)}")
        logger.error("Please install the following packages manually: " + ", ".join(missing_packages))
        sys.exit(1)

# Now import the required packages
import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm
try:
    import fitz  # PyMuPDF
except ImportError:
    import pymupdf as fitz

class AsyncDeepWebScraper:
    """Asynchronous deep web scraper for SRM University-AP website."""
    
    def __init__(
        self, 
        start_url: str = "https://srmap.edu.in/", 
        max_concurrent_requests: int = 15,
        timeout: int = 60,
        max_retries: int = 5,
        retry_delay: int = 2,
        output_file: str = "srmap_data_deep.pkl",
        save_interval: int = 50  # Save after every 50 URLs
    ):
        """
        Initialize the scraper with configuration parameters.
        
        Args:
            start_url: The starting URL for the scraper
            max_concurrent_requests: Maximum number of concurrent requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            output_file: Path to save the scraped data
            save_interval: Number of URLs to process before saving intermediate results
        """
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.output_file = output_file
        self.save_interval = save_interval
        
        # Track URLs
        self.visited_urls: Set[str] = set()
        self.urls_to_visit: Set[str] = {start_url}
        self.failed_urls: Dict[str, str] = {}
        self.url_depths: Dict[str, int] = {start_url: 0}  # Track depth of each URL
        self.max_depth_reached = 0
        
        # Store scraped data
        self.scraped_data: Dict[str, str] = {}
        
        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Progress tracking
        self.start_time = time.time()
        self.last_save_time = self.start_time
        
        # File type patterns
        self.pdf_pattern = re.compile(r'\.pdf$', re.IGNORECASE)
        self.excluded_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in [
                r'\.(jpg|jpeg|png|gif|bmp|svg|webp|ico|tif|tiff)$',  # Images
                r'\.(zip|rar|tar|gz|7z)$',  # Archives
                r'\.(mp4|avi|mov|wmv|flv|mkv)$',  # Videos
                r'\.(mp3|wav|ogg|flac|aac)$',  # Audio
                r'\.(doc|docx|ppt|pptx|xls|xlsx|csv)$',  # Office files
                r'\.(js|css|map)$',  # Web assets
                r'(facebook|twitter|instagram|linkedin|youtube)',  # Social media
                r'mailto:',  # Email links
                r'tel:',  # Phone links
                r'#',  # Anchors
                r'javascript:',  # JavaScript links
            ]
        ]
        
        
        self.priority_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in [
                r'/about',
                r'/academics',
                r'/admissions',
                r'/research',
                r'/faculty',
                r'/campus',
                r'/courses',
                r'/programs',
                r'/departments',
                r'/schools',
                r'/placements',
                r'/scholarships',
                r'/international',
                r'/contact',
                r'/news',
                r'/events',
            ]
        ]
    
    async def fetch_url(self, url: str, session: aiohttp.ClientSession) -> Tuple[str, str, str]:
        """
        Fetch and process a URL with retry logic.
        
        Args:
            url: URL to fetch
            session: aiohttp ClientSession
            
        Returns:
            Tuple of (url, content_type, text_content)
        """
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                    
                    async with session.get(url, timeout=self.timeout, headers=headers, ssl=False) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                            return url, "", ""
                        
                        content_type = response.headers.get('Content-Type', '').lower()
                        
                        
                        if 'application/pdf' in content_type or self.pdf_pattern.search(url):
                            try:
                                pdf_content = await response.read()
                                text = self.extract_text_from_pdf(pdf_content)
                                return url, 'pdf', text
                            except Exception as e:
                                logger.error(f"Error processing PDF {url}: {str(e)}")
                                return url, "", ""
                        
                        
                        elif 'text/html' in content_type:
                            try:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                
                              
                                text = self.extract_text_from_html(soup)
                                self.extract_links(soup, url)
                                
                                return url, 'html', text
                            except Exception as e:
                                logger.error(f"Error processing HTML {url}: {str(e)}")
                                return url, "", ""
                        
                        
                        else:
                            logger.info(f"Skipping unsupported content type: {content_type} for {url}")
                            return url, "", ""
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url} (attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
            except aiohttp.ClientConnectorError as e:
                logger.warning(f"Connection error for {url}: {str(e)} (attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
            except aiohttp.ClientError as e:
                logger.warning(f"Client error for {url}: {str(e)} (attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)} (attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay)
        
        
        self.failed_urls[url] = f"Max retries ({self.max_retries}) exceeded"
        return url, "", ""
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content.
        
        Args:
            pdf_content: Binary PDF content
            
        Returns:
            Extracted text from the PDF
        """
        try:
            with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_text_from_html(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text from HTML content.
        
        Args:
            soup: BeautifulSoup object of the HTML
            
        Returns:
            Extracted text from the HTML
        """
        try:
            
            for element in soup(["script", "style", "header", "footer", "nav"]):
                element.decompose()
            
            
            text = soup.get_text(separator=' ', strip=True)
            
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            return ""
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> None:
        """
        Extract and filter links from HTML content.
        
        Args:
            soup: BeautifulSoup object of the HTML
            base_url: Base URL for resolving relative links
        """
        try:
            current_depth = self.url_depths.get(base_url, 0)
            next_depth = current_depth + 1
            
            
            if current_depth > self.max_depth_reached:
                self.max_depth_reached = current_depth
                logger.info(f"Reached depth level: {self.max_depth_reached}")
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                
                if not href or href.startswith('javascript:'):
                    continue
                
                try:
                    full_url = urljoin(base_url, href)
                    
                    
                    if full_url in self.visited_urls or full_url in self.urls_to_visit:
                        continue
                    
                    
                    parsed_url = urlparse(full_url)
                    
                    
                    if parsed_url.netloc != self.base_domain:
                        continue
                    
                    
                    if any(pattern.search(full_url) for pattern in self.excluded_patterns):
                        continue
                    
                    
                    if self.pdf_pattern.search(full_url) and 'filesize' in link.attrs and int(link['filesize']) > 10000000:
                        logger.info(f"Skipping large PDF: {full_url}")
                        continue
                    
                    
                    self.urls_to_visit.add(full_url)
                    self.url_depths[full_url] = next_depth
                    
                except Exception as e:
                    logger.warning(f"Error processing link {href}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
    
    def prioritize_urls(self) -> List[str]:
        """
        Prioritize URLs for processing.
        
        Returns:
            List of URLs sorted by priority
        """
        
        urls_by_depth = {}
        for url in self.urls_to_visit:
            depth = self.url_depths.get(url, 0)
            if depth not in urls_by_depth:
                urls_by_depth[depth] = []
            urls_by_depth[depth].append(url)
        
        
        for depth, urls in urls_by_depth.items():
            # Prioritize URLs matching priority patterns
            priority_urls = []
            normal_urls = []
            
            for url in urls:
                if any(pattern.search(url) for pattern in self.priority_patterns):
                    priority_urls.append(url)
                else:
                    normal_urls.append(url)
            
            urls_by_depth[depth] = priority_urls + normal_urls
        
        
        prioritized_urls = []
        for depth in sorted(urls_by_depth.keys()):
            prioritized_urls.extend(urls_by_depth[depth])
        
        return prioritized_urls
    
    async def process_url(self, url: str, session: aiohttp.ClientSession) -> None:
        """
        Process a single URL.
        
        Args:
            url: URL to process
            session: aiohttp ClientSession
        """
        url, content_type, text = await self.fetch_url(url, session)
        
        if text:
            self.scraped_data[url] = text
            logger.info(f"Successfully scraped {url} ({content_type}) at depth {self.url_depths.get(url, 0)}")
        
        self.visited_urls.add(url)
        if url in self.urls_to_visit:
            self.urls_to_visit.remove(url)
    
    def save_data(self, output_file=None) -> None:
        """
        Save the scraped data to a pickle file.
        
        Args:
            output_file: Optional alternative output file path
        """
        file_to_save = output_file or self.output_file
        try:
            with open(file_to_save, 'wb') as f:
                pickle.dump(self.scraped_data, f)
            logger.info(f"Saved scraped data to {file_to_save}")
            self.last_save_time = time.time()
        except Exception as e:
            logger.error(f"Error saving data to {file_to_save}: {str(e)}")
            
            try:
                backup_file = f"backup_{int(time.time())}.pkl"
                with open(backup_file, 'wb') as f:
                    pickle.dump(self.scraped_data, f)
                logger.info(f"Saved backup data to {backup_file}")
                self.last_save_time = time.time()
            except Exception as e2:
                logger.error(f"Error saving backup data: {str(e2)}")
    
    def print_statistics(self) -> None:
        """Print scraping statistics."""
        elapsed_time = time.time() - self.start_time
        urls_per_second = len(self.visited_urls) / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"=== Scraping Statistics ===")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"URLs processed: {len(self.visited_urls)}")
        logger.info(f"URLs in queue: {len(self.urls_to_visit)}")
        logger.info(f"Failed URLs: {len(self.failed_urls)}")
        logger.info(f"Scraped data: {len(self.scraped_data)} URLs")
        logger.info(f"Processing rate: {urls_per_second:.2f} URLs/second")
        logger.info(f"Max depth reached: {self.max_depth_reached}")
        logger.info(f"=========================")
    
    async def run(self, max_urls: Optional[int] = None, max_depth: Optional[int] = None) -> Dict[str, str]:
        """
        Run the scraper.
        
        Args:
            max_urls: Maximum number of URLs to scrape (None for unlimited)
            max_depth: Maximum depth to scrape (None for unlimited)
            
        Returns:
            Dictionary of scraped data {url: text_content}
        """
        self.start_time = time.time()
        self.last_save_time = self.start_time
        logger.info(f"Starting deep scraper with {self.max_concurrent_requests} concurrent requests")
        logger.info(f"Max URLs: {'Unlimited' if max_urls is None else max_urls}")
        logger.info(f"Max depth: {'Unlimited' if max_depth is None else max_depth}")
        
        
        connector = aiohttp.TCPConnector(ssl=False, limit=self.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.timeout, sock_read=self.timeout)
        
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                while self.urls_to_visit and (max_urls is None or len(self.visited_urls) < max_urls):
                    
                    if max_depth is not None and self.max_depth_reached >= max_depth:
                        remaining_urls = [url for url in self.urls_to_visit if self.url_depths.get(url, 0) <= max_depth]
                        if not remaining_urls:
                            logger.info(f"Reached maximum depth of {max_depth}. Stopping.")
                            break
                    
                    
                    prioritized_urls = self.prioritize_urls()
                    
                    
                    batch_size = min(len(prioritized_urls), self.max_concurrent_requests)
                    batch_urls = prioritized_urls[:batch_size]
                    
                    
                    tasks = [self.process_url(url, session) for url in batch_urls]
                    await asyncio.gather(*tasks)
                    
                    
                    if len(self.visited_urls) % 10 == 0:
                        self.print_statistics()
                    
                    
                    if len(self.visited_urls) % self.save_interval == 0 or (time.time() - self.last_save_time) > 300:  
                        self.save_data(f"{self.output_file}.partial")
                        logger.info(f"Saved intermediate data to {self.output_file}.partial")
        except KeyboardInterrupt:
            logger.info("Scraping interrupted by user")
            
            if self.scraped_data:
                self.save_data(f"{self.output_file}.interrupted")
                logger.info(f"Saved partial data to {self.output_file}.interrupted")
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            
            if self.scraped_data:
                self.save_data(f"{self.output_file}.error")
                logger.info(f"Saved partial data to {self.output_file}.error")
        
        
        elapsed_time = time.time() - self.start_time
        logger.info(f"Scraping completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {len(self.visited_urls)} URLs")
        logger.info(f"Successfully scraped {len(self.scraped_data)} URLs")
        logger.info(f"Failed to scrape {len(self.failed_urls)} URLs")
        logger.info(f"Maximum depth reached: {self.max_depth_reached}")
        
        
        self.save_data()
        
        return self.scraped_data

async def main():
    """Main function to run the scraper."""
    try:
        
        scraper = AsyncDeepWebScraper(
            start_url="https://srmap.edu.in/",
            max_concurrent_requests=15,
            timeout=60,
            max_retries=5,
            retry_delay=2,
            output_file="srmap_data_deep.pkl",
            save_interval=50
        )
        
        
        priority_urls = {
            "https://srmap.edu.in/about-university/",
            "https://srmap.edu.in/academics/",
            "https://srmap.edu.in/admissions/",
            "https://srmap.edu.in/research/",
            "https://srmap.edu.in/campus-life/",
            "https://srmap.edu.in/faculty/",
            "https://srmap.edu.in/placements/",
            "https://srmap.edu.in/scholarships/",
            "https://srmap.edu.in/international-collaborations/",
            "https://srmap.edu.in/contact-us/",
            "https://srmap.edu.in/programs/",
            "https://srmap.edu.in/schools/",
            "https://srmap.edu.in/departments/",
            "https://srmap.edu.in/news/",
            "https://srmap.edu.in/events/",
        }
       
        for url in priority_urls:
            if url not in scraper.urls_to_visit:
                scraper.urls_to_visit.add(url)
                scraper.url_depths[url] = 1 
        
        await scraper.run(max_urls=None, max_depth=5)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scraper stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)
