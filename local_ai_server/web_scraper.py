"""Web scraping module using Selenium."""
import logging
import time
from typing import List, Dict, Optional, Any, Union

# Import Selenium components
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# For automatic ChromeDriver management
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

class WebScraper:
    """A class to handle web scraping operations using Selenium."""
    
    def __init__(self, headless: bool = True, timeout: int = 30):
        """Initialize the web scraper.
        
        Args:
            headless: Whether to run Chrome in headless mode
            timeout: Default timeout for operations in seconds
        """
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        
    def __enter__(self):
        """Set up the Chrome driver when entering a context."""
        self._setup_driver()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Tear down the Chrome driver when exiting a context."""
        self.close()
        
    def _setup_driver(self):
        """Set up the Chrome driver with appropriate options."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")  # Use newer headless mode
            
        # Add common options to improve stability and security
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--allow-insecure-localhost")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Anti-bot detection measures
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
        
        try:
            # Use webdriver_manager to automatically download the correct ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Execute CDP command to mask WebDriver usage
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Chrome driver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            raise
    
    def close(self):
        """Close the Chrome driver."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
            except Exception as e:
                logger.warning(f"Error closing Chrome driver: {e}")
    
    def search(self, query: str, engine: str = "google", num_results: int = 5) -> List[Dict[str, str]]:
        """Perform a web search using the specified search engine.
        
        Args:
            query: The search query
            engine: Search engine to use ('google' or 'bing')
            num_results: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results, each containing
            'title', 'url', and 'snippet' keys
        """
        if not self.driver:
            self._setup_driver()
            
        results = []
        
        try:
            if engine.lower() == "google":
                results = self._search_google(query, num_results)
            elif engine.lower() == "bing":
                results = self._search_bing(query, num_results)
            else:
                raise ValueError(f"Unsupported search engine: {engine}. Use 'google' or 'bing'.")
                
            logger.info(f"Found {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during {engine} search: {e}")
            # Return any partial results we might have
            return results
    
    def _search_google(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Perform a Google search and extract results.
        
        Args:
            query: The search query
            num_results: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        results = []
        
        try:
            # Navigate to Google search
            self.driver.get(f"https://www.google.com/search?q={query}")
            
            # Try different selectors for search results to be more resilient to changes
            result_selectors = [
                "div.g", 
                "div[jscontroller][data-hveid]",
                "div.MjjYud",
                "div.Gx5Zad"
            ]
            
            for selector in result_selectors:
                try:
                    # Wait for search results with this selector
                    WebDriverWait(self.driver, self.timeout/2).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    search_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    if search_elements:
                        for element in search_elements[:num_results]:
                            result = self._extract_google_result(element)
                            if result and result.get('title') and result.get('url'):
                                results.append(result)
                                
                        # If we found enough results, break out of the selector loop
                        if len(results) >= num_results:
                            break
                except TimeoutException:
                    logger.warning(f"Timed out waiting for selector: {selector}")
                    continue
            
            # If no results found with our selectors, try a fallback approach
            if not results:
                logger.warning("Using fallback approach for Google results")
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                count = 0
                
                for link in all_links:
                    try:
                        url = link.get_attribute("href")
                        if (url and url.startswith("http") and 
                            "google.com" not in url and 
                            not url.endswith(".css") and 
                            not url.endswith(".js")):
                            
                            title = link.text.strip()
                            # Only add meaningful links
                            if title and len(title) > 5 and count < num_results:
                                results.append({
                                    "title": title,
                                    "url": url,
                                    "snippet": ""
                                })
                                count += 1
                    except Exception:
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Google search: {e}")
            return results
    
    def _extract_google_result(self, element) -> Dict[str, str]:
        """Extract information from a Google search result element.
        
        Args:
            element: The WebElement containing the search result
            
        Returns:
            Dictionary with title, url, and snippet
        """
        title = ""
        url = ""
        snippet = ""
        
        try:
            # Try to find title with various possible selectors
            for title_selector in ["h3", "h3.LC20lb", ".DKV0Md"]:
                try:
                    title_element = element.find_element(By.CSS_SELECTOR, title_selector)
                    title = title_element.text
                    if title:
                        break
                except (NoSuchElementException, WebDriverException):
                    continue
            
            # Try to find link with various possible selectors
            for link_selector in ["a", "a[href]", ".yuRUbf a"]:
                try:
                    link_element = element.find_element(By.CSS_SELECTOR, link_selector)
                    url = link_element.get_attribute("href")
                    if url and url.startswith("http"):
                        break
                except (NoSuchElementException, WebDriverException):
                    continue
            
            # Try to find snippet with various possible selectors
            for snippet_selector in ["div.VwiC3b", ".lyLwlc", ".lEBKkf", ".yXK7lf", ".MUxGbd"]:
                try:
                    snippet_element = element.find_element(By.CSS_SELECTOR, snippet_selector)
                    snippet = snippet_element.text
                    if snippet:
                        break
                except (NoSuchElementException, WebDriverException):
                    continue
                    
            return {
                "title": title,
                "url": url,
                "snippet": snippet
            }
        except Exception as e:
            logger.warning(f"Error extracting Google search result: {e}")
            return {}
    
    def _search_bing(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Perform a Bing search and extract results.
        
        Args:
            query: The search query
            num_results: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        results = []
        
        try:
            # Navigate to Bing search
            self.driver.get(f"https://www.bing.com/search?q={query}")
            
            # Try different selectors for search results
            result_selectors = [
                "li.b_algo", 
                "div.b_algo",
                ".b_results .b_algo"
            ]
            
            for selector in result_selectors:
                try:
                    # Wait for search results
                    WebDriverWait(self.driver, self.timeout/2).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    search_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    if search_elements:
                        for element in search_elements[:num_results]:
                            result = self._extract_bing_result(element)
                            if result and result.get('title') and result.get('url'):
                                results.append(result)
                                
                        # If we found enough results, break out of the selector loop
                        if len(results) >= num_results:
                            break
                except TimeoutException:
                    logger.warning(f"Timed out waiting for selector: {selector}")
                    continue
            
            # If no results found with our selectors, try a fallback approach
            if not results:
                logger.warning("Using fallback approach for Bing results")
                all_links = self.driver.find_elements(By.TAG_NAME, "a")
                count = 0
                
                for link in all_links:
                    try:
                        url = link.get_attribute("href")
                        if (url and url.startswith("http") and 
                            "bing.com" not in url and 
                            not url.endswith(".css") and 
                            not url.endswith(".js")):
                            
                            title = link.text.strip()
                            # Only add meaningful links
                            if title and len(title) > 5 and count < num_results:
                                results.append({
                                    "title": title,
                                    "url": url,
                                    "snippet": ""
                                })
                                count += 1
                    except Exception:
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Bing search: {e}")
            return results
    
    def _extract_bing_result(self, element) -> Dict[str, str]:
        """Extract information from a Bing search result element.
        
        Args:
            element: The WebElement containing the search result
            
        Returns:
            Dictionary with title, url, and snippet
        """
        title = ""
        url = ""
        snippet = ""
        
        try:
            # Try to find title with various possible selectors
            for title_selector in ["h2 a", "h2", ".b_title a"]:
                try:
                    title_element = element.find_element(By.CSS_SELECTOR, title_selector)
                    title = title_element.text
                    
                    # If this selector is an anchor, also get the URL
                    if title_selector.endswith("a"):
                        url = title_element.get_attribute("href")
                        
                    if title:
                        break
                except (NoSuchElementException, WebDriverException):
                    continue
            
            # Try to find URL (if not already found)
            if not url:
                for link_selector in ["a", "cite"]:
                    try:
                        link_element = element.find_element(By.CSS_SELECTOR, link_selector)
                        if link_selector == "a":
                            url = link_element.get_attribute("href")
                        else:
                            url = link_element.text
                        if url and url.startswith("http"):
                            break
                    except (NoSuchElementException, WebDriverException):
                        continue
            
            # Try to find snippet
            for snippet_selector in ["div.b_caption p", "p", ".b_snippet"]:
                try:
                    snippet_element = element.find_element(By.CSS_SELECTOR, snippet_selector)
                    snippet = snippet_element.text
                    if snippet:
                        break
                except (NoSuchElementException, WebDriverException):
                    continue
                    
            return {
                "title": title,
                "url": url,
                "snippet": snippet
            }
        except Exception as e:
            logger.warning(f"Error extracting Bing search result: {e}")
            return {}
    
    def fetch_content(self, url: str) -> Optional[str]:
        """Fetch and extract the main content from a URL.
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            The extracted content as a string, or None if content couldn't be extracted
        """
        if not self.driver:
            self._setup_driver()
            
        try:
            # Navigate to the URL
            self.driver.get(url)
            
            # Wait for page to load
            try:
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Find the body element
                body_element = self.driver.find_element(By.TAG_NAME, "body")
                
                # Try to extract main content from common content containers
                main_content = ""
                for selector in ["article", "main", ".content", "#content", 
                                ".main", "#main", ".article-content", 
                                "#article-content", ".post-content"]:
                    try:
                        content_element = body_element.find_element(By.CSS_SELECTOR, selector)
                        main_content = content_element.text
                        break
                    except (NoSuchElementException, WebDriverException):
                        continue
                        
                # If no main content container found, use the body text
                if not main_content:
                    main_content = body_element.text
                    
                # Clean up the content (remove excess whitespace)
                main_content = " ".join(main_content.split())
                
                # Limit content length to prevent extremely large texts
                if len(main_content) > 10000:
                    logger.info(f"Truncating content from {len(main_content)} to 10000 characters")
                    main_content = main_content[:10000]
                    
                return main_content
                
            except TimeoutException:
                logger.warning(f"Timed out waiting for page to load: {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return None

def search_and_scrape(query: str, engine: str = "google", num_results: int = 5, 
                      fetch_content: bool = False, timeout: int = 30,
                      headless: bool = True) -> Dict[str, Any]:
    """Search the web and optionally scrape content from result pages.
    
    Args:
        query: Search query
        engine: Search engine to use ('google' or 'bing')
        num_results: Maximum number of results to return
        fetch_content: Whether to fetch the full content from each result page
        timeout: Maximum wait time in seconds
        headless: Whether to run the browser in headless mode
    
    Returns:
        Dictionary with search results and metadata
    """
    with WebScraper(headless=headless, timeout=timeout) as scraper:
        # Perform the search
        search_results = scraper.search(query, engine, num_results)
        
        # Fetch content if requested
        if fetch_content and search_results:
            for result in search_results:
                try:
                    if result.get('url'):
                        content = scraper.fetch_content(result['url'])
                        if content:
                            result['content'] = content
                        else:
                            result['content'] = "Failed to fetch content"
                except Exception as e:
                    logger.error(f"Error fetching content for {result.get('url')}: {e}")
                    result['content'] = "Error fetching content"
    
    # Return the full result object
    return {
        "query": query,
        "engine": engine,
        "results": search_results,
        "count": len(search_results),
        "timestamp": time.time()
    }
