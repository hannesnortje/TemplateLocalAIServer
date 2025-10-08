# Web Scraping Capabilities

LocalAIServer includes a web scraping feature that allows you to search the web and optionally save the results to the vector database for use with RAG.

## Requirements

The web scraping feature requires:
- Chrome or Chromium browser installed on your system
- Selenium and webdriver-manager Python packages (installed automatically)

## EU Regulatory Compliance

In accordance with EU regulations (GDPR, ePrivacy Directive), the web scraping feature requires explicit consent:

1. You must provide the `explicit_consent=true` parameter for any web scraping request
2. You should provide a `privacy_policy_url` parameter pointing to your privacy policy
3. The system records a timestamp with each scraping operation for compliance documentation
4. Without explicit consent, the scraping operation will not proceed

## Using the Web Scraping API

### Basic Usage

To scrape web content based on a search query:

