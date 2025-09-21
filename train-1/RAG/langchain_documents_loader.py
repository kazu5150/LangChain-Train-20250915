# Web Document Loader for RAG Implementation
from langchain_community.document_loaders import WebBaseLoader
import bs4


def load_web_content_simple(urls):
    """
    Simple and fast text extraction using WebBaseLoader with BeautifulSoup
    """
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs={"parse_only": bs4.SoupStrainer(
            ["article", "main", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
        )}
    )
    docs = loader.load()
    return docs


def load_web_content_with_custom_parser(urls, css_selector=None):
    """
    Custom parsing using WebBaseLoader with specific CSS selectors
    """
    if css_selector:
        bs_kwargs = {"parse_only": bs4.SoupStrainer(class_=css_selector)}
    else:
        bs_kwargs = {"parse_only": bs4.SoupStrainer(
            ["article", "main", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
        )}

    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=bs_kwargs
    )
    docs = loader.load()
    return docs


async def load_web_content_async(urls):
    """
    Async loading for better performance with multiple URLs
    """
    loader = WebBaseLoader(web_paths=urls)
    docs = await loader.aload()
    return docs


# Example usage
if __name__ == "__main__":
    # Example URLs to load
    web_urls = [
        "https://python.langchain.com/docs/how_to/document_loader_web/",
        "https://python.langchain.com/docs/concepts/document_loaders/"
    ]

    # Simple web loading
    print("Loading web content with WebBaseLoader...")
    simple_docs = load_web_content_simple(web_urls)
    print(f"Loaded {len(simple_docs)} documents")

    # Custom parsing with CSS selector
    print("\nLoading web content with custom parser...")
    custom_docs = load_web_content_with_custom_parser(web_urls)
    print(f"Loaded {len(custom_docs)} documents with custom parsing")

    # Print sample content
    if simple_docs:
        print(f"\nSample content (first 200 chars): {simple_docs[0].page_content[:200]}...")
