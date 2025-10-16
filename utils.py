import os
import tiktoken
import jinja2


def load_tokenizer():
    """
    Load the tokenizer from cache.
    """
    # Set tiktoken cache directory to local repo
    TIKTOKEN_CACHE_DIR = os.path.join(
        os.path.dirname(__file__),
        ".tiktoken_cache",
    )
    os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR

    # Pre-load tokenizer from cache to ensure it's available
    try:
        # Load o200k_base encoding from cache
        tiktoken.get_encoding("o200k_base")
    except Exception as e:
        print(f"⚠ Warning: Could not load tokenizer from cache: {e}")


def load_hashes():
    """
    Load the document hashes from cache.
    """
    DOCUMENT_HASH_FILE = os.path.join(
        os.path.dirname(__file__),
        ".document_hashes.json",
    )
    os.environ["DOCUMENT_HASH_FILE"] = DOCUMENT_HASH_FILE


def render_template(template_name: str, context: dict = None) -> str:
    """
    Render a Jinja2 template with the given context.
    """
    # Configure the Jinja2 environment
    TEMPLATE_FOLDER = os.path.join(os.path.dirname(__file__), "prompts")
    loader = jinja2.FileSystemLoader(TEMPLATE_FOLDER)
    jinja_env = jinja2.Environment(loader=loader)
    template = jinja_env.get_template(template_name)
    return template.render(context) if context else template.render()
