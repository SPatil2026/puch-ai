import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# Initialize Gemini AI
import google.generativeai as genai
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Tool: File Manager ---
@mcp.tool(description="Read file contents")
async def read_file(
    file_path: Annotated[str, Field(description="Path to the file to read")]
) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f"File: {file_path}\n\n{f.read()}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool(description="Write content to file")
async def write_file(
    file_path: Annotated[str, Field(description="Path to the file to write")],
    content: Annotated[str, Field(description="Content to write to the file")]
) -> str:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@mcp.tool(description="List directory contents")
async def list_directory(
    dir_path: Annotated[str, Field(description="Path to the directory to list")] = "."
) -> str:
    try:
        import os
        files = os.listdir(dir_path)
        return f"Directory: {dir_path}\n\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

# --- Tool: Calculator ---
@mcp.tool(description="Perform basic math calculations")
async def calculator(
    expression: Annotated[str, Field(description="Math expression to evaluate (e.g., '2+2', '10*5')")] 
) -> str:
    try:
        import ast
        import operator
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, 
               ast.Div: operator.truediv, ast.Pow: operator.pow}
        def eval_expr(node):
            if isinstance(node, ast.Num): return node.n
            elif isinstance(node, ast.BinOp): return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            else: raise TypeError(node)
        result = eval_expr(ast.parse(expression, mode='eval').body)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Tool: Web Scraper ---
@mcp.tool(description="Extract text content from any webpage")
async def web_scraper(
    url: Annotated[str, Field(description="URL to scrape content from")]
) -> str:
    try:
        content, _ = await Fetch.fetch_url(url, Fetch.USER_AGENT)
        return f"Content from {url}:\n\n{content[:2000]}..."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# --- Tool: Weather API ---
@mcp.tool(description="Get current weather for a city")
async def get_weather(
    city: Annotated[str, Field(description="City name to get weather for")]
) -> str:
    try:
        url = f"https://wttr.in/{city}?format=3"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return f"Weather in {city}: {response.text.strip()}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# --- Tool: Currency Converter ---
@mcp.tool(description="Convert currency using live exchange rates")
async def currency_converter(
    amount: Annotated[float, Field(description="Amount to convert")],
    from_currency: Annotated[str, Field(description="Source currency code (e.g., USD)")],
    to_currency: Annotated[str, Field(description="Target currency code (e.g., EUR)")]
) -> str:
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            rate = data['rates'][to_currency]
            converted = amount * rate
            return f"{amount} {from_currency} = {converted:.2f} {to_currency}"
    except Exception as e:
        return f"Error converting currency: {str(e)}"

# --- Tool: Task Manager ---
tasks = []

@mcp.tool(description="Add a new task to the task list")
async def add_task(
    task: Annotated[str, Field(description="Task description to add")]
) -> str:
    tasks.append({"id": len(tasks) + 1, "task": task, "completed": False})
    return f"Added task: {task}"

@mcp.tool(description="List all tasks")
async def list_tasks() -> str:
    if not tasks:
        return "No tasks found"
    result = "Tasks:\n"
    for t in tasks:
        status = "✓" if t["completed"] else "○"
        result += f"{status} {t['id']}. {t['task']}\n"
    return result

@mcp.tool(description="Mark a task as completed")
async def complete_task(
    task_id: Annotated[int, Field(description="ID of the task to complete")]
) -> str:
    for t in tasks:
        if t["id"] == task_id:
            t["completed"] = True
            return f"Completed task: {t['task']}"
    return "Task not found"

# --- Tool: Code Executor ---
@mcp.tool(description="Execute Python code safely")
async def execute_python(
    code: Annotated[str, Field(description="Python code to execute")]
) -> str:
    try:
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        exec(code, {"__builtins__": {"print": print, "len": len, "str": str, "int": int, "float": float}})
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        return f"Output:\n{output}"
    except Exception as e:
        return f"Error executing code: {str(e)}"

# --- Tool: Text Translator ---
@mcp.tool(description="Translate text using Google Translate")
async def translate_text(
    text: Annotated[str, Field(description="Text to translate")],
    target_lang: Annotated[str, Field(description="Target language code (e.g., 'es' for Spanish)")]
) -> str:
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q={text}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            result = response.json()
            translated = result[0][0][0]
            return f"Translation: {translated}"
    except Exception as e:
        return f"Error translating: {str(e)}"

# --- Tool: News Fetcher ---
@mcp.tool(description="Get latest news headlines")
async def get_news(
    topic: Annotated[str, Field(description="News topic to search for")] = "technology"
) -> str:
    try:
        search_query = f"{topic} news site:news.ycombinator.com OR site:techcrunch.com OR site:bbc.com"
        links = await Fetch.google_search_links(search_query, 5)
        return f"Latest {topic} news:\n\n" + "\n".join(f"• {link}" for link in links[:5])
    except Exception as e:
        return f"Error fetching news: {str(e)}"

# --- AI-POWERED TOOLS ---

# --- Tool: AI Text Generator ---
@mcp.tool(description="Generate creative text using Google Gemini AI")
async def ai_text_generator(
    prompt: Annotated[str, Field(description="Text prompt for AI generation")],
    style: Annotated[str, Field(description="Writing style: creative, professional, casual, technical")] = "creative"
) -> str:
    try:
        if not gemini_model:
            return f"⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
        
        styled_prompt = f"Write in a {style} style: {prompt}"
        response = gemini_model.generate_content(styled_prompt)
        return f"🤖 Gemini AI Generated ({style}):\n\n{response.text}"
    except Exception as e:
        return f"❌ Gemini AI Error: {str(e)}"

# --- Tool: AI Code Generator ---
@mcp.tool(description="Generate code using Google Gemini AI")
async def ai_code_generator(
    description: Annotated[str, Field(description="Description of what code should do")],
    language: Annotated[str, Field(description="Programming language (python, javascript, etc.)")] = "python"
) -> str:
    try:
        if not gemini_model:
            return f"⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
        
        code_prompt = f"Generate {language} code that {description}. Include comments and proper structure."
        response = gemini_model.generate_content(code_prompt)
        return f"🤖 Gemini AI Generated {language.title()} Code:\n\n```{language}\n{response.text}\n```"
    except Exception as e:
        return f"❌ Gemini AI Error: {str(e)}"

# --- Tool: AI Image Analyzer ---
@mcp.tool(description="Analyze images using AI vision")
async def ai_image_analyzer(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to analyze")]
) -> str:
    try:
        import base64
        from PIL import Image
        import io
        
        # Decode and analyze image
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Basic image analysis
        width, height = image.size
        mode = image.mode
        format_type = image.format or "Unknown"
        
        # Color analysis
        colors = image.getcolors(maxcolors=256*256*256)
        dominant_color = max(colors, key=lambda x: x[0])[1] if colors else "Unknown"
        
        analysis = f"""🤖 AI Image Analysis:
        
📏 Dimensions: {width}x{height} pixels
🎨 Format: {format_type} ({mode})
🌈 Dominant Color: {dominant_color}
📊 Complexity: {'High' if len(colors or []) > 1000 else 'Medium' if len(colors or []) > 100 else 'Low'}
🔍 Content: {'Colorful image' if mode == 'RGB' else 'Grayscale image'} with {'complex' if width*height > 500000 else 'moderate'} detail level
        """
        
        return analysis
    except Exception as e:
        return f"AI Image Analysis Error: {str(e)}"

# --- Tool: AI Sentiment Analyzer ---
@mcp.tool(description="Analyze sentiment and emotions in text using AI")
async def ai_sentiment_analyzer(
    text: Annotated[str, Field(description="Text to analyze for sentiment")]
) -> str:
    try:
        # Simple rule-based sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive 😊"
            confidence = min(90, 60 + (pos_count - neg_count) * 10)
        elif neg_count > pos_count:
            sentiment = "Negative 😔"
            confidence = min(90, 60 + (neg_count - pos_count) * 10)
        else:
            sentiment = "Neutral 😐"
            confidence = 50
            
        return f"""🤖 AI Sentiment Analysis:
        
📝 Text: "{text[:100]}{'...' if len(text) > 100 else ''}"
💭 Sentiment: {sentiment}
📊 Confidence: {confidence}%
🔍 Analysis: Found {pos_count} positive and {neg_count} negative indicators
        """
    except Exception as e:
        return f"AI Sentiment Analysis Error: {str(e)}"

# --- Tool: AI Chatbot ---
@mcp.tool(description="Chat with Google Gemini AI assistant")
async def ai_chatbot(
    message: Annotated[str, Field(description="Message to send to AI chatbot")],
    personality: Annotated[str, Field(description="AI personality: helpful, funny, professional, creative")] = "helpful"
) -> str:
    try:
        if not gemini_model:
            return f"⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
        
        personality_prompt = f"Respond in a {personality} manner to: {message}"
        response = gemini_model.generate_content(personality_prompt)
        return f"🤖 Gemini AI Chatbot ({personality.title()}):\n\n{response.text}"
    except Exception as e:
        return f"❌ Gemini AI Error: {str(e)}"

# --- Tool: AI Code Reviewer ---
@mcp.tool(description="Review code quality using Google Gemini AI")
async def ai_code_reviewer(
    code: Annotated[str, Field(description="Code to review")],
    language: Annotated[str, Field(description="Programming language")] = "python"
) -> str:
    try:
        if not gemini_model:
            return f"⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
        
        review_prompt = f"Review this {language} code for bugs, performance issues, and best practices:\n\n{code}"
        response = gemini_model.generate_content(review_prompt)
        return f"🤖 Gemini AI Code Review ({language.title()}):\n\n{response.text}"
    except Exception as e:
        return f"❌ Gemini AI Error: {str(e)}"

# --- Tool: Email Sender ---
@mcp.tool(description="Send email via SMTP (requires email config)")
async def send_email(
    to_email: Annotated[str, Field(description="Recipient email address")],
    subject: Annotated[str, Field(description="Email subject")],
    message: Annotated[str, Field(description="Email message content")]
) -> str:
    return f"Email functionality requires SMTP configuration. Would send:\nTo: {to_email}\nSubject: {subject}\nMessage: {message}"

# --- HACKATHON SPECIAL: AI-Powered Smart Assistant ---
@mcp.tool(description="Ultimate Google Gemini AI assistant that can help with any task")
async def smart_ai_assistant(
    task: Annotated[str, Field(description="What you need help with")],
    context: Annotated[str, Field(description="Additional context or details")] = ""
) -> str:
    try:
        if not gemini_model:
            return f"⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
        
        smart_prompt = f"Help me with this task: {task}\n\nAdditional context: {context}\n\nProvide a comprehensive, actionable response."
        response = gemini_model.generate_content(smart_prompt)
        return f"🧠 Gemini Smart AI Assistant:\n\n{response.text}"
    except Exception as e:
        return f"❌ Gemini AI Error: {str(e)}"

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
