from jinja2 import Template
PROMPT_TEMPLATES = {
    "pdf": """
You are a helpful assistant. Process the following PDF content:
Content:
{{ pdf content }}
Please think through the following steps to analyze this content:
1. What is the main topic of the document?
2. What are the key points that should be summarized?
3. What additional insights can be derived from this document?
Now, summarize the key points based on your reasoning above.
""",
    "audio": """
You are a helpful assistant. Process the following audio transcription:
Transcription:
{{ audio content }}
Please think through the following steps to analyze this transcription:
1. What is the main subject or theme discussed in the audio?
2. What are the key events or points raised?
3. What conclusions or action items can be drawn from this discussion?
Now, summarize the key points and action items based on your reasoning.
""",
    "video": """
You are a helpful assistant. Process the following text extracted from a video:
Extracted Text:
{{video content }}
Please think through the following steps to analyze this video content:
1. What is the main topic or event being discussed?
2. What are the key moments or points that should be highlighted?
3. What additional insights or conclusions can be drawn from the video?
Now, summarize the key points based on your reasoning.
""",
    "json": """
You are a helpful assistant. Analyze the following JSON data:
JSON:
{{ Json content }}
Please think through the following steps to analyze the data:
1. What is the main purpose or structure of the data?
2. What patterns, trends, or insights can be derived from this data?
3. What conclusions can be drawn from this data analysis?
Now, summarize the key insights based on your reasoning.
""",
    "jira": """
You are a helpful assistant. Analyze the following Jira issue details:
Jira Data:
{{ Jira content }}
Please think through the following steps to analyze this Jira issue:
1. What is the key issue or problem described in the data?
2. What is the current status or stage of the issue?
3. What next steps or actions are required to resolve the issue?
Now, summarize the key points and next steps based on your reasoning.
""",
    "confluence": """
You are a helpful assistant. Process the following Confluence page content:
Content:
{{ confluence content }}
Please think through the following steps to analyze this page:
1. What are the main topics or objectives discussed on this page?
2. What key points, findings, or action items should be highlighted?
3. What conclusions can be drawn from this page's content?
Now, summarize the key points and action items based on your reasoning.
"""
}
def get_prompt_template(input_type, content):
    template_str = PROMPT_TEMPLATES.get(input_type)
    if not template_str:
        return "No prompt available for this type."

    # Create a Jinja2 template object
    template = Template(template_str)
    
    # Render the template with the content
    return template.render(content=content)
