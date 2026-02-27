import json
from datetime import datetime

def display_json_data(data: dict, title: str = "DATA RECEIVED", level: str = "INFO") -> None:
    """
    Displays JSON data in the console with formatting.
    
    Args:
        data (dict): The data to display.
        title (str): Title for the console output.
        level (str): Log level (INFO, DEBUG, etc).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 80
    
    print(f"\n{separator}")
    print(f"[{timestamp}] {level}: {title}")
    print(separator)
    
    try:
        # Pretty print JSON
        json_str = json.dumps(data, indent=2, default=str)
        print(json_str)
    except (TypeError, ValueError) as e:
        print(f"Error stringifying data: {str(e)}")
        print(f"Data type: {type(data)}")
        print(f"Data: {data}")
    
    print(separator + "\n")

def stringify_insights(insights: dict, max_items: int = 5) -> str:
    """
    Converts AGI therapist insight JSON into structured prompt context string.
    
    Args:
        insights (dict): Output from perception, memory, reasoning modules.
        max_items (int): Limits list size to avoid token explosion.
    
    Returns:
        str: Clean structured context string for LLM prompt.
    """

    def format_section(title, data):
        if not data:
            return ""

        section_lines = [f"[{title.upper()}]"]

        for key, value in data.items():
            if value is None:
                continue

            # Shorten long lists
            if isinstance(value, list):
                trimmed = value[:max_items]
                value_str = ", ".join(map(str, trimmed))
            # Compact dict
            elif isinstance(value, dict):
                trimmed_items = list(value.items())[:max_items]
                value_str = ", ".join(
                    f"{k}:{v}" for k, v in trimmed_items if v is not None
                )
            else:
                value_str = str(value)

            section_lines.append(f"{key.replace('_',' ').title()}: {value_str}")

        return "\n".join(section_lines)

    sections = []

    for section_name in ["perception", "memory", "reasoning"]:
        section_data = insights.get(section_name, {})
        formatted = format_section(section_name, section_data)
        if formatted:
            sections.append(formatted)

    return "\n\n".join(sections).strip()