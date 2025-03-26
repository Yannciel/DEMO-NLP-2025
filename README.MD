# Chatbot (Multi Agent system)

A Natural Language Processing demonstration project featuring a chatbot implementation and data processing capabilities.

## Overview

This project showcases various NLP techniques and applications, including:
- Chatbot implementation with modern NLP approaches
- Data processing for different file formats (Excel, PowerPoint)
- Interactive web interface for demonstration


## Project Structure

```
DEMO-NLP-2025/
├── chatbot/
│   ├── app.py         # Main chatbot application
│   └── graph.py       # LangGraph  module
├── data/
│   ├── data_exemple_Excel.jsonl  # Excel data samples
│   └── data_exemple_ppt.jsonl    # PowerPoint data samples
├── index.html         # Web interface
└── requirement.txt    # Project dependencies
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirement.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yannciel/DEMO-NLP-2025.git
cd DEMO-NLP-2025
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirement.txt
```

## Usage

1. Start the chatbot application:
```bash
streamlit chatbot/app.py
```


## Features

- **Intelligent Chatbot**: Advanced NLP-powered conversational agent
- **Data Processing**: Support for Excel and PowerPoint file analysis
- **Web Interface**: User-friendly interaction platform

## Development

- The project follows standard Python coding conventions
- Use the provided `.gitignore` for proper version control
- Contributions should follow the existing code structure


## Contact

For any questions or suggestions, please open an issue in the GitHub repository.

## Acknowledgments

- Thanks to all contributors and users of this demo project
- Special thanks to the open-source NLP community
