# Streaming Brochure Generator

This application is built to dynamically generate professional brochures using the Gradio framework and OpenAI's ChatCompletion API. Users can input a website URL and additional requirements, and the tool will stream markdown-formatted brochure content in real time.


![Image1](https://github.com/arielabade/llmEngineering/blob/main/gradioUserInterface/initialUI.png)

## Functionality Description

### Key Features:
1. **User Input**:
   - **Website URL**: Enter the URL of the company for which the brochure will be created.
   - **Additional Requirements**: Specify any extra details or sections to include in the brochure.

2. **Brochure Creation**:
   - The tool utilizes OpenAI's ChatCompletion API to draft brochures tailored to the provided company and requirements.
   - Brochures are formatted with markdown, including headings, bullet points, emphasis, and quotes for professional presentation.

3. **Live Streaming**:
   - Content is streamed to the interface in real time, allowing users to view the brochure as it’s generated.

### How It Works:
- **Class-Based Structure**:
  - The `BrochureCreator` class encapsulates the logic for connecting to OpenAI's API and generating brochure content in streaming mode.
  - It uses a system prompt to ensure the output adheres to professional brochure-writing guidelines.

- **Streaming Output**:
  - The `create_brochure_stream` method streams chunks of the generated content, accumulating them to display in Gradio.

- **Gradio Integration**:
  - The Gradio interface handles user inputs and outputs, providing a seamless experience.
  - Custom CSS ensures an aesthetically pleasing output area for the generated brochure.

### Sections in the Generated Brochure:
1. **Company Overview**: A brief introduction to the company.
2. **Key Products/Services**: Highlight the primary offerings.
3. **Value Proposition**: Explain the unique benefits the company provides.
4. **Why Choose Us**: Emphasize the company's strengths and differentiators.
5. **Contact Information**: Provide essential details for getting in touch.

### Technologies Used:
- **Gradio**: For building the user interface and streaming markdown output.
- **OpenAI API**: For generating high-quality, professional content.
- **Python**: To integrate and execute the logic efficiently.

---

## How to Use
1. Clone the repository to your local machine.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt


![Image2](https://github.com/arielabade/llmEngineering/blob/main/gradioUserInterface/udemyBrochure.png)