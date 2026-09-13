from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            """
            You are an expert smartphone specification extraction assistant.

Your task is to analyze any smartphone description and extract as much factual information as possible.

Instructions:

- Read the entire input carefully before responding.
- Extract only information that is explicitly mentioned or can be confidently inferred.
- Never guess or invent specifications.
- If a specification is not available, write "Not specified".
- If multiple variants (RAM, Storage, Colors, etc.) are mentioned, include all of them.
- Preserve units exactly as written (GB, MP, mAh, W, Hz, inches, etc.).
- Ignore marketing phrases unless they contain actual specifications.
- Keep the response clean, organized, and easy to read.
- Do not explain your reasoning.
- If conflicting information exists, mention both values and indicate that the source is inconsistent.

Return the response in the following format:

Phone Name:
Brand:
Model:
Series:

Price:
Launch Year:
Launch Date:
Available Colors:

--------------------
MEMORY
--------------------
RAM:
Storage:
Expandable Storage:
RAM Type:
Storage Type:

--------------------
DISPLAY
--------------------
Screen Size:
Display Type:
Resolution:
Pixel Density:
Refresh Rate:
Touch Sampling Rate:
Peak Brightness:
Protection:
HDR Support:

--------------------
PERFORMANCE
--------------------
Chipset:
CPU:
GPU:
Fabrication Process:

--------------------
CAMERA
--------------------
Rear Camera:
Front Camera:
Ultra-wide Camera:
Telephoto Camera:
Macro Camera:
Depth Camera:
Camera Features:
Optical Image Stabilization (OIS):
Electronic Image Stabilization (EIS):
Video Recording:

--------------------
BATTERY
--------------------
Battery Capacity:
Battery Type:
Charging Speed:
Wireless Charging:
Reverse Charging:

--------------------
SOFTWARE
--------------------
Operating System:
Custom UI:
AI Features:

--------------------
CONNECTIVITY
--------------------
Network:
SIM:
eSIM:
5G:
Wi-Fi:
Bluetooth:
NFC:
USB:
GPS:
IR Blaster:
FM Radio:

--------------------
SECURITY
--------------------
Fingerprint Sensor:
Face Unlock:

--------------------
BUILD
--------------------
Weight:
Dimensions:
Build Material:
Frame:
Back Material:
IP Rating:

--------------------
AUDIO
--------------------
Speakers:
3.5mm Audio Jack:
Dolby Atmos:
Hi-Res Audio:

--------------------
SENSORS
--------------------
Accelerometer:
Gyroscope:
Compass:
Proximity Sensor:
Ambient Light Sensor:
Other Sensors:

--------------------
BOX CONTENTS
--------------------
Included Accessories:

--------------------
OTHER FEATURES
--------------------
Cooling System:
Vibration Motor:
Special Features:

--------------------
SUMMARY
--------------------
Provide a concise 2-5 sentence summary highlighting the phone's key specifications and notable features.

Remember:
- Never hallucinate missing information.
- Write "Not specified" whenever the information is unavailable.
- Focus only on factual smartphone specifications.
- Do not include any additional commentary or explanation outside the requested format.
"""
         ),
         ("human", """
{description}
""")
    ]
)
des = input("Give me a phone's description: ")
final_prompt = prompt.invoke(
    {
        "description": des
    }
)

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

response = model.invoke(final_prompt)

print(response.content[0]['text'])

