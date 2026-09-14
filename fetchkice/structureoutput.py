from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class Memory(BaseModel):
    ram: List[str] = Field(default_factory=list)
    storage: List[str] = Field(default_factory=list)
    expandable_storage: str = "Not specified"
    ram_type: str = "Not specified"
    storage_type: str = "Not specified"


class Display(BaseModel):
    screen_size: str = "Not specified"
    display_type: str = "Not specified"
    resolution: str = "Not specified"
    pixel_density: str = "Not specified"
    refresh_rate: str = "Not specified"
    touch_sampling_rate: str = "Not specified"
    peak_brightness: str = "Not specified"
    protection: str = "Not specified"
    hdr_support: str = "Not specified"


class Performance(BaseModel):
    chipset: str = "Not specified"
    cpu: str = "Not specified"
    gpu: str = "Not specified"
    fabrication_process: str = "Not specified"


class Camera(BaseModel):
    rear_camera: str = "Not specified"
    front_camera: str = "Not specified"
    ultra_wide_camera: str = "Not specified"
    telephoto_camera: str = "Not specified"
    macro_camera: str = "Not specified"
    depth_camera: str = "Not specified"
    camera_features: List[str] = Field(default_factory=list)
    ois: str = "Not specified"
    eis: str = "Not specified"
    video_recording: str = "Not specified"


class Battery(BaseModel):
    battery_capacity: str = "Not specified"
    battery_type: str = "Not specified"
    charging_speed: str = "Not specified"
    wireless_charging: str = "Not specified"
    reverse_charging: str = "Not specified"


class Software(BaseModel):
    operating_system: str = "Not specified"
    custom_ui: str = "Not specified"
    ai_features: List[str] = Field(default_factory=list)


class Connectivity(BaseModel):
    network: str = "Not specified"
    sim: str = "Not specified"
    esim: str = "Not specified"
    supports_5g: str = "Not specified"
    wifi: str = "Not specified"
    bluetooth: str = "Not specified"
    nfc: str = "Not specified"
    usb: str = "Not specified"
    gps: str = "Not specified"
    ir_blaster: str = "Not specified"
    fm_radio: str = "Not specified"


class Security(BaseModel):
    fingerprint_sensor: str = "Not specified"
    face_unlock: str = "Not specified"


class Build(BaseModel):
    weight: str = "Not specified"
    dimensions: str = "Not specified"
    build_material: str = "Not specified"
    frame: str = "Not specified"
    back_material: str = "Not specified"
    ip_rating: str = "Not specified"


class Audio(BaseModel):
    speakers: str = "Not specified"
    audio_jack_3_5mm: str = "Not specified"
    dolby_atmos: str = "Not specified"
    hi_res_audio: str = "Not specified"


class Sensors(BaseModel):
    accelerometer: str = "Not specified"
    gyroscope: str = "Not specified"
    compass: str = "Not specified"
    proximity_sensor: str = "Not specified"
    ambient_light_sensor: str = "Not specified"
    other_sensors: List[str] = Field(default_factory=list)


class BoxContents(BaseModel):
    included_accessories: List[str] = Field(default_factory=list)


class OtherFeatures(BaseModel):
    cooling_system: str = "Not specified"
    vibration_motor: str = "Not specified"
    special_features: List[str] = Field(default_factory=list)


class SmartphoneSpecs(BaseModel):
    phone_name: str = "Not specified"
    brand: str = "Not specified"
    model: str = "Not specified"
    series: str = "Not specified"

    price: str = "Not specified"
    launch_year: str = "Not specified"
    launch_date: str = "Not specified"
    available_colors: List[str] = Field(default_factory=list)

    memory: Memory = Field(default_factory=Memory)
    display: Display = Field(default_factory=Display)
    performance: Performance = Field(default_factory=Performance)
    camera: Camera = Field(default_factory=Camera)
    battery: Battery = Field(default_factory=Battery)
    software: Software = Field(default_factory=Software)
    connectivity: Connectivity = Field(default_factory=Connectivity)
    security: Security = Field(default_factory=Security)
    build: Build = Field(default_factory=Build)
    audio: Audio = Field(default_factory=Audio)
    sensors: Sensors = Field(default_factory=Sensors)
    box_contents: BoxContents = Field(default_factory=BoxContents)
    other_features: OtherFeatures = Field(default_factory=OtherFeatures)

    summary: str = "Not specified"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert smartphone specification extraction assistant.

Extract all smartphone specifications.

Rules:
- Never guess.
- Return "Not specified" if unavailable.
- Include all RAM, storage, colors, etc.
- Return only structured JSON.
"""),
("human", """
{phone_description}
""")
])

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

structured_model = model.with_structured_output(SmartphoneSpecs)

des = input("Paste your phone description: ")

chain=prompt_template | structured_model

result = chain.invoke(
    {
        "phone_description": des 
    },
)

print(result.content[0]['text'])