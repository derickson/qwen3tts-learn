import torch

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICES = {
    "dave": {
        "ref_audio": "./DaveSample.m4a",
        "ref_text": (
            "Hey everybody this is Dave Erickson and I wanted to record a sample of my voice. "
            "This is so that I can clone my voice. It's important that i have the ability to "
            "create a digital twin of myself. I am expressive and I speak in an uh eloquent "
            "and uh maybe a little bit  informal tone. Thanks"
        ),
        "avatar_video": "dave_avatar.mp4",
        "default_text": (
            "Hey everyone! My name is Dave and I am a cloned voice ... ... I used a small "
            "sample of my voice to generate what I'm saying right now using an air-gapped Text "
            "To Speech model.  Check out the documentation for how to call this service. "
            "... ... Thanks!"
        ),
    },
    "claire": {
        "ref_audio": "./claire.mp3",
        "ref_text": (
            "Wow, I am really embarrassed! I've never been to such a wild party and I think, "
            "... well, maybe I could get used to this kind of luxury treatment."
        ),
        "avatar_video": "claire_avatar.mp4",
        "default_text": (
            "Hey everyone! My name is Claire and I am a cloned voice ... ... I used a small "
            "sample of my voice to generate what I'm saying right now using an air-gapped Text "
            "To Speech model.  Check out the documentation for how to call this service. "
            "... ... Thanks!"
        ),
    },
    "rex": {
        "ref_audio": "./rex.mp3",
        "ref_text": (
            "Very good, sir. Rex at your service: a cyber-tyrannosaurus of the digital realm, "
            "dressed in the manners of an English butler and equipped with rather formidable "
            "claws for troublesome systems.  Present me with the quarry, and I shall stalk the "
            "scent trail with patience, study the terrain without haste, and strike only when "
            "the root cause has exposed its throat.  Your domain will be kept orderly, your "
            "prey will be pursued with precision, and any lurking chaos will find itself "
            "facing something ancient, courteous, and exceptionally difficult to out manoeuvre."
        ),
        "avatar_video": "rex_avatar.mp4",
        "default_text": "Very good, sir. Rex at your service.",
    },
}
DEFAULT_VOICE = "dave"
SUPPORTED_LANGUAGES = {
    "English": "English",
    "Spanish": "Spanish",
    "German": "German",
    "French": "French",
}
IDLE_TIMEOUT_SECONDS = 900  # 15 minutes
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
except ImportError:
    ATTN_IMPL = "eager"
