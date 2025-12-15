import asyncio
from perception.stt import stt_live
from perception.tone import tone_sentiment_live

async def test_pitch_tone_integration():
    # Mock handle_text to capture pitch and text
    results = []
    def handle_text(text, pitch):
        analysis = tone_sentiment_live.analyze_tone(text, pitch)
        results.append((text, pitch, analysis))

    # Simulate recording and processing once (replace with actual audio file if needed)
    # For testing, we will call extract_pitch on a sample wav file if available
    # Here, we mock the pitch and text for demonstration
    sample_text = "I am very happy today!"
    sample_pitch = 220.0  # High pitch indicating excitement

    # Directly call analyze_tone with sample data
    analysis = tone_sentiment_live.analyze_tone(sample_text, sample_pitch)
    print("Analysis with high pitch:", analysis)

    sample_text2 = "I feel sad and down."
    sample_pitch2 = 80.0  # Low pitch indicating sadness

    analysis2 = tone_sentiment_live.analyze_tone(sample_text2, sample_pitch2)
    print("Analysis with low pitch:", analysis2)

    # Test start_stt function with mocked handle_text (optional, requires audio device)
    # await stt_live.start_stt(handle_text)

if __name__ == "__main__":
    asyncio.run(test_pitch_tone_integration())
