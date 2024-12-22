# import required modules
import os
import time
import tempfile
import logging
from typing import Optional
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
from speech_to_text import SpeechToText
import numpy as np
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MeetConfig:
    """Configuration class for Google Meet bot."""
    email: str
    password: str
    meet_link: str
    recording_duration: int = 60
    headless: bool = False
    timeout: int = 20

class AudioRecorder:
    def __init__(self):
        self.sample_rate = int(os.getenv('SAMPLE_RATE', 44100))
        self.channels = 1  # Using mono for better quality
        self.dtype = np.float32  # Better precision for audio data
        
    def _reduce_noise(self, audio_data):
        """Apply noise reduction to the audio data."""
        try:
            b, a = signal.butter(4, 100/(self.sample_rate/2), 'highpass')
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            normalized = np.int16(filtered_audio * 32767)
            return normalized
            
        except Exception as e:
            logging.error(f"Error in noise reduction: {e}")
            return audio_data

    def get_audio(self, filename, duration):
        try:
            logging.info("Starting audio recording...")
            print("Recording...")
            
            # Configure audio stream with better quality
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=1024,
                latency='low'
            ) as stream:
                frames = []
                for _ in range(int(duration * self.sample_rate / 1024)):
                    data, _ = stream.read(1024)
                    frames.append(data)
                
                recording = np.concatenate(frames)
                
                # cleaned_audio = self._reduce_noise(recording)
                
                write(filename, self.sample_rate, recording)
                print(f"Recording finished. Saved as {filename}.")
                
        except Exception as e:
            logging.error(f"Error during audio recording: {e}")
            raise

class GoogleMeetBot:
    """
    A bot to automatically join Google Meet sessions.
    
    Features:
    - Automated login to Google account
    - Joins specified Google Meet link
    - Controls microphone and camera
    - Records audio and transcribes it
    """
    
    def __init__(self, config: MeetConfig):
        """Initialize the bot with given configuration."""
        self.config = config
        self.driver = self._setup_driver()
        self.wait = WebDriverWait(self.driver, 30)
        
    def _setup_driver(self) -> webdriver.Chrome:
        """Set up and configure Chrome WebDriver."""
        options = Options()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--start-maximized')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        if self.config.headless:
            options.add_argument('--headless')
            
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.media_stream_mic": 1,
            "profile.default_content_setting_values.media_stream_camera": 1,
            "profile.default_content_setting_values.geolocation": 0,
            "profile.default_content_setting_values.notifications": 1
        })
        
        return webdriver.Chrome(options=options)
    
    def login(self) -> None:
        """Log in to Google account with improved waiting mechanisms."""
        try:
            logger.info("Starting Google login process")
            self.driver.get('https://accounts.google.com/ServiceLogin')
            
            email_field = self.wait.until(
                EC.presence_of_element_located((By.ID, "identifierId"))
            )
            email_field.clear()
            email_field.send_keys(self.config.email)
            logger.info("Email entered successfully")
            
            next_button = self.wait.until(
                EC.element_to_be_clickable((By.ID, "identifierNext"))
            )
            next_button.click()
            logger.info("Clicked next after email")
            
            try:
                password_field = WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.NAME, "Passwd"))
                )
                time.sleep(2)  # Brief pause to ensure page transition
                password_field = WebDriverWait(self.driver, 30).until(
                    EC.element_to_be_clickable((By.NAME, "Passwd"))
                )
                password_field.clear()
                password_field.send_keys(self.config.password)
                logger.info("Password entered successfully")
                
                time.sleep(1)
                
                password_next = WebDriverWait(self.driver, 20).until(
                    EC.element_to_be_clickable((By.ID, "passwordNext"))
                )
                password_next.click()
                logger.info("Clicked next after password")
                
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.ID, "gb"))
                )
                logger.info("Successfully logged into Google account")
                
            except TimeoutException:
                logger.error("Timeout waiting for password field")
                self._capture_screenshot("password_field_timeout")
                raise
                
        except Exception as e:
            logger.error(f"Failed to login: {str(e)}")
            self._capture_screenshot("login_failure")
            raise

    def _capture_screenshot(self, name: str) -> None:
        """Capture screenshot for debugging purposes."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{name}_{timestamp}.png"
            self.driver.save_screenshot(filename)
            logger.info(f"Screenshot saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {str(e)}")

    def _control_media_devices(self) -> None:
        """Control microphone and camera settings with improved waiting."""
        try:
            logger.info("Waiting for media controls to load...")
            time.sleep(5)
            
            mic_selectors = [
                "div[role='button'][aria-label*='Turn off microphone']",
                "div[aria-label*='microphone'][role='button']",
                "div[data-is-muted='false'][aria-label*='microphone']",
                "button[aria-label*='Turn off microphone']",
                "div[jscontroller='t2mBxb']"
            ]
            
            camera_selectors = [
                "div[role='button'][aria-label*='Turn off camera']",
                "div[aria-label*='camera'][role='button']",
                "div[data-is-muted='false'][aria-label*='camera']",
                "button[aria-label*='Turn off camera']",
                "div[jscontroller='bwqwSd']"
            ]
            
            mic_found = False
            for selector in mic_selectors:
                try:
                    logger.info(f"Trying microphone selector: {selector}")
                    mic_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    mic_button.click()
                    logger.info("Microphone muted successfully")
                    mic_found = True
                    time.sleep(2)
                    break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {str(e)}")
                    continue
                    
            if not mic_found:
                logger.warning("Could not find microphone button with any selector")
                
            camera_found = False
            for selector in camera_selectors:
                try:
                    logger.info(f"Trying camera selector: {selector}")
                    camera_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    camera_button.click()
                    logger.info("Camera turned off successfully")
                    camera_found = True
                    time.sleep(2)
                    break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {str(e)}")
                    continue
                    
            if not camera_found:
                logger.warning("Could not find camera button with any selector")

            logger.info("Dumping page source for debugging...")
            with open("page_source.html", "w") as f:
                f.write(self.driver.page_source)
                
        except Exception as e:
            logger.error(f"Failed to control media devices: {str(e)}")
            self._capture_screenshot("media_control_failure")
            raise

    def join_meeting(self) -> None:
        """Join the specified Google Meet meeting."""
        try:
            logger.info(f"Navigating to meeting: {self.config.meet_link}")
            self.driver.get(self.config.meet_link)
            
            time.sleep(5) 
            
            self._control_media_devices()
            
            join_selectors = [
                "button[jsname='Qx7uuf']",
                "button[aria-label*='Join now']",
                "button[aria-label*='Ask to join']"
            ]
            
            join_button = None
            for selector in join_selectors:
                try:
                    logger.info(f"Looking for join button with selector: {selector}")
                    join_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    if join_button:
                        break
                except:
                    continue
            
            if join_button:
                logger.info("Found join button, attempting to click...")
                time.sleep(2)
                join_button.click()
                logger.info("Join button clicked")
                
                time.sleep(5)
            else:
                raise Exception("Could not find any join button")

        except Exception as e:
            logger.error(f"Failed to join meeting: {str(e)}")
            self._capture_screenshot("join_failure")
            raise

    def _verify_join_status(self) -> bool:
        """Verify that we've successfully joined the meeting."""
        try:
            join_indicators = [
                "div[data-self-name]",
                "div[aria-label*='Meeting details']",
                "div[aria-label*='participants']"
            ]
            
            for indicator in join_indicators:
                try:
                    if self.driver.find_elements(By.CSS_SELECTOR, indicator):
                        logger.info(f"Found join indicator: {indicator}")
                        return True
                except:
                    continue
                    
            return False
        except Exception as e:
            logger.error(f"Error verifying join status: {str(e)}")
            return False
    
    def _verify_muted_state(self) -> bool:
        """Verify that both microphone and camera are muted."""
        try:
            self.wait.until(EC.presence_of_element_located(
                (By.XPATH, "//div[@data-is-muted='true'][@data-tooltip-id='microphone']")
            ))
            self.wait.until(EC.presence_of_element_located(
                (By.XPATH, "//div[@data-is-muted='true'][@data-tooltip-id='camera']")
            ))
            return True
        except TimeoutException:
            return False
    
    def _click_join_button(self) -> None:
        """Click the join meeting button."""
        try:
            join_button = self.wait.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, 'button[jsname="Qx7uuf"]')
            ))
            join_button.click()
            logger.info("Clicked join button")
            
        except TimeoutException:
            logger.error("Join button not found")
            raise
    
    def _capture_screenshot(self, name: str) -> None:
        """Capture screenshot for debugging purposes."""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"screenshot_{name}_{timestamp}.png"
            self.driver.save_screenshot(filename)
            logger.info(f"Screenshot saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.driver:
                self.driver.quit()
                logger.info("Browser session closed")
        except Exception as e:
            logger.error(f"Failed to cleanup: {str(e)}")

def main():
    """Main entry point of the script."""
    load_dotenv()
    
    temp_dir = 'tmp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    audio_path = os.path.join(temp_dir, "output.wav")
    bot = None
    
    try:
        config = MeetConfig(
            email=os.getenv('EMAIL_ID'),
            password=os.getenv('EMAIL_PASSWORD'),
            meet_link=os.getenv('MEET_LINK'),
            recording_duration=int(os.getenv('RECORDING_DURATION', 30)),
            headless=bool(os.getenv('HEADLESS', False))
        )
        
        if not all([config.email, config.password, config.meet_link]):
            raise ValueError("Missing required environment variables")
        
        bot = GoogleMeetBot(config)
        bot.login()
        bot.join_meeting()

        recorder = AudioRecorder()
        # recorder.list_audio_devices()
        
        recorder.get_audio(audio_path, duration=60)
        transcript = SpeechToText().transcribe(audio_path)
        logger.info(f"Transcription: {transcript}")
        
    except Exception as e:
        logger.error(f"Bot execution failed: {str(e)}")
        raise
        
    finally:
        try:
            pass
            # if os.path.exists(audio_path):
            #     os.remove(audio_path)
            # if os.path.exists(temp_dir):
            #     os.rmdir(temp_dir)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
        
        if bot is not None:
            bot.cleanup()

if __name__ == "__main__":
    main()