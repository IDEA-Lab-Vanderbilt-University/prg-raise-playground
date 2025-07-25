import { ArgumentType, BlockType, Environment, ExtensionMenuDisplayDetails, extension, block, fetchWithTimeout, wrapClamp } from "$common";
import BlockUtility from "$scratch-vm/engine/block-utility";
import { getSynthesisURL } from "./services/synthesis";
import { getState, setState, State } from "./state";
import timer from "./timer";
import voices, { Voice } from "./voices";

const details: ExtensionMenuDisplayDetails = {
  name: "Speech",
  // description: "Blocks for speech synthesis and recognition.",
  description: "Blocks for speech synthesis.",
  iconURL: "Replace with the name of your icon image file (which should be placed in the same directory as this file)",
  insetIconURL: "Replace with the name of your inset icon image file (which should be placed in the same directory as this file)"
};

export default class SpeechExtension extends extension(details) {

  init(env: Environment) {
    // this.lastRecognizedSpeech = "";
  }

  // lastRecognizedSpeech: string;
  currentLoudness: number;
  loudnessTimer = timer();
  soundPlayers = new Map();


  @block({
    type: BlockType.Command,
    text: (text) => `speak ${text}`,
    args: [{ type: ArgumentType.String, defaultValue: "Hello!" }]
  })
  async speakText(text: string, { target }: BlockUtility) {
    await this.speak(text, getState(target));
  }

  // @block({
  //   type: BlockType.Command,
  //   text: (prompt) => `speak ${prompt} and wait for response`,
  //   args: [{ type: ArgumentType.String, defaultValue: "How are you?" }]
  // })
  // async askSpeechRecognition(prompt: string, { target }: BlockUtility) {
  //   await this.speak(prompt, getState(target));
  //   await this.recognizeSpeech();
  // }

  // @block({
  //   type: BlockType.Reporter,
  //   text: "response"
  // })
  // getRecognizedSpeech() {
  //   return this.lastRecognizedSpeech;
  // }

  @block({
    type: BlockType.Command,
    text: (voice) => `Set voice to ${voice}`,
    args: [{ type: ArgumentType.String, options: ["SQUEAK", "TENOR", "ALTO", "GIANT"], defaultValue: "SQUEAK" }]
  })
  setVoice(voice: Voice, { target }: BlockUtility) {
    const state = getState(target);
    state.currentVoice = voice ?? state.currentVoice;
    setState(target, state);
  }

  @block({type: BlockType.Hat,
    text: (threshold) => `when heard sound > ${threshold}`,
    args: [{type: ArgumentType.Number, defaultValue: 10}]
  })
  onHeardSound(threshold: number) {
    return this.getLoudness() > threshold;
  }

  private async speak(text: string, { currentVoice }: State) {
    const locale = 'en-US';
    const { audioEngine } = this.runtime;
    const { gender, playbackRate } = voices[currentVoice];
    const encoded = encodeURIComponent(text.substring(0, 128));
    const endpoint = getSynthesisURL({ gender, locale, text: encoded });

    await new Promise<void>(async (resolve) => {
      try {
        const response = await fetchWithTimeout(endpoint, { timeoutMs: 40000 });
        if (!response.ok) return console.warn(response.statusText);
        const buffer = await response.arrayBuffer();
        const soundPlayer = await audioEngine.decodeSoundPlayer({ data: { buffer } });
        this.soundPlayers.set(soundPlayer.id, soundPlayer);
        soundPlayer.setPlaybackRate(playbackRate);
        const chain = audioEngine.createEffectChain();
        chain.set('volume', 250);
        soundPlayer.connect(chain);
        soundPlayer.play();
        soundPlayer.on('stop', () => {
          this.soundPlayers.delete(soundPlayer.id);
          resolve();
        });
      }
      catch (error) {
        console.warn(error);
      }
    });
  }

  private getLoudness() {
    const { audioEngine, currentStepTime } = this.runtime;
    if (!audioEngine || !currentStepTime) return -1;

    if (this.loudnessTimer.elapsed() > this.runtime.currentStepTime) {
      this.currentLoudness = audioEngine.getLoudness();
      this.loudnessTimer.start();
    }

    return this.currentLoudness;
  }

  // private async recognizeSpeech() {
  //   return await new Promise<string>((resolve, reject) => {
  //     const recognition = new webkitSpeechRecognition();
  //     recognition.continuous = false;
  //     recognition.interimResults = false;
  //     recognition.lang = 'en-US';

  //     let resolved = false;

  //     recognition.onresult = (event) => {
  //       if (!resolved) {
  //         resolved = true;
  //         const transcript = event.results.length === 0 ? null : event.results[0][0].transcript;
  //         resolve(transcript);
  //       }
  //       recognition.stop();
  //     };

  //     recognition.onerror = (event) => {
  //       if (!resolved) {
  //         resolved = true;
  //         reject(event.error);
  //       }
  //       recognition.stop();
  //     };

  //     recognition.onend = () => {
  //       if (!resolved) {
  //         resolved = true;
  //         resolve(null);
  //       }
  //     };

  //     try {
  //       recognition.start();
  //     } catch (err) {
  //       if (!resolved) {
  //         resolved = true;
  //         reject(err);
  //       }
  //     }
  //   }).then(result => {
  //     this.lastRecognizedSpeech = result ?? this.lastRecognizedSpeech;
  //     return result;
  //   }).catch(error => {
  //     console.warn("Speech recognition error:", error);
  //     return null;
  //   });
  // }
}