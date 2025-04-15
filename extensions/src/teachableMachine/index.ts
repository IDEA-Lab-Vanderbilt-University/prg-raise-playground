import { Environment, buttonBlock, extension } from "$common";
import tmImage from '@teachablemachine/image';
import tmPose from '@teachablemachine/pose';
import { create } from '@tensorflow-models/speech-commands';
import { legacyFullSupport, legacyIncrementalSupport, } from "./legacy";

const { legacyBlock, legacyExtension } = legacyIncrementalSupport.for<teachableMachine>();
const VideoState = {
  /** Video turned off. */
  OFF: 'off',
  /** Video turned on with default y axis mirroring. */
  ON: 'on',
  /** Video turned on without default y axis mirroring. */
  ON_FLIPPED: 'on-flipped',
} as const;


let apiEndpoint;

const dynamicClassMenu = (self: teachableMachine) => ({
  argumentMethods: { 0: { getItems: () => self.getClasses() } }
})

const dynamicModelMenu = (self: teachableMachine) => ({
  argumentMethods: {
    0: {
      // handler: (reported: unknown) => String(reported),
      getItems: () => self.getModels(),
    }
  }
})

@legacyExtension()
export default class teachableMachine extends extension({
  name: "Traininator",
  description: "Use your Machine Learning models in your Scratch project!",
  iconURL: "teachable-machine-blocks.png",
  insetIconURL: "teachable-machine-blocks-small.svg",
  tags: ["Dancing with AI", "Made by PRG"]
}, "indicators") {
  lastUpdate: number;
  modelsList: { text: string, value: string }[];
  maxConfidence: number;
  modelConfidences: {};
  isPredicting: number;
  predictionState = {};
  teachableImageModel;
  latestAudioResults: any;

  test: string = "";

  /**
   * Video refresh rate
   * @type {number}
   */
  INTERVAL = 33;
  /**
   * Dimensions of the video frame
   * @type {number[]}
   */
  DIMENSIONS = [480, 360];

  ModelType = {
    POSE: 'pose',
    IMAGE: 'image',
    AUDIO: 'audio',
  };

  async init(env: Environment) {
    const urlParams = new URLSearchParams(window.location.search);

    // Detect if running on localhost
    const host = urlParams.get('host') || window.location.hostname;
    
    if (host.includes('localhost')) {
      apiEndpoint = 'http://localhost:8080';
    } else {
      apiEndpoint = 'https://spotcommandapp.com/api';
    }

    /**
     * The last millisecond epoch timestamp that the video stream was
     * analyzed.
     * @type {number}
     */
    this.lastUpdate = null;


    // What is the confidence of the latest prediction
    this.maxConfidence = null;
    this.modelConfidences = {};

    // get list of models
    const studentId = urlParams.get('student_id');
    try {
      const res = await (await fetch(`${apiEndpoint}/traininator-models?student_id=${studentId}`)).json();
      this.modelsList = res.map(m => ({ text: m.name, value: `${apiEndpoint}/traininator-models/${m.id}/` }))
    } catch (e) {
      console.log(e);
    }

    if (this.runtime.ioDevices) {
      // Configure the video device with values from globally stored locations.
      // this.runtime.on(Runtime.PROJECT_LOADED, this.updateVideoDisplay.bind(this));

      // this.runtime.ioDevices.video.enableVideo();

      // Kick off looping the analysis logic.
      this._loop();
    }

    window['submitTravelLog'] = (description = "codinatorimage", status = "complete") => {
      const svgElement = document.querySelector("svg.blocklySvg");
      // Check if the SVG element exists
      if (!svgElement) {
        console.error("SVG element not found.");
        return;
      }

      this.svgToPng(svgElement, (pngUrl) => {
        // send the image to the API endpoint
        fetch(`${apiEndpoint}/travel-logs`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify({
            description,
            data: JSON.stringify({ response: pngUrl }),
            status,
            student_id: studentId,
          })
        })
        .then(response => {
          if (!response.ok) {
            throw new Error('Failed to submit travel log');
          }
          return response.json();
        });
      });
    };

    window.addEventListener('message', (event) => {
      if (event.data.type === 'submitTravelLog') {
        const { description, status } = event.data;
        window['submitTravelLog'](description, status);
      }
    });
  }

  /**
     * Occasionally step a loop to sample the video, stamp it to the preview
     * skin, and add a TypedArray copy of the canvas's pixel data.
     * @private
     */
  _loop() {
    setTimeout(this._loop.bind(this), Math.max(this.runtime.currentStepTime, this.INTERVAL));

    // Add frame to detector
    const time = Date.now();
    if (this.lastUpdate === null) {
      this.lastUpdate = time;
    }
    if (!this.isPredicting) {
      this.isPredicting = 0;
    }
    const offset = time - this.lastUpdate;

    // TODO: Self-throttle interval if slow to run predictions
    if (offset > this.INTERVAL && this.isPredicting === 0) {
      const frame = this.runtime.ioDevices.video.getFrame({
        format: 'image-data',
        dimensions: this.DIMENSIONS
      });

      this.lastUpdate = time;
      this.isPredicting = 0;
      this.predictAllBlocks(frame);
    }
  }

  async predictAllBlocks(frame) {
    for (let modelUrl in this.predictionState) {
      if (!this.predictionState[modelUrl].model) {
        continue;
      }
      if (this.teachableImageModel !== modelUrl) {
        continue;
      }
      ++this.isPredicting;
      const prediction = await this.predictModel(modelUrl, frame);
      this.predictionState[modelUrl].topClass = prediction;
      // this.runtime.emit(this.runtime.constructor.PERIPHERAL_CONNECTED);
      --this.isPredicting;
    }
  }

  async predictModel(modelUrl, frame) {
    const predictions = await this.getPredictionFromModel(modelUrl, frame);
    if (!predictions) {
      return;
    }
    let maxProbability = 0;
    let maxClassName = "";
    for (let i = 0; i < predictions.length; i++) {
      const probability = predictions[i].probability.toFixed(2);
      const className = predictions[i].className;
      this.modelConfidences[className] = probability; // update for reporter block
      if (probability > maxProbability) {
        maxClassName = className;
        maxProbability = probability;
      }
    }
    this.maxConfidence = maxProbability; // update for reporter block
    return maxClassName;
  }

  async getPredictionFromModel(modelUrl, frame) {
    const { model, modelType } = this.predictionState[modelUrl];
    switch (modelType) {
      case this.ModelType.IMAGE:
        if (!frame) return null;
        const imageBitmap = await createImageBitmap(frame);
        return await model.predict(imageBitmap);
      case this.ModelType.POSE:
        if (!frame) return null;
        const { pose, posenetOutput } = await model.estimatePose(frame);
        return await model.predict(posenetOutput);
      case this.ModelType.AUDIO:
        if (this.latestAudioResults) {
          return model.wordLabels().map((label, i) => {
            return { className: label, probability: this.latestAudioResults.scores[i] }
          });
        } else {
          return null;
        }
    }
  }

  async startPredicting(modelDataUrl) {
    const alreadyLoaded = Boolean(this.predictionState[modelDataUrl]);
    try {
      const indicator = await this.indicate({
        type: "warning",
        msg: alreadyLoaded ? "Updating model" : "Loading model"
      });
      this.predictionState[modelDataUrl] = {};
      // https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/image
      const { model, type } = await this.initModel(modelDataUrl);
      this.predictionState[modelDataUrl].modelType = type;
      this.predictionState[modelDataUrl].model = model;
      this.runtime.requestToolboxExtensionsUpdate();
      indicator.close();
      this.indicateFor({ type: "success", msg: "Model loaded" }, 1);
    } catch (e) {
      this.predictionState[modelDataUrl] = {};
      console.log("Model initialization failure!", e);
      this.indicateFor({ type: "error", msg: "Unable to load model." }, 1);
    }
  }

  /**
   * A scratch reporter that returns the top class seen in the current video frame
   * @returns {string} class name if video frame matched, empty string if model not loaded yet
   */
  getModelPrediction() {
    const modelUrl = this.teachableImageModel;
    const predictionState: { topClass: string } = this.getPredictionStateOrStartPredicting(modelUrl);
    if (!predictionState) {
      return '';
    }
    return predictionState.topClass;
  }

  async initModel(modelUrl: string) {
    const avoidCache = `?x=${Date.now()}`;
    const modelURL = modelUrl + "model.json" + avoidCache;
    const metadataURL = modelUrl + "metadata.json" + avoidCache;
    const customMobileNet = await tmImage.load(modelURL, metadataURL);
    if ((customMobileNet as any)._metadata.hasOwnProperty('tfjsSpeechCommandsVersion')) {
      // customMobileNet.dispose(); // too early to dispose
      //console.log("We got a speech net yay")
      const recognizer = create("BROWSER_FFT", undefined, modelURL, metadataURL);
      await recognizer.ensureModelLoaded();
      await recognizer.listen(async result => {
        this.latestAudioResults = result;
        //console.log(result);
      }, {
        includeSpectrogram: true, // in case listen should return result.spectrogram
        probabilityThreshold: 0.75,
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: 0.50 // probably want between 0.5 and 0.75. More info in README
      });
      return { model: recognizer, type: this.ModelType.AUDIO };
    } else if ((customMobileNet as any)._metadata.packageName === "@teachablemachine/pose") {
      const customPoseNet = await tmPose.load(modelURL, metadataURL);
      return { model: customPoseNet, type: this.ModelType.POSE };
    } else {
      console.log(customMobileNet.getMetadata(), customMobileNet.getTotalClasses(), customMobileNet.getClassLabels());
      return { model: customMobileNet, type: this.ModelType.IMAGE };
    }
  }

  useModel(url: string) {
    try {
      this.getPredictionStateOrStartPredicting(url, true);
      this.updateStageModel(url);
    } catch (e) {
      this.teachableImageModel = null;
    }
  }

  updateStageModel(modelUrl) {
    const stage = this.runtime.getTargetForStage();
    this.teachableImageModel = modelUrl;
    if (stage) {
      (stage as any).teachableImageModel = modelUrl;
    }
  }

  getPredictionStateOrStartPredicting(modelUrl, override = false) {
    const hasPredictionState = this.predictionState.hasOwnProperty(modelUrl);
    if (!hasPredictionState || override) {
      this.startPredicting(modelUrl);
      return null;
    }
    return this.predictionState[modelUrl];
  }

  getClasses() {
    if (
      !this.teachableImageModel ||
      !this.predictionState ||
      !this.predictionState[this.teachableImageModel] ||
      !this.predictionState[this.teachableImageModel].hasOwnProperty('model')
    ) {
      return ["Select a class"];
    }

    if (this.predictionState[this.teachableImageModel].modelType === this.ModelType.AUDIO) {
      return this.predictionState[this.teachableImageModel].model.wordLabels();
    }

    return this.predictionState[this.teachableImageModel].model.getClassLabels();
  }

  getModels() {
    return this.modelsList;
  }

  model_match(state) {
    const modelUrl = this.teachableImageModel;
    const className = state;

    const predictionState = this.getPredictionStateOrStartPredicting(modelUrl);
    if (!predictionState) {
      return false;
    }

    const currentMaxClass = predictionState.topClass;
    return (currentMaxClass === String(className));
  }

  getClassConfidence(state): number {
    return this.modelConfidences[state];
  }

  /**
   * Turns the video camera off/on/on and flipped. This is called in the operation of videoToggleBlock
   * @param state 
   */
  toggleVideo(state: string) {
    if (state === VideoState.OFF) return this.runtime.ioDevices.video.disableVideo();

    this.runtime.ioDevices.video.enableVideo();
    // Mirror if state is ON. Do not mirror if state is ON_FLIPPED.
    this.runtime.ioDevices.video.mirror = (state === VideoState.ON);
  }

  /**
   * Sets the video's transparency. This is called in the operation of setVideoTransparencyBlock
   * @param transparency 
   */
  setTransparency(transparency: number) {
    const trans = Math.max(Math.min(transparency, 100), 0);
    this.runtime.ioDevices.video.setPreviewGhost(trans);
  }

  /**
   * Opens a new tab with the Google Teachable Machine website
   */
  // @buttonBlock("Teachable Machine Site ↗")
  // openTeachableMachine() {
  //   window.open('https://teachablemachine.withgoogle.com/train', '_blank');
  // }

  @legacyBlock.useModelBlock(dynamicModelMenu)
  useModelBlock(url: string) {
    this.useModel(url);
  }

  // @legacyBlock.whenModelMatches(dynamicClassMenu)
  // whenModelMatches(state: string) {
  //   return this.model_match(state);
  // }

  @legacyBlock.modelPrediction()
  modelPrediction() {
    return this.getModelPrediction();
  }

  @legacyBlock.modelMatches(dynamicClassMenu)
  modelMatches(state: string) {
    return this.model_match(state);
  }

  // @legacyBlock.classConfidence(dynamicClassMenu)
  // classConfidence(state: string) {
  //   return this.getClassConfidence(state);
  // }

  @legacyBlock.videoToggle({
    argumentMethods: {
      0: {
        handler: (video_state: string) => {
          return ['on', 'off', 'on-flipped'].includes(video_state) ? video_state : VideoState.ON;
        },
      }
    }
  })
  videoToggle(state: string) {
    this.toggleVideo(state);
  }

  // @legacyBlock.setVideoTransparency()
  // setVideoTransparency(transparency: number) {
  //   this.setTransparency(transparency);
  // }

  
  /**
   * Converts an SVG element on the page to a PNG data URL.
   * It clones the SVG, inlines computed styles, and inlines any embedded images.
   * @param {SVGElement} svgElement - The SVG element to convert.
   * @param {function} callback - Called with the resulting PNG data URL.
   */
   svgToPng(svgElement, callback) {
    // Clone the SVG element and inline its computed styles.
    const clonedSvg = svgElement.cloneNode(true);
    this.inlineAllStyles(svgElement, clonedSvg);

    // Create a hidden container and append the cloned SVG
    const container = document.createElement('div');
    container.style.position = 'absolute';
    container.style.left = '-9999px';
    document.body.appendChild(container);
    container.appendChild(clonedSvg);

    // Remove background, zoom, and scrollbar background from the cloned SVG
    const background = clonedSvg.querySelector('.blocklyMainBackground');
    const zoom = clonedSvg.querySelector('.blocklyZoom');
    const scrollbarBackground = clonedSvg.querySelector('.blocklyScrollbarBackground');
    if (background) background.remove();
    if (zoom) zoom.remove();
    if (scrollbarBackground) scrollbarBackground.remove();

    const clonedBlocklyCanvas = clonedSvg.querySelector('.blocklyBlockCanvas');
    clonedBlocklyCanvas.style.transform = 'none';
    
    // Set new transform on the clonedBlocklyCanvas to bring it into the view
    const clonedBlocklyCanvasBBox = clonedBlocklyCanvas.getBBox();
    const translateX = -clonedBlocklyCanvasBBox.x;
    const translateY = -clonedBlocklyCanvasBBox.y;
    clonedBlocklyCanvas.style.transform = `translate(${translateX}px, ${translateY}px)`;

    // First, record the original dimensions before making changes
    const originalWidth = svgElement.clientWidth || svgElement.getBoundingClientRect().width;
    const originalHeight = svgElement.clientHeight || svgElement.getBoundingClientRect().height;

    // Use bbox to fit full code in svg view with padding
    const bbox = clonedSvg.getBBox();
    const canvasBbox = clonedBlocklyCanvas.getBBox();
    const padding = 10; // Add padding to ensure all content is visible
    const width = Math.max(bbox.width, canvasBbox.width) + padding * 2;
    const height = Math.max(bbox.height, canvasBbox.height) + padding * 2;

    // Force the SVG dimensions using multiple approaches
    clonedSvg.setAttribute('width', `${width}px`);
    clonedSvg.setAttribute('height', `${height}px`);
    clonedSvg.style.width = `${width}px`;
    clonedSvg.style.height = `${height}px`;
    clonedSvg.setAttribute('viewBox', `${-padding} ${-padding} ${width} ${height}`);

    // Also ensure the container doesn't constrain the SVG
    container.style.width = `${width}px`;
    container.style.height = `${height}px`;
    container.style.overflow = 'visible';

    // Inline any embedded images (e.g., <image> elements referencing external SVGs)
    this.inlineImagesInSvg(clonedSvg)
    .then(() => {
        // Now that all images are inlined, serialize the SVG to a string.
        const svgString = new XMLSerializer().serializeToString(clonedSvg);
        const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
  
        // Create an image element and load the SVG blob.
        const img = new Image();
        // If needed, set crossOrigin for external resources.
        img.crossOrigin = 'anonymous';
        img.onload = function () {
          // Create a canvas with dimensions matching the SVG.
          const canvas = document.createElement('canvas');
          canvas.width = clonedSvg.clientWidth || clonedSvg.getBoundingClientRect().width;
          canvas.height = clonedSvg.clientHeight || clonedSvg.getBoundingClientRect().height;
          const ctx = canvas.getContext('2d');
  
          // Draw the loaded image (which contains our SVG) onto the canvas.
          ctx.drawImage(img, 0, 0);
  
          // Convert the canvas content to a PNG data URL.
          const pngUrl = canvas.toDataURL('image/png');
          callback(pngUrl);
  
          // Clean up the temporary object URL.
          URL.revokeObjectURL(url);

          // Remove the hidden container from the document.
          document.body.removeChild(container);
        };
        img.src = url;
      })
      .catch(err => {
        console.error("Error inlining images:", err);
      });
  }
  
  /**
   * Recursively inlines computed styles from the source element to the cloned element.
   * This ensures that all external or inherited CSS styles are embedded directly.
   *
   * @param {Element} sourceElem - The original element.
   * @param {Element} clonedElem - The cloned element.
   */
   inlineAllStyles(sourceElem, clonedElem) {
    const computedStyle = window.getComputedStyle(sourceElem);
    let styleString = "";
    for (let i = 0; i < computedStyle.length; i++) {
      const key = computedStyle[i];
      const value = computedStyle.getPropertyValue(key);
      styleString += `${key}:${value};`;
    }
    clonedElem.setAttribute('style', styleString);
  
    // Process child elements recursively.
    for (let i = 0; i < sourceElem.children.length; i++) {
      this.inlineAllStyles(sourceElem.children[i], clonedElem.children[i]);
    }
  }
  
  /**
   * Finds all <image> elements within the SVG and inlines their external content
   * by fetching the resource and converting it to a data URL.
   *
   * @param {SVGElement} svg - The SVG element to process.
   * @return {Promise} A promise that resolves when all images have been inlined.
   */
  inlineImagesInSvg(svg) {
    const images = svg.querySelectorAll('image');
    const promises = [];
  
    images.forEach((image) => {
      // Retrieve the image source from either the "href" or "xlink:href" attribute.
      let href = image.getAttribute('href') || image.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
      if (!href) return;
      // Skip if already a data URL.
      if (href.startsWith('data:')) return;
  
      const p = fetch(href)
        .then(response => response.blob())
        .then(blob => new Promise<void>((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => {
            const dataUrl = reader.result;
            // Set both attributes for maximum browser compatibility.
            image.setAttribute('href', dataUrl);
            image.setAttributeNS('http://www.w3.org/1999/xlink', 'href', dataUrl);
            resolve();
          };
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        }))
        .catch(err => {
          console.error(`Failed to inline image ${href}:`, err);
        });
      promises.push(p);
    });
  
    return Promise.all(promises);
  }

  // @buttonBlock("Save PNG")
  // savePng() {
  //   const svgElement = document.querySelector("svg.blocklySvg");
  //   // Check if the SVG element exists
  //   if (!svgElement) {
  //     console.error("SVG element not found.");
  //     return;
  //   }

  //   this.svgToPng(svgElement, (pngUrl) => {
  //     const a = document.createElement('a');
  //     a.href = pngUrl;
  //     a.download = 'image.png';
  //     document.body.appendChild(a);
  //     a.click();
  //     document.body.removeChild(a);
  //   });
  // }
}