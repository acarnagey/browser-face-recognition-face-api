import * as faceapi from "face-api.js";
import $ from "jquery";

(async () => {
    await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
})();

let faceMatcher = null;

let refImgUploadInput = document.getElementById("refImgUploadInput");
let queryImgUploadInput = document.getElementById("queryImgUploadInput");

// let refImg = document.getElementById("refImg");

refImgUploadInput.addEventListener(
  "change",
  async (e) => {
    const imgFile = $("#refImgUploadInput").get(0).files[0];
    const img = await faceapi.bufferToImage(imgFile);
    $("#refImg").get(0).src = img.src;
    updateReferenceImageResults();
  },
  false
);
queryImgUploadInput.addEventListener(
  "change",
  async (e) => {
    const imgFile = $("#queryImgUploadInput").get(0).files[0];
    const img = await faceapi.bufferToImage(imgFile);
    $("#queryImg").get(0).src = img.src;
    updateQueryImageResults();
  },
  false
);

async function updateReferenceImageResults() {
  const inputImgEl = $("#refImg").get(0);
  const canvas = $("#refImgOverlay").get(0);
  console.log(faceapi.nets)

//   debugger;
  const options = getFaceDetectorOptions();
  const fullFaceDescriptions = await faceapi.detectAllFaces(inputImgEl, options).withFaceLandmarks().withFaceDescriptors();
  if (!fullFaceDescriptions.length) {
    return;
  }

  // create FaceMatcher with automatically assigned labels
  // from the detection results for the reference image
  faceMatcher = new faceapi.FaceMatcher(fullFaceDescriptions);

  faceapi.matchDimensions(canvas, inputImgEl);
  // resize detection and landmarks in case displayed image is smaller than
  // original size
  const resizedResults = faceapi.resizeResults(
    fullFaceDescriptions,
    inputImgEl
  );
  // draw boxes with the corresponding label as text
  const labels = faceMatcher.labeledDescriptors.map((ld) => ld.label);
  resizedResults.forEach(({ detection, descriptor }) => {
    const label = faceMatcher.findBestMatch(descriptor).toString();
    const options = { label };
    const drawBox = new faceapi.draw.DrawBox(detection.box, options);
    drawBox.draw(canvas);
  });
}

async function updateQueryImageResults() {
    if (!faceMatcher) {
      return
    }

    const inputImgEl = $('#queryImg').get(0)
    const canvas = $('#queryImgOverlay').get(0)

    const results = await faceapi
      .detectAllFaces(inputImgEl, getFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptors()

    faceapi.matchDimensions(canvas, inputImgEl)
    // resize detection and landmarks in case displayed image is smaller than
    // original size
    const resizedResults = faceapi.resizeResults(results, inputImgEl)

    resizedResults.forEach(({ detection, descriptor }) => {
      const label = faceMatcher.findBestMatch(descriptor).toString()
      const options = { label }
      const drawBox = new faceapi.draw.DrawBox(detection.box, options)
      drawBox.draw(canvas)
    })
  }

// --- face dectection controls

const SSD_MOBILENETV1 = "ssd_mobilenetv1";
const TINY_FACE_DETECTOR = "tiny_face_detector";

let selectedFaceDetector = SSD_MOBILENETV1;

// ssd_mobilenetv1 options
let minConfidence = 0.5;

// tiny_face_detector options
let inputSize = 512;
let scoreThreshold = 0.5;

function getFaceDetectorOptions() {
  return selectedFaceDetector === SSD_MOBILENETV1
    ? new faceapi.SsdMobilenetv1Options({ minConfidence })
    : new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold });
}
