import { Component, ViewChild, ElementRef } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

export interface DetectedObject {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'ml-demo';
  @ViewChild('videoCamera', { static: true }) videoCamera: ElementRef;
  @ViewChild('canvas', { static: true }) canvas: ElementRef;

  model: tf.GraphModel;
  ngOnViewInit() {
    this.webcam_init();
    this.loadModel();
  }

  webcam_init() {
    navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'environment',
      }
    }).then(stream => {
      this.videoCamera.nativeElement.srcObject = stream;
      this.videoCamera.nativeElement.onloadedmetadata = () => {
        this.videoCamera.nativeElement.play();
      };
    });
  }

  async loadModel() {
    const modelURL = '../assets/web/model.json';
    this.model = await tf.loadGraphModel(modelURL);
    const result = await this.model.executeAsync(tf.zeros([1, 300, 300, 3])) as any;
    await Promise.all(result.map(t => t.data()));
    result.map(t => t.dispose());
    this.predictImages(this.videoCamera.nativeElement, this.model);
  }

  async predictImages(video, model) {

    const maxNumBoxes = 30;
    const batched = tf.tidy(() => {
      let img = this.canvas.nativeElement;
      if (!(img instanceof tf.Tensor)) {
        img = tf.browser.fromPixels(img);
        img = tf.cast(img, 'float32');
      }
      return img.expandDims(0);
    });
    const height = batched.shape[1];
    const width = batched.shape[2];
    const result = await this.model.executeAsync(batched) as tf.Tensor[];
    const scores = result[0].dataSync() as Float32Array;
    const boxes = result[1].dataSync() as Float32Array;

    batched.dispose();
    tf.dispose(result);

    const [maxScores, classes] = this.calculateMaxiumScore(scores, result[0].shape[1], result[0].shape[2]);
    const tensorIndex = tf.tidy(() => {
      const boxes2 = tf.tensor2d(boxes, [result[1].shape[1], result[1].shape[3]]);
      return tf.image.nonMaxSuppression(boxes2, maxScores, maxNumBoxes, 0.5, 0.5);
    });
    const index = tensorIndex.dataSync() as Float32Array;
    tensorIndex.dispose();
    const output = this.generateOutputObject(width, height, boxes, maxScores, index, classes);
    if (output.length > 0) {
      this.renderPredictions(output);
    } else {
      this.renderCameraVideo();
    }
    requestAnimationFrame(() => {
      this.predictImages(video, model);
    });

  }


  generateOutputObject(width: number, height: number, boxes: Float32Array, scores: number[], index: Float32Array, classes: number[]): DetectedObject[] {

    const count = index.length;
    const objects: DetectedObject[] = [];
    for (let i = 0; i < count; i++) {
      const bbox = [];
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[index[i] * 4 + j];
      }
      const minY = bbox[0] * height;
      const minX = bbox[1] * width;
      const maxY = bbox[2] * height;
      const maxX = bbox[3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;
      objects.push({
        bbox: bbox as [number, number, number, number],
        class: (classes[index[i]] + 1).toString(),
        score: scores[index[i]]
      });
    }
    return objects;
  }

  // function is defined to find the optimum max score.
  calculateMaxiumScore(scores: Float32Array, numBoxes: number, numClasses: number) {

    const maxes = [];
    const classes = [];
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE;
      let index = -1;
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j];
          index = j;
        }
      }
      maxes[i] = max;
      classes[i] = index;
    }
    return [maxes, classes];
  }

  // function is defined to draw bbox on the canvas
  renderPredictions(predictions: any) {
    const canvas = this.canvas.nativeElement;
    const ctx = canvas.getContext("2d");
    canvas.width = 350;
    canvas.height = 450;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    ctx.textBaseline = "top";
    ctx.drawImage(this.videoCamera.nativeElement, 0, 0, 350, 450);
    const data = canvas.toDataURL('image/png');

    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      const width = prediction.bbox[2];
      const height = prediction.bbox[3];
      // Draw the bounding box.
      if (prediction.class) {
        ctx.strokeStyle = "#FF0000";
        ctx.fillStyle = "#FF0000";
      }
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.rect(x, y, width, height);
      ctx.stroke();
      // Draw the label background.
      const textWidth = ctx.measureText(prediction.class).width;
      const textHeight = parseInt("16px sans-serif", 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    // for rendering class name over the rect
    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      if (prediction.class) {
        ctx.fillStyle = "#FFFFFF";
      }
      ctx.fillText(prediction.class, x, y);
    });

  }

  // render the video if no prediction is found
  renderCameraVideo() {
    const canvas = this.canvas.nativeElement;
    const ctx = canvas.getContext('2d');
    canvas.width = 350;
    canvas.height = 450;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(this.videoCamera.nativeElement, 0, 0, 350, 450);

  }



}














