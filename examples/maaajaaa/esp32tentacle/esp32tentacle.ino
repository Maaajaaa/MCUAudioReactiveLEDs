/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License forM the specific language governing permissions and
 * limitations under the License.
 *
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK 0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 1

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s.h"
#include <DSPforAudioReactiveMCU.h>


//https://forum.edgeimpulse.com/t/error-compiling-arduino-library-for-xiao-esp32s3-sense/8901

/** Audio buffers, pointers and selectors */
typedef struct {
  signed short *buffers[2];
  unsigned char buf_select;
  unsigned char buf_ready;
  unsigned int buf_count;
  unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
//static signed short *sampleBuffer;
static bool debug_nn = false;  // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);
static bool record_status = true;

static const uint32_t sample_buffer_size = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
static signed short sampleBuffer[sample_buffer_size];

//size needed for the mfcc buffer, seems to report just 1 x num filters
matrix_size_t mfe_buffer_size = speechpy::feature::calculate_mfe_buffer_size(
  EI_CLASSIFIER_SLICE_SIZE,
  EI_CLASSIFIER_FREQUENCY,
  ei_dsp_config_4.frame_length,
  ei_dsp_config_4.frame_stride,
  ei_dsp_config_4.num_filters,
  ei_dsp_config_4.implementation_version);

ei::matrix_t outputMatrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);



/* NEOPIXEL STUFF -----------------------------------------------------------------*/
#include <Adafruit_NeoPixel.h>

#define NUMPIXELS 135// 23 + 19 + 17 + 23 + 21 + 23 + 5 + 3 //+1 because needs to be uneven  // limited because the neopixel writing is rather slow, needs to be threaded
//needs to be divisable by 2 with remainder 1
#define PIN_NEO_PIXEL 2  // for some reason the pin mapping does not exaxtly match that printed
Adafruit_NeoPixel pixels(NUMPIXELS, PIN_NEO_PIXEL, NEO_GRB + NEO_KHZ800);


/* GRAPH PLOTTING (no Arduino IDE and cutecom support, only puttY confimred so far)--------------------------------------*/
bool printGraph = false;
int nonPrintCycles = 0;
int printEvery = 30;
int graphMaxLength = 100.0;

#define LINE_CASCADING 2
#define SYMMETRIC_CASCADING 3
#define SINGLE_CEPTRUM 1
int ceptrumToShow = 0;

/* Low-Pass Filter Constants --------------------------------------------------- */


float alphaLowPass = 0.6;
static bool debug_arduino_filtering = false;

/* Gaussian Filter ------------------------------------------------------------- */


//obtained from https://github.com/Maaajaaa/Gaussian_filter_1D
#include <GaussianFilter1D.h>

//start filter in cached mode to increase computation speed
GaussianFilter1D gaussianFilter = GaussianFilter1D(true);

#define SIGMA_FINAL_GAUSSIAN 0.2

/* Rolling Average ---------------------------------------------------------------*/

// should be around 2-3 seconds
int ravSamplesize = 60;

//lets stat with all values high so we can use the output directly for gain adjustment
float rollingPeakAvg = 128.0;

//obtained from PDM docs https://docs.arduino.cc/learn/built-in-libraries/pdm/#setgain
const int maxGain = 255;
const int minGain = 0;


//increase gain when the average drops below this
float minimumAverage = 50.0;
float maximumAverage = 150.0;

int gain = 128;
int gainHysteresis = 20;

float inputScalar = 0.6;
float inputScalarHysteresis = 0.2;
float inputScalarMin = 0.5;
float inputScalarMax = 4.0;


int numCycles = 0;

int outputMode = SYMMETRIC_CASCADING;

struct RGBColour {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

RGBColour pixelArray[NUMPIXELS];

RGBColour pixelArrayOld[NUMPIXELS];

void updateBatteryLevel(bool);

/**
 * @brief      Arduino setup function
 */
void setup() {

  pixels.begin();
  pixels.setBrightness(255);
  // put your setup code here, to run once:
  Serial.begin(115200);
  //don't normally wait for serial
  //while (!Serial)
  //  ;
  //initialize and test neoPixel

  if (NUMPIXELS % 2 != 1 && outputMode == SYMMETRIC_CASCADING) {
    //turn strip red
    for (int i = NUMPIXELS; i > 0; i--) {
      pixels.setPixelColor(i, 255, 0, 0);
    }
    pixels.show();
    //wait for serial
    while (!Serial)
      ;
    //do not run the rest of the code
    while (1) {
      Serial.println("ERROR: NUMPIXELS must be an ueneven number");
      delay(2000);
    }
  }
  pixels.setPixelColor(1, 0, 0, 255);
  for (int j = 0; j < NUMPIXELS; j++) {
    for (int i = NUMPIXELS; i > 0; i--) {
      pixels.setPixelColor(i, pixels.getPixelColor(i - 1));
    }
    pixels.show();
    delay(50);
  }

  pixels.clear();


  //calculate kernel for gaussian filter
  gaussianFilter.begin(SIGMA_FINAL_GAUSSIAN);

  Serial.println("Edge Impulse Inferencing Demo");

  // summary of inferencing settings (from model_metadata.h)
  ei_printf("Inferencing settings:\n");
  ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
  ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
  ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));
  ei_printf("\tNumber of NN_Input: %d\n", EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);
  ei_printf("\tIdeal output size: %dx%d\n", mfe_buffer_size.cols, mfe_buffer_size.rows);

  run_classifier_init();
  ei_printf("classifier initialized");
  if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
    ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    return;
  }
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
  displayAnimation();
}

void displayAnimation() {

  bool m = microphone_inference_record();
  if (!m) {
    ei_printf("ERR: Failed to record audio...\n");
    return;
  }

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
  signal.get_data = &microphone_audio_signal_get_data;
  ei_impulse_result_t result = { 0 };

  if (!outputMatrix.buffer) {
    ei_printf("allocation of output matrix failed\n");
  }
  run_mfcc_maaajaaa(&signal, &outputMatrix, debug_nn);

  double rMax = -100.0;
  int rMaxIndex = -1;
  double gMax = -100.0;
  int gMaxIndex = -1;
  double bMax = -100.0;
  int bMaxIndex = -1;

  //relevant buffer area where the mfcc output is stored
  int relevantBuferCols = mfe_buffer_size.cols;

  int firstThird, secondThird;
  firstThird = 3;
  secondThird = 5;


  //print graph, can't be viewed in arduino viewer, but putty or cutecom do support the clear screen command
  //stackoverflow.com/a/15559322
  if (printGraph && nonPrintCycles >= printEvery) {
    Serial.write(27);     //ESC
    Serial.print("[2J");  //clear screen
    Serial.write(27);     //ESC
    Serial.print("[H");   //cursor to home
  }
  if (!printGraph) {
    //Serial.print("output: ");
  }
  for (int i = 0; i < relevantBuferCols; i++) {
    //find maxima of the thrids of the spectrum
    if (i < firstThird) {
      if (outputMatrix.buffer[i] > rMax) {
        rMax = outputMatrix.buffer[i];
        rMaxIndex = i;
      }
    } else if (i < secondThird) {
      if (outputMatrix.buffer[i] > gMax) {
        gMax = outputMatrix.buffer[i];
        gMaxIndex = i;
      }
    } else {
      if (outputMatrix.buffer[i] > bMax) {
        bMax = outputMatrix.buffer[i];
        bMaxIndex = i;
      }
    }

    if (!printGraph /*&& i < 20*/) {
      //Serial.print(sampleBuffer[i]);
      Serial.print(" ");
    }

    //print graph bar
    if (printGraph && nonPrintCycles >= printEvery) {
      for (int j = 0; j < round(graphMaxLength * outputMatrix.buffer[i]); j++) {
        Serial.print("▮");
      }
      Serial.println();
    }
    if (i == ceptrumToShow && outputMode == SINGLE_CEPTRUM) {
      for (int j = 0; j < NUMPIXELS; j++) {
        if (j <= round(NUMPIXELS * outputMatrix.buffer[i])) {
          pixels.setPixelColor(j, 255, 0, 0);
        } else {
          pixels.setPixelColor(j, 0, 0, 0);
        }
      }
    }
  }
  if (!printGraph)
    Serial.print("\n");

  if (printGraph && nonPrintCycles >= printEvery) {
    nonPrintCycles = 0;
  } else {
    nonPrintCycles++;
  }

  //calculate new 8-bit rbg values, assuming mfcc output is normed to 0..1
  int rNew = pow(rMax, 2) * inputScalar;
  int gNew = pow(gMax, 2) * inputScalar;
  int bNew = pow(bMax, 2) * inputScalar;

  if (rMax < 0.4) {
    rNew = 0;
  }
  if (gMax < 0.5) {
    gNew = 0;
  }
  if (bMax < 0.3) {
    bNew = 0;
  }

  if (!printGraph && debug_arduino_filtering) {

    Serial.print("new rgb: ");
    Serial.print(rNew);
    Serial.print(" ");
    Serial.print(gNew);
    Serial.print(" ");
    Serial.println(bNew);

    Serial.print("max rgb: ");
    Serial.print(rMax);
    Serial.print(" ");
    Serial.print(gMax);
    Serial.print(" ");
    Serial.println(bMax);

    Serial.print("max rgb index: ");
    Serial.print(rMaxIndex);
    Serial.print(" ");
    Serial.print(gMaxIndex);
    Serial.print(" ");
    Serial.println(bMaxIndex);
  }
  //make copy of pixel array (needed for filtering only)
  std::copy(pixelArray, pixelArray + NUMPIXELS, pixelArrayOld);
  switch (outputMode) {

    case LINE_CASCADING:
      //cascade
      for (int i = NUMPIXELS; i > 0; i--) {
        pixelArray[i] = pixelArray[i - 1];
      }
      //set 0th pixel
      pixelArray[0] = { rNew, gNew, bNew };
      break;

    case SYMMETRIC_CASCADING:

      int centerPixel = NUMPIXELS / 2 + 1;
      //center to left cascading
      for (int i = NUMPIXELS; i > centerPixel; i--) {
        pixelArray[i] = pixelArray[i - 1];
      }
      //right to center cascading
      for (int i = 0; i < centerPixel; i++) {
        pixelArray[i] = pixelArray[i + 1];
      }
      //set center pixel
      pixelArray[centerPixel] = { rNew, gNew, bNew };
      break;
  }
  //apply filter and apply array to pixels
  //caching arrays to use 3 separate 1d gaussian blur fliters
  float reds[NUMPIXELS];
  float greens[NUMPIXELS];
  float blues[NUMPIXELS];
  for (int i = 0; i < NUMPIXELS; i++) {
    //apply low pass filter
    RGBColour filteredCol = lowPassFilterRGB(pixelArray[i], pixelArrayOld[i]);
    reds[i] = filteredCol.r;
    greens[i] = filteredCol.g;
    blues[i] = filteredCol.b;
  }

  //apply gaussian filter for each colour

  //gaussianFilter.filter(reds, NUMPIXELS);
  //gaussianFilter.filter(greens, NUMPIXELS);
  //gaussianFilter.filter(blues, NUMPIXELS);

  for (int i = 0; i < NUMPIXELS; i++) {
    pixels.setPixelColor(i, reds[i], greens[i], blues[i]);
  }

  pixels.show();  // Send the updated pixel colors to the hardware.

  /*------------gain adjustment--------------------------------------*/
  //add new values to rolling average

  //updateRollingAverage(sqrt(rNew*rNew + gNew*gNew + bNew*bNew));

  /*Serial.print("Rolling average: ");
  Serial.print(rollingPeakAvg);
  Serial.print(" gain: ");
  Serial.print(gain);
  Serial.print(" inputScalar: ");
  Serial.println(inputScalar);*/
  if (rollingPeakAvg < minimumAverage) {
    gain += gainHysteresis;
    if (gain > maxGain) {
      gain = maxGain;
      inputScalar += inputScalarHysteresis;
      if (inputScalar > inputScalarMax) {
        inputScalar = inputScalarMax;
      }
    }
    Serial.print("increasing gain to: ");
    Serial.println(gain);
    //reset avg to take some time for adjustment
    rollingPeakAvg = 128.0;
  }

  if (rollingPeakAvg > maximumAverage) {
    gain -= gainHysteresis;
    if (gain < minGain) {
      gain = minGain;
      inputScalar = inputScalarMin;
    }
    Serial.print("decreasing gain to: ");
    Serial.println(gain);
    //reset avg to take some time for adjustment
    rollingPeakAvg = 128.0;
  }
}

static void audio_inference_callback(uint32_t n_bytes) {
  for (int i = 0; i < n_bytes >> 1; i++) {
    inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

    if (inference.buf_count >= inference.n_samples) {
      inference.buf_select ^= 1;
      inference.buf_count = 0;
      inference.buf_ready = 1;
    }
  }
}

static void capture_samples(void *arg) {

  const int32_t i2s_bytes_to_read = (uint32_t)arg;
  size_t bytes_read = i2s_bytes_to_read;

  while (record_status) {

    /* read data at once from i2s */
    i2s_read((i2s_port_t)1, (void *)sampleBuffer, i2s_bytes_to_read, &bytes_read, 100);

    if (bytes_read <= 0) {
      ei_printf("Error in I2S read : %d", bytes_read);
    } else {
      if (bytes_read < i2s_bytes_to_read) {
        ei_printf("Partial I2S read");
      }

      // scale the data (otherwise the sound is too quiet)
      for (int x = 0; x < i2s_bytes_to_read / 2; x++) {
        sampleBuffer[x] = (int16_t)(sampleBuffer[x]) * 8;
      }

      if (record_status) {
        audio_inference_callback(i2s_bytes_to_read);
      } else {
        break;
      }
    }
  }
  vTaskDelete(NULL);
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples) {
  inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

  if (inference.buffers[0] == NULL) {
    return false;
  }

  inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

  if (inference.buffers[1] == NULL) {
    ei_free(inference.buffers[0]);
    return false;
  }

  inference.buf_select = 0;
  inference.buf_count = 0;
  inference.n_samples = n_samples;
  inference.buf_ready = 0;

  if (i2s_init(EI_CLASSIFIER_FREQUENCY)) {
    ei_printf("Failed to start I2S!");
  }

  ei_sleep(100);

  record_status = true;

  xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void *)sample_buffer_size, 10, NULL);

  return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void) {
  bool ret = true;

  if (inference.buf_ready == 1) {
    ei_printf(
      "Error sample buffer overrun. Decrease the number of slices per model window "
      "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
    ret = false;
  }

  while (inference.buf_ready == 0) {
    delay(1);
  }

  inference.buf_ready = 0;
  return true;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

  return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void) {
  i2s_deinit();
  ei_free(inference.buffers[0]);
  ei_free(inference.buffers[1]);
}


static int i2s_init(uint32_t sampling_rate) {
  // Start listening for audio: MONO @ 8/16KHz
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
    .sample_rate = sampling_rate,
    .bits_per_sample = (i2s_bits_per_sample_t)16,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = -1,
  };
  i2s_pin_config_t pin_config = {
    .bck_io_num = 5,     // IIS_SCLK
    .ws_io_num = 27,     // IIS_LCLK
    .data_out_num = -1,  // IIS_DSIN
    .data_in_num = 26,   // IIS_DOUT
  };
  esp_err_t ret = 0;

  ret = i2s_driver_install((i2s_port_t)1, &i2s_config, 0, NULL);
  if (ret != ESP_OK) {
    ei_printf("Error in i2s_driver_install");
  }

  ret = i2s_set_pin((i2s_port_t)1, &pin_config);
  if (ret != ESP_OK) {
    ei_printf("Error in i2s_set_pin");
  }

  ret = i2s_zero_dma_buffer((i2s_port_t)1);
  if (ret != ESP_OK) {
    ei_printf("Error in initializing dma buffer with 0");
  }

  return int(ret);
}

static int i2s_deinit(void) {
  i2s_driver_uninstall((i2s_port_t)1);  //stop & destroy i2s driver
  return 0;
}


//Tf is the filter time constant
//Ts ist the sampling time
float lowPassFilter(float alpha, float y, float y_prev) {
  //based on simple FOC equation https://docs.simplefoc.com/low_pass_filter
  // calculate the filtering
  //float alpha = Tf/(Tf + Ts);
  return alpha * y_prev + (1.0f - alpha) * y;
}



RGBColour lowPassFilterRGB(RGBColour rgbColourCurrent, RGBColour rgbColourLast) {
  RGBColour rgbColour;
  rgbColour.r = lowPassFilter(alphaLowPass, rgbColourCurrent.r, rgbColourLast.r);
  rgbColour.g = lowPassFilter(alphaLowPass, rgbColourCurrent.g, rgbColourLast.g);
  rgbColour.b = lowPassFilter(alphaLowPass, rgbColourCurrent.b, rgbColourLast.b);
  return rgbColour;
}

float updateRollingAverage(float newVal) {
  rollingPeakAvg -= rollingPeakAvg / ravSamplesize;
  rollingPeakAvg += newVal / ravSamplesize;
  return rollingPeakAvg;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
