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
#include "esp_dsp.h"
//#include <DSPforAudioReactiveMCU.h>

//static bool debug_nn = false;  // Set this to true to see e.g. features generated from the raw signal
static bool record_status = true;


// //size needed for the mfcc buffer, seems to report just 1 x num filters
// matrix_size_t mfe_buffer_size = speechpy::feature::calculate_mfe_buffer_size(
//   EI_CLASSIFIER_SLICE_SIZE,
//   EI_CLASSIFIER_FREQUENCY,
//   ei_dsp_config_4.frame_length,
//   ei_dsp_config_4.frame_stride,
//   ei_dsp_config_4.num_filters,
//   ei_dsp_config_4.implementation_version);

// ei::matrix_t outputMatrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

/******New FFT stuff ------------------------------------------------------------ */

#define FFT_SIZE 256 //needs to be base 4 so we can use radix-4 fft, so next would be 4096
#define SAMPLE_RATE 16000
#define FFT_BIN_SPACING SAMPLE_RATE/FFT_SIZE
#define FFT_SLICE_SIZE FFT_SIZE//1024//800 is 20 fft runs per second;
//there's half as many complex as real outputs as complex has two components
#define FFT_SIZE_COMPLEX_OUTPUTS FFT_SIZE//1024//FFT_SLICE_SIZE / 2
//#define FFT_SLICE_SIZE 600 //30 fft runs per second;

float fftInOut[FFT_SIZE_COMPLEX_OUTPUTS*2];
float window[FFT_SIZE_COMPLEX_OUTPUTS];


static const uint32_t sample_buffer_size = FFT_SLICE_SIZE;
static signed short sampleBuffer[sample_buffer_size];

//inspired by
//https://forum.edgeimpulse.com/t/error-compiling-arduino-library-for-xiao-esp32s3-sense/8901
//kept becuase its need
/** Audio buffers, pointers and selectors */
typedef struct {
  signed short *buffers[2];
  unsigned char selectedBufferIndex;
  unsigned char selectedBufferReady;
  unsigned int selectedBufferLen;
} captureBuffers_t;

static captureBuffers_t captureBuf;
static bool record_ready = false;

/* NEOPIXEL STUFF -----------------------------------------------------------------*/
#include <NeoPixelBus.h>


//needs to be divisable by 2 with remainder 1 for symmetric cascading
#define NUMPIXELS 27//135// 23+ 17+ 13+ 20+ 23+ 17+ 19+ 3
int tentacles[] = { 23, 17, 13, 20, 23, 17, 19, 3};
int numTentacles = 8;
#define PIN_NEO_PIXEL 4  //pin 2 can cause issues with some voltage converter boards (esp won't go into flashing mode) (maybe a small pull down resistor would mitigate this)
NeoPixelBus<NeoRgbwFeature, NeoEsp32I2s0Sk6812Method> strip(NUMPIXELS, PIN_NEO_PIXEL);



/* GRAPH PLOTTING (no Arduino IDE and cutecom support, only puttY confimred so far)--------------------------------------*/
bool printGraph = false;
int nonPrintCycles = 0;
int printEvery = 30;
int graphMaxLength = 100.0;
///TODO: ENUM this
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

double inputScalar = 0.6;
float inputScalarHysteresis = 0.4;
float inputScalarMin = 0.5;
float inputScalarMax = 8.0;


int numCycles = 0;

int outputMode = SYMMETRIC_CASCADING;

struct RGBColour {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

RGBColour pixelArray[NUMPIXELS];

RGBColour pixelArrayOld[NUMPIXELS];

void stripBootAnimation();

/**
 * @brief      Arduino setup function
 */
void setup() {


  strip.Begin();
  strip.Show();
  // put your setup code here, to run once:
  Serial.begin(115200);
  //don't normally wait for serial
  while (!Serial)
  ;

  Serial.println("ESP32 Dragon");
  //initialize and test neoPixel

  stripBootAnimation();

  Serial.println("Boot animation finished");
  delay(2 * 1000); 
  strip.ClearTo(RgbColor(0,0,0));
  strip.Show();
  //FFT radix-4 init
  esp_err_t ret;
  ESP_ERROR_CHECK(ret = dsps_fft4r_init_fc32(NULL, FFT_SIZE_COMPLEX_OUTPUTS));
  if (ret  != ESP_OK) {
      Serial.printf("Not possible to initialize FFT4R. Error = %i", ret);
      return;
  }

  //calculate hann window
  dsps_wind_hann_f32(window, FFT_SIZE_COMPLEX_OUTPUTS);

  //calculate kernel for gaussian filter
  if(gaussianFilter.begin(SIGMA_FINAL_GAUSSIAN) != 0){
    Serial.println("Gaussian init FAILED!");
  }

  //start microphone recording
  startRecordingToBufferInNewThread(FFT_SLICE_SIZE);
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop() {
  //fill the buffer first, so we have data to begin with
  //capturing will continue while said data is processed
  bool result = waitUntilCaptureBufferFull();
  if (!result) {
    Serial.printf("ERR: Failed to record audio...\n");
    return;
  }
  //run signal processing
  runDSP();

  //show the output
  displayAnimation();
}

void runDSP(){
  static int notPrintedFor = 0;
  unsigned int start_dsp = dsp_get_cpu_cycle_count();
  //apply hamming window and copy over "old" data
  for(u16_t i = 0; i < FFT_SLICE_SIZE; i++){
    //even indices (and 0) get the value
    fftInOut[i*2] = captureBuf.buffers[captureBuf.selectedBufferIndex ^ 1][i] * window [i];
    //odd indices are the complex part and thus 0
    fftInOut[i*2+1] = 0;
  }
  esp_err_t ret;
  //run FFT
  ret = dsps_fft4r_fc32(fftInOut, FFT_SIZE_COMPLEX_OUTPUTS);
  if (ret  != ESP_OK) {
    Serial.printf("FAILED to run FFT4R. Error = %i", ret);
    return;
  }
  // Bit reverse
  ret = dsps_bit_rev4r_fc32(fftInOut, FFT_SIZE_COMPLEX_OUTPUTS);
  if (ret  != ESP_OK) {
    Serial.printf("FAILED to run dsps_bit_rev4r_fc32. Error = %i", ret);
    return;
  }
  // Convert one complex vector with length N/2 to one real spectrum vector with length N/2
  ret = dsps_cplx2real_fc32(fftInOut, FFT_SIZE_COMPLEX_OUTPUTS);
  if (ret  != ESP_OK) {
    Serial.printf("FAILED to run dsps_cplx2real_fc32. Error = %i", ret);
    return;
  }

  unsigned int end_dsp = dsp_get_cpu_cycle_count();
  notPrintedFor++;
  if(notPrintedFor > 20){
  Serial.printf("DSP took: %f us / %i ticks\n", (end_dsp - start_dsp) /240.0, (end_dsp - start_dsp) );
  //using FFT4R @ 256 DSP took: 113.325000 us / 27198 ticks
  
  notPrintedFor = 0;
  }
}

void displayAnimation() {

  // signal_t signal;
  // signal.total_length = FFT_SLICE_SIZE;
  // signal.get_data = &microphone_audio_signal_get_data;
  // ei_impulse_result_t result = { 0 };

  // if (!fftInOut) {
  //   Serial.printf("allocation of output matrix failed\n");
  // }
  // run_mfe_maaajaaa(&signal, &outputMatrix, debug_nn);

  double rMax = -100.0;
  int rMaxIndex = -1;
  double gMax = -100.0;
  int gMaxIndex = -1;
  double bMax = -100.0;
  int bMaxIndex = -1;

  double maxOf4s[numTentacles] = {-100.0};

  //relevant buffer area where the mfcc output is stored
  int relevantBuferCols = FFT_SIZE;

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
  //intersting output is at the end of the matrix, see ei_run_dsp.h:833
  if(debug_arduino_filtering) Serial.print("\n\nData: ");

  for (int i = FFT_SIZE - relevantBuferCols; i < FFT_SIZE; i++) {
    if(debug_arduino_filtering) Serial.printf("%5.2f ", log(fftInOut[i]));
    if(fftInOut[i] > maxOf4s[i%8]){
      maxOf4s[i%8] = fftInOut[i];
    }
    //find maxima of the thrids of the spectrum
    if (i < firstThird) {
      if (fftInOut[i] > rMax) {
        rMax = fftInOut[i];
        rMaxIndex = i;
      }
    } else if (i < secondThird) {
      if (fftInOut[i] > gMax) {
        gMax = fftInOut[i];
        gMaxIndex = i;
      }
    } else {
      if (fftInOut[i] > bMax) {
        bMax = fftInOut[i];
        bMaxIndex = i;
      }
    }

    //print graph bar
    if (printGraph && nonPrintCycles >= printEvery) {
      for (int j = 0; j < round(graphMaxLength * fftInOut[i]); j++) {
        Serial.print("▮");
      }
      Serial.println();
    }
    if (i == ceptrumToShow && outputMode == SINGLE_CEPTRUM) {
      for (int j = 0; j < NUMPIXELS; j++) {
        if (j <= round(NUMPIXELS * fftInOut[i])) {
          strip.SetPixelColor(j,RgbColor(255, 0, 0));
          //pixels.setPixelColor(j, 255, 0, 0);
        } else {
          strip.SetPixelColor(j,RgbColor(0, 0, 0));
          //pixels.setPixelColor(j, 0, 0, 0);
        }
      }
    }
  }
  //if (!printGraph)
    //Serial.print("\n");

  if (printGraph && nonPrintCycles >= printEvery) {
    nonPrintCycles = 0;
  } else {
    nonPrintCycles++;
  }

  //calculate new 8-bit rbg values, assuming mfcc output is normed to 0..1
  uint8_t rNew = static_cast<uint8_t>((20.0 + log(rMax)) * inputScalar);
  uint8_t gNew = static_cast<uint8_t>((20.0 + log(gMax)) * inputScalar);
  uint8_t bNew = static_cast<uint8_t>((20.0 + log(bMax)) * inputScalar);

  ///TODO: Figure out what this was intended for and if we need it maybe
  // if (rMax < 0.4) {
  //   rNew = 0;
  // }
  // if (gMax < 0.5) {
  //   gNew = 0;
  // }
  // if (bMax < 0.3) {
  //   bNew = 0;
  // }

  if (!printGraph && debug_arduino_filtering) {

    Serial.print("new rgb: ");
    Serial.print(rNew);
    Serial.print(" ");
    Serial.print(gNew);
    Serial.print(" ");
    Serial.println(bNew);

    Serial.print("max rgb: ");
    Serial.print(log(rMax));
    Serial.print(" ");
    Serial.print(log(gMax));
    Serial.print(" ");
    Serial.println(log(bMax));

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
    {
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

    strip.SetPixelColor(i,RgbColor(reds[i], greens[i],  blues[i]));
  }
  
  // Send the updated pixel colors to the hardware.
  strip.Show();

  /*------------gain adjustment--------------------------------------*/
  //add new values to rolling average
  updateRollingAverage(sqrt(rNew*rNew + gNew*gNew + bNew*bNew));

  /*Serial.print("Rolling average: ");
  Serial.print(rollingPeakAvg);
  Serial.print(" gain: ");
  Serial.print(gain);
  Serial.print(" inputScalar: ");
  Serial.println(inputScalar);*/
  if (rollingPeakAvg < minimumAverage) {
    inputScalar += inputScalarHysteresis;
    if (inputScalar > inputScalarMax) {
      inputScalar = inputScalarMax;
    }
    Serial.print("increasing sclae to: ");
    Serial.println(inputScalar);
    //reset avg to take some time for adjustment
    rollingPeakAvg = 128.0;
  }

  if (rollingPeakAvg > maximumAverage) {
    inputScalar -= inputScalarHysteresis;
    if (inputScalar < inputScalarMin) {
      inputScalar = inputScalarMin;
    }
    Serial.print("decreasing scalar to: ");
    Serial.println(inputScalar);
    //reset avg to take some time for adjustment
    rollingPeakAvg = 128.0;
  }
}

static void audio_inference_callback(uint32_t n_bytes) {
  for (int i = 0; i < n_bytes >> 1; i++) {
    captureBuf.buffers[captureBuf.selectedBufferIndex][captureBuf.selectedBufferLen++] = sampleBuffer[i];

    if (captureBuf.selectedBufferLen >= FFT_SLICE_SIZE) {
      captureBuf.selectedBufferIndex ^= 1;
      captureBuf.selectedBufferLen = 0;
      captureBuf.selectedBufferReady = 1;
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
      Serial.printf("Error in I2S read : %d", bytes_read);
    } else {
      if (bytes_read < i2s_bytes_to_read) {
        Serial.printf("Partial I2S read");
      }

      // scale the data (otherwise the sound is too quiet)
      for (int x = 0; x < i2s_bytes_to_read / 2; x++) {
        sampleBuffer[x] = (int16_t)(sampleBuffer[x]) * 8;
      }
      audio_inference_callback(i2s_bytes_to_read);
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
static bool startRecordingToBufferInNewThread(uint32_t n_samples) {
  captureBuf.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

  if (captureBuf.buffers[0] == NULL) {
    return false;
  }

  captureBuf.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

  if (captureBuf.buffers[1] == NULL) {
    free(captureBuf.buffers[0]);
    return false;
  }

  captureBuf.selectedBufferIndex = 0;
  captureBuf.selectedBufferLen = 0;
  captureBuf.selectedBufferReady = 0;

  if (i2s_init(SAMPLE_RATE)) {
    Serial.printf("Failed to start I2S!");
  }

  delay(100);

  record_status = true;

  xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void *)sample_buffer_size, 10, NULL);

  return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool waitUntilCaptureBufferFull(void) {
  bool ret = true;

  if (captureBuf.selectedBufferReady == 1) {
    Serial.printf(
      "Error sample buffer overrun. Decrease the number of slices per model window\n");
    ret = false;
  }

  while (captureBuf.selectedBufferReady == 0) {
    delay(1);
  }

  captureBuf.selectedBufferReady = 0;
  return true;
}

/**
 * Get raw audio signal data
 */
// static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
//   numpy::int16_to_float(&captureBuf.buffers[captureBuf.selectedBufferIndex ^ 1][offset], out_ptr, length);

//   return 0;
// }

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void) {
  i2s_deinit();
  free(captureBuf.buffers[0]);
  free(captureBuf.buffers[1]);
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
    Serial.printf("Error in i2s_driver_install");
  }

  ret = i2s_set_pin((i2s_port_t)1, &pin_config);
  if (ret != ESP_OK) {
    Serial.printf("Error in i2s_set_pin");
  }

  ret = i2s_zero_dma_buffer((i2s_port_t)1);
  if (ret != ESP_OK) {
    Serial.printf("Error in initializing dma buffer with 0");
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

void stripBootAnimation(){
    if (NUMPIXELS % 2 != 1 && outputMode == SYMMETRIC_CASCADING) {
    //turn strip red
    for (int i = NUMPIXELS; i > 0; i--) {
      RgbColor redColor(255, 0, 0);
      strip.SetPixelColor(i, redColor);
    }
    strip.Show();
    //wait for serial
    while (!Serial)
      ;
    //do not run the rest of the code
    while (1) {
      Serial.println("ERROR: NUMPIXELS must be an ueneven number");
      delay(2000);
    }
  }
  
  RgbColor testColor(128, 0, 128);
  strip.SetPixelColor(1, testColor);
  for (int j = 0; j < NUMPIXELS; j++) {
    for (int i = NUMPIXELS; i > 0; i--) {
      strip.SetPixelColor(i, testColor);
    }
    strip.Show();
    delay(10);
  }
  strip.ClearTo(RgbColor(0,0,0));
  RgbColor evenTent(128, 0, 0);
  RgbColor oddTent(0, 128, 0);
  for(int i = 0; i < NUMPIXELS; i ++){
    if(i % 8 == 0){
      strip.SetPixelColor(i, RgbColor(255,0,0));
    }else if(i % 8 == 1){
      strip.SetPixelColor(i, RgbColor(0,255,0));
    }else if(i % 8 == 2){
      strip.SetPixelColor(i, RgbColor(0,0,255));
    }else if(i % 8 == 3){
      strip.SetPixelColor(i, RgbColor(255,255,0));
    }else if(i % 8 == 4){
      strip.SetPixelColor(i, RgbColor(0,255,255));
    }else if(i % 8 == 5){
      strip.SetPixelColor(i, RgbColor(255,0,255));
    }else if(i % 8 == 6){
      strip.SetPixelColor(i, RgbColor(255,0,0));
    }else if(i % 8 == 7){
      strip.SetPixelColor(i, RgbColor(128,0,255));
    }
  }

  strip.Show();
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
