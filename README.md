# `AIClass`
This is an orchestrator class. This class works in conjunction with the [`AudioModel`](#`class-AudioModel`), which handles audio recording and target recognition functions.

*TODO: Continue to DOC.*


## `class AudioModel`

`AudioModel` is a thread-safe class works to access microphone, to record audio data and provide inference funcitons it on `torch::jit`.

### Example Usage:
``` 
AudioModel* audio_model = new AudioModel("path/to/model.pt", 16000, 8);
audio_model->start_recording(); // Starts the voice recording.
audio_model->stop_recording();  // Stops the voice recording.

// Access to the recorded data
std::vector<float>* audio_data = audio_model->get_audio_buffer();

// Access info if voice recording
bool audio_recording = audio_model->get_is_recording();

// Convert to torch tensor from vector
torch::Tensor audio_tensor = audio_model->toTorchTensor();

// Inference on torch::jit
torch::Tensor result = audio_model->inference(audio_tensor);

// Access to targets 
std::vector<bool>& targets = audio_model->get_targets();
```

### Class Members
#### Variables
- `is_recording`: Holds a boolean if the model recording.\
- `audio_buffer`: Holds recorded data.\
- `model`: Pretrained Audio recognition model.\
- `buffer_mutex`: Mutex for audio buffer while data saving.\
- `ds_config_file_path`: Deepstream configuration file path.\
- `audio_model_file_path`: Audio recognition model pack file path.\
- `target_sr`: Model trained sample rate.\
- `max_seconds`: Madel trained maximum audio lenght.\
- `targets`: Recognized Targest boolean val for Red, Green, Blue respectively.

#### Functions
- **`AudioModel(const std::string& model_path, int sample_rate = 16000, int max_sec = 8)`**: **Constructor**. Class has just one constructor.\
  - `model_path` is audio model absolute path. Used for `torch::jit::load(model_path)`.\
  - `sample_rate` is microphone sound access count per time. It effect sound quality directly. And model must have trained dedicated sample rate. Default is 16000 (16Khz).\
  - `max_sec` is maximum second of model can process. if recorded data longer than `max_sec` then it will cropped till 'max_sec', but if recorded data shorter than `max_sec` then it will be padded with zeros till data riched as data as much amound of 'max_sec'. This default is 8 (8 seconds).

- **`data_callback`**: Microphone data record funciton in each frame (sound package). It is private.

- **`start_recording()`**: Firstly cleans to the `audio_buffer` using `buffer_mutex` and then starts to read microphone.

- **`stop_recording()`**: Stops recording.

- **`toTorchTensor()`**: Convert to torch tensor from vector data. If `audio_buffer` is empty, it returns an empty torch tensor.

- **`preprocess_audio()`**: Data preprocessor according to the 'max_sec'

- **`inference(torch::Tensor recorded_data)`**: Takes input data (`recorded_data`), runs model, attach `targets` returns out data. Returned data is empty when `audio_buffer` is empty or inference had met an error. In the other hand, returns raw color data which model recognized.

- **`get_targets()`**: Returns attached `targets`.

- **`get_is_recording()`**: Returns `is_recordings` to figure out data whether recorfing.

- **`get_audio_buffer()`**: Returns 'audio_buffer', this is thread-safe.

