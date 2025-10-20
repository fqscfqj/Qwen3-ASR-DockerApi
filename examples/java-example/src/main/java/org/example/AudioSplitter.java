package org.example;

import ai.onnxruntime.OrtException;
import org.example.utils.Util;

import javax.sound.sampled.*;
import java.io.File;
import java.io.IOException;
import java.util.*;


public class AudioSplitter {
    // ONNX model path - using the model file from the project
    private static final String MODEL_PATH = System.getProperty("user.dir") + "\\src\\main\\resources\\silero_vad.onnx";
    // Sampling rate
    private static final int SAMPLE_RATE = 16000;
    // Speech threshold (consistent with Python default)
    private static final float THRESHOLD = 0.5f;
    // Negative threshold (used to determine speech end)
    private static final float NEG_THRESHOLD = 0.35f; // threshold - 0.15
    // Minimum speech duration (milliseconds)
    private static final int MIN_SPEECH_DURATION_MS = 1500;
    // Minimum silence duration (milliseconds)
    private static final int MIN_SILENCE_DURATION_MS = 500;
    // Speech padding (milliseconds)
    private static final int SPEECH_PAD_MS = 30;
    // Window size (samples) - 512 samples for 16kHz
    private static final int WINDOW_SIZE_SAMPLES = 512;

    public AudioInputStream loadAudioFile(String filePath) throws UnsupportedAudioFileException, IOException {
        File audioFile = new File(filePath);
        AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile);
        return audioStream; // 返回音频输入流
    }

    public long[] calculateSplitPointsTest(AudioInputStream audioStream, long durationInSeconds) {
        long totalFrames = audioStream.getFrameLength();
        long frameRate = (long) audioStream.getFormat().getFrameRate();
        long durationInFrames = totalFrames / frameRate;

        long[] splitPoints = new long[(int) (durationInFrames / durationInSeconds + 1)];
        for (int i = 0; i < splitPoints.length; i++) {
            if (i == splitPoints.length - 1) {
                splitPoints[i] = totalFrames;
            } else {

                splitPoints[i] = i * durationInSeconds * frameRate; // 计算每个切分点的帧数
            }
        }

        return splitPoints; // 返回切分点的数组

    }

    public void splitAudioFile(AudioInputStream audioStream, List<Long> splitPoints) throws IOException {
        for (int i = 0; i < splitPoints.size() - 1; i++) {
            /*long startFrame = splitPoints[i];
            long endFrame = (i + 1 < splitPoints.length) ? splitPoints[i + 1] : audioStream.getFrameLength();*/
            long startFrame = splitPoints.get(i);
            long endFrame = splitPoints.get(i + 1);
            AudioInputStream splitStream = new AudioInputStream(audioStream, audioStream.getFormat(), endFrame - startFrame);
            // 保存切分后的音频文件

            Util util = new Util();
            File outputFile = new File(util.currentFileName() + "\\split_audio_" + i + ".wav");
            AudioSystem.write(splitStream, AudioFileFormat.Type.WAVE, outputFile);
        }
    }

    public List<Long> calculateSplitPoints(AudioInputStream audioStream, long vadDuration, String filePath) {
        //System.out.println("=".repeat(60));
        System.out.println("Silero VAD Java ONNX Example");
        //System.out.println("=".repeat(60));
//        long totalFrames = audioStream.getFrameLength();
//        long frameRate = (long) audioStream.getFormat().getFrameRate();
        List<Long> list = new ArrayList<>();
        // Load ONNX model
        SlieroVadOnnxModel model;
        try {
            System.out.println("Loading ONNX model: " + MODEL_PATH);
            model = new SlieroVadOnnxModel(MODEL_PATH);
            System.out.println("Model loaded successfully!");
        } catch (OrtException e) {
            System.err.println("Failed to load model: " + e.getMessage());
            e.printStackTrace();
            return null;
        }

        // Read WAV file
        float[] audioData;
        try {
            System.out.println("\nReading audio file: " + filePath);
            audioData = readWavFileAsFloatArray(filePath);
            System.out.println("Audio file read successfully, samples: " + audioData.length);
            System.out.println("Audio duration: " + String.format("%.2f", (audioData.length / (float) SAMPLE_RATE)) + " seconds");
        } catch (Exception e) {
            System.err.println("Failed to read audio file: " + e.getMessage());
            e.printStackTrace();
            return null;
        }

        // Get speech timestamps (batch mode, consistent with Python's get_speech_timestamps)
        System.out.println("\nDetecting speech segments...");
        List<Map<String, Integer>> speechTimestamps;
        try {
            speechTimestamps = getSpeechTimestamps(
                    audioData,
                    model,
                    THRESHOLD,
                    SAMPLE_RATE,
                    MIN_SPEECH_DURATION_MS,
                    MIN_SILENCE_DURATION_MS,
                    SPEECH_PAD_MS,
                    NEG_THRESHOLD
            );
        } catch (OrtException e) {
            System.err.println("Failed to detect speech timestamps: " + e.getMessage());
            e.printStackTrace();
            return null;
        }

        // Output detection results
        System.out.println("\nDetected speech timestamps (in samples):");
        list.add(0L);
        int j = 1;
        for (int i = 0; i < speechTimestamps.size(); i++) {

            long temp = j * vadDuration * 16000L;
            if (i == speechTimestamps.size() - 1) {
                //splitPoints[j] = audioData.length;
                list.add((long) audioData.length);
            } else if (i < speechTimestamps.size() - 1) {
                Map<String, Integer> first = speechTimestamps.get(i);
                Map<String, Integer> last = speechTimestamps.get(i + 1);
                if (temp > first.get("start") && temp < last.get("start")) {
                    //splitPoints[j] = first.get("end");
                    list.add(Long.valueOf(first.get("start")));
                    j++;
                }
            }

        }
        // Output detection results
        System.out.println("\nDetected speech timestamps (in samples):");
        for (Map<String, Integer> timestamp : speechTimestamps) {
            System.out.println(timestamp);
        }

        // Output summary
        System.out.println("Detection completed!");
        System.out.println("Total detected " + speechTimestamps.size() + " speech segments");

        // Close model
        try {
            model.close();
        } catch (OrtException e) {
            System.err.println("Error closing model: " + e.getMessage());
        }
        return list;
    }

    public List<Long> getSplitPoints(AudioInputStream audioStream, long vadDuration, String filePath) {

        System.out.println("Silero VAD Java ONNX Example");

        // Load ONNX model
        SlieroVadOnnxModel model;
        try {
            System.out.println("Loading ONNX model: " + MODEL_PATH);
            model = new SlieroVadOnnxModel(MODEL_PATH);
            System.out.println("Model loaded successfully!");
        } catch (OrtException e) {
            System.err.println("Failed to load model: " + e.getMessage());
            e.printStackTrace();
            return null;
        }

        // Read WAV file
        float[] audioData;
        try {
            System.out.println("\nReading audio file: " + filePath);
            audioData = readWavFileAsFloatArray(filePath);
            System.out.println("Audio file read successfully, samples: " + audioData.length);
            System.out.println("Audio duration: " + String.format("%.2f", (audioData.length / (float) SAMPLE_RATE)) + " seconds");
        } catch (Exception e) {
            System.err.println("Failed to read audio file: " + e.getMessage());
            e.printStackTrace();
            return null;
        }

        // Get speech timestamps (batch mode, consistent with Python's get_speech_timestamps)
        System.out.println("\nDetecting speech segments...");
        List<Map<String, Integer>> speechTimestamps;
        try {
            speechTimestamps = getSpeechTimestamps(
                    audioData,
                    model,
                    THRESHOLD,
                    SAMPLE_RATE,
                    MIN_SPEECH_DURATION_MS,
                    MIN_SILENCE_DURATION_MS,
                    SPEECH_PAD_MS,
                    NEG_THRESHOLD
            );
        } catch (OrtException e) {
            System.err.println("Failed to detect speech timestamps: " + e.getMessage());
            e.printStackTrace();
            return null;
        }

        // Output detection results
        List<Long> startTime = new ArrayList<>();
        for (Map<String, Integer> timestamp : speechTimestamps) {
            timestamp.get("start");
            startTime.add(Long.valueOf(timestamp.get("start")));
        }

        // 初始化分割点集合（包含起始点 0.0 和音频长度）
        Set<Long> finalSplitPoints = new HashSet<>();
        finalSplitPoints.add(0L);
        finalSplitPoints.add((long) audioData.length);

        // 计算分割阈值对应的样本数
        long segmentThresholdSamples = vadDuration * SAMPLE_RATE;
        long targetTime = segmentThresholdSamples;

        // 循环添加分割点
        while (targetTime < audioData.length) {
            long closestPoint = findClosest(startTime, targetTime);
            finalSplitPoints.add(closestPoint);
            targetTime += segmentThresholdSamples;
        }

        // 将分割点排序并输出
        List<Long> finalOrderedSplits = new ArrayList<>(finalSplitPoints);
        Collections.sort(finalOrderedSplits);
        System.out.println("Final Ordered Splits: " + finalOrderedSplits);
        //-------
        // Output detection results
        System.out.println("\nDetected speech timestamps (in samples):");
       /* for (Map<String, Integer> timestamp : speechTimestamps) {
            System.out.println(timestamp);
        }*/

        // Output summary
        System.out.println("Detection completed!");
        System.out.println("Total detected " + speechTimestamps.size() + " speech segments");

        // Close model
        try {
            model.close();
        } catch (OrtException e) {
            System.err.println("Error closing model: " + e.getMessage());
        }
        return finalOrderedSplits;
    }
    // 查找最接近 targetTime 的分割点
    private static long findClosest(List<Long> list, long target) {
        long closest = list.get(0);
        for (long num : list) {
            if (Math.abs(num - target) < Math.abs(closest - target)) {
                closest = num;
            }
        }
        return closest;
    }

    /**
     * Get speech timestamps
     * Implements the same logic as Python's get_speech_timestamps
     *
     * @param audio                Audio data (float array)
     * @param model                ONNX model
     * @param threshold            Speech threshold
     * @param samplingRate         Sampling rate
     * @param minSpeechDurationMs  Minimum speech duration (milliseconds)
     * @param minSilenceDurationMs Minimum silence duration (milliseconds)
     * @param speechPadMs          Speech padding (milliseconds)
     * @param negThreshold         Negative threshold (used to determine speech end)
     * @return List of speech timestamps
     */
    private static List<Map<String, Integer>> getSpeechTimestamps(
            float[] audio,
            SlieroVadOnnxModel model,
            float threshold,
            int samplingRate,
            int minSpeechDurationMs,
            int minSilenceDurationMs,
            int speechPadMs,
            float negThreshold) throws OrtException {

        // Reset model states
        model.resetStates();

        // Calculate parameters
        int minSpeechSamples = samplingRate * minSpeechDurationMs / 1000;
        int speechPadSamples = samplingRate * speechPadMs / 1000;
        int minSilenceSamples = samplingRate * minSilenceDurationMs / 1000;
        int windowSizeSamples = samplingRate == 16000 ? 512 : 256;
        int audioLengthSamples = audio.length;

        // Calculate speech probabilities for all audio chunks
        List<Float> speechProbs = new ArrayList<>();
        for (int currentStart = 0; currentStart < audioLengthSamples; currentStart += windowSizeSamples) {
            float[] chunk = new float[windowSizeSamples];
            int chunkLength = Math.min(windowSizeSamples, audioLengthSamples - currentStart);
            System.arraycopy(audio, currentStart, chunk, 0, chunkLength);

            // Pad with zeros if chunk is shorter than window size
            if (chunkLength < windowSizeSamples) {
                for (int i = chunkLength; i < windowSizeSamples; i++) {
                    chunk[i] = 0.0f;
                }
            }

            float speechProb = model.call(new float[][]{chunk}, samplingRate)[0];
            speechProbs.add(speechProb);
        }

        // Detect speech segments using the same algorithm as Python
        boolean triggered = false;
        List<Map<String, Integer>> speeches = new ArrayList<>();
        Map<String, Integer> currentSpeech = null;
        int tempEnd = 0;

        for (int i = 0; i < speechProbs.size(); i++) {
            float speechProb = speechProbs.get(i);

            // Reset temporary end if speech probability exceeds threshold
            if (speechProb >= threshold && tempEnd != 0) {
                tempEnd = 0;
            }

            // Detect speech start
            if (speechProb >= threshold && !triggered) {
                triggered = true;
                currentSpeech = new HashMap<>();
                currentSpeech.put("start", windowSizeSamples * i);
                continue;
            }

            // Detect speech end
            if (speechProb < negThreshold && triggered) {
                if (tempEnd == 0) {
                    tempEnd = windowSizeSamples * i;
                }
                if (windowSizeSamples * i - tempEnd < minSilenceSamples) {
                    continue;
                } else {
                    currentSpeech.put("end", tempEnd);
                    if (currentSpeech.get("end") - currentSpeech.get("start") > minSpeechSamples) {
                        speeches.add(currentSpeech);
                    }
                    currentSpeech = null;
                    tempEnd = 0;
                    triggered = false;
                }
            }
        }

        // Handle the last speech segment
        if (currentSpeech != null &&
                (audioLengthSamples - currentSpeech.get("start")) > minSpeechSamples) {
            currentSpeech.put("end", audioLengthSamples);
            speeches.add(currentSpeech);
        }

        // Add speech padding - same logic as Python
        for (int i = 0; i < speeches.size(); i++) {
            Map<String, Integer> speech = speeches.get(i);
            if (i == 0) {
                speech.put("start", Math.max(0, speech.get("start") - speechPadSamples));
            }
            if (i != speeches.size() - 1) {
                int silenceDuration = speeches.get(i + 1).get("start") - speech.get("end");
                if (silenceDuration < 2 * speechPadSamples) {
                    speech.put("end", speech.get("end") + silenceDuration / 2);
                    speeches.get(i + 1).put("start",
                            Math.max(0, speeches.get(i + 1).get("start") - silenceDuration / 2));
                } else {
                    speech.put("end", Math.min(audioLengthSamples, speech.get("end") + speechPadSamples));
                    speeches.get(i + 1).put("start",
                            Math.max(0, speeches.get(i + 1).get("start") - speechPadSamples));
                }
            } else {
                speech.put("end", Math.min(audioLengthSamples, speech.get("end") + speechPadSamples));
            }
        }

        return speeches;
    }

    /**
     * Read WAV file and return as float array
     *
     * @param filePath WAV file path
     * @return Audio data as float array (normalized to -1.0 to 1.0)
     */
    private static float[] readWavFileAsFloatArray(String filePath)
            throws UnsupportedAudioFileException, IOException {
        File audioFile = new File(filePath);
        AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile);

        // Get audio format information
        AudioFormat format = audioStream.getFormat();
        System.out.println("Audio format: " + format);

        // Read all audio data
        byte[] audioBytes = audioStream.readAllBytes();

        // 替代方案：手动读取所有字节
        /*ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[1024];
        while ((nRead = audioStream.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        byte[] audioBytes = buffer.toByteArray();*/

        audioStream.close();

        // Convert to float array
        float[] audioData = new float[audioBytes.length / 2];
        for (int i = 0; i < audioData.length; i++) {
            // 16-bit PCM: two bytes per sample (little-endian)
            short sample = (short) ((audioBytes[i * 2] & 0xff) | (audioBytes[i * 2 + 1] << 8));
            audioData[i] = sample / 32768.0f; // Normalize to -1.0 to 1.0
        }

        return audioData;
    }
}