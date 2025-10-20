package org.example;

import org.example.utils.Result;
import org.example.utils.Util;

import javax.sound.sampled.AudioInputStream;
import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Test {
    public static void main(String[] args) throws Exception {
        //请输入文件路径
        String url = "https://nls-tech-support2.oss-cn-hangzhou.aliyuncs.com/zy02044583/2.mp3?x-oss-credential=LTAI5tAtWLXLw3V1KHomb13n%2F20251020%2Fcn-hangzhou%2Foss%2Faliyun_v4_request&x-oss-date=20251020T024659Z&x-oss-expires=3600&x-oss-signature-version=OSS4-HMAC-SHA256&x-oss-signature=4489d6144d616caaeb65189a55b7ceb328c32f8c3db555f1f17f52b6ae6b8af0";
        //请输入调用模型
        String modelName = "qwen3-asr-flash";
        //请输入上下文
        String context="此处用于配置定制化识别的Context";
        //请输入DASHSCOPE_API_KEY,默认不传取环境变量的值
        String apiKey = "";
        // 线程数
        int numThreads = 4;
        //vad分割音频块的目标持续时间
        int vadDuration = 120;


        String filePath;
        Util util = new Util();
        if (url.toLowerCase().startsWith("http://") || url.toLowerCase().startsWith("https://")) {
            URL u = new URL(url);
            String path = u.getPath();
            String fileName = new File(path).getName();

            String savePath = util.currentFileName()+"\\" + fileName;

            try (InputStream inputStream = new URL(url).openStream();
                 FileOutputStream outputStream = new FileOutputStream(savePath)) {

                byte[] buffer = new byte[4096];
                int bytesRead;

                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
                System.out.println("The audio has been successfully downloaded: " + savePath);

            } catch (IOException e) {
                e.printStackTrace();
            }
            filePath = savePath;
        } else {
            filePath = url;
        }
        //asr


        Result result = util.getAudioDurationWithFFmpeg(filePath);
        StringBuilder stringBuilder = new StringBuilder();
        System.out.println("Loaded wav duration:  " + result.getDuration() + "s");
        if (result.getDuration() < 180) {
            System.out.println("The audio duration is within 3 minutes.");
            String asr = util.QWen3Asr(modelName, filePath, context, apiKey);
            String text = util.parseTranscription(asr);
            stringBuilder.append(text);
        } else {
            System.out.println("The audio duration is more than 3 minutes");

            //ffmpeg
            System.out.println("<<<<<<<ffmpeg start<<<<<<<<");
            String outFile = util.convertToWAV16KInMemory(filePath);
            Map<Integer, String> resultMap = new LinkedHashMap<>();
            AudioSplitter audioSplitter = new AudioSplitter();
            try (AudioInputStream audioInputStream = audioSplitter.loadAudioFile(outFile)) {
                //Divide into 3 minutes
                // long[] splitPoints = audioSplitter.calculateSplitPointsTest(audioInputStream, 180);
                List<Long> splitPoints = audioSplitter.getSplitPoints(audioInputStream, vadDuration, outFile);

                for (long splitPoint : splitPoints) {
                    System.out.println(splitPoint);
                }

                audioSplitter.splitAudioFile(audioInputStream, splitPoints);

                System.out.println("Segmenting done, total segments:"+(splitPoints.size()-1));
                // 创建线程池
                ExecutorService executor = Executors.newFixedThreadPool(numThreads);
                List<Future<Map.Entry<Integer, String>>> futures = new ArrayList<>();

                for (int i = 0; i < splitPoints.size() - 1; i++) {
                    final int index = i;
                    Future<Map.Entry<Integer, String>> future = executor.submit(() -> {
                        try {
                            String asr = util.QWen3Asr(modelName, util.currentFileName()+"\\split_audio_" + index + ".wav",context,apiKey);
                            String text = util.parseTranscription(asr);
                            return new AbstractMap.SimpleEntry<>(index, text);
                        } catch (Exception e) {
                            throw new RuntimeException("ASR 处理失败", e);
                        }
                    });
                    futures.add(future);
                }

                for (Future<Map.Entry<Integer, String>> future : futures) {
                    Map.Entry<Integer, String> entry = future.get();
                    resultMap.put(entry.getKey(), entry.getValue());
                }

                executor.shutdown();
            } catch (IOException e) {
                System.err.println("音频加载失败: " + e.getMessage());
            }

            for (int i = 0; i < resultMap.size(); i++) {
                String entry = resultMap.get(i);
                stringBuilder.append(entry);
                util.deleteFile(util.currentFileName()+"\\split_audio_" + i + ".wav");
                //System.out.println("Completed:"+(i+1));
                //System.out.println(entry);
            }
        }


        //save
        if (stringBuilder.length() > 0) {
            System.out.println("Full Transcription:"+stringBuilder);
            int lastSeparatorIndex = filePath.lastIndexOf("\\");
            String directoryPath = filePath.substring(0, lastSeparatorIndex);
            File tempFile = new File(filePath.trim());
            String file = tempFile.getName();

            String outPath = directoryPath + "\\" + file.substring(0, file.lastIndexOf('.')) + ".txt"; // 文件路径


            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outPath))) {
                writer.write(stringBuilder.toString());
                System.out.println("Qwen3-ASR-Flash API saved to: " + outPath);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

        if (url.toLowerCase().startsWith("http://") || url.toLowerCase().startsWith("https://")) {
            util.deleteFile(filePath);
        }

    }

}
