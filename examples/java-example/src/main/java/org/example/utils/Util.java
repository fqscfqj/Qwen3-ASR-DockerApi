package org.example.utils;

import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversation;
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationParam;
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationResult;
import com.alibaba.dashscope.common.MultiModalMessage;
import com.alibaba.dashscope.common.Role;
import com.alibaba.dashscope.utils.JsonUtils;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;


import java.io.*;
import java.util.*;

public class Util {

    public Result getAudioDurationWithFFmpeg(String filePath) {
        try {
            Result audio = new Result();
            String[] pb = {"ffmpeg",
                    "-i", filePath,
            };

            // 运行 FFmpeg 解码命令
            Process process = Runtime.getRuntime().exec(pb);

            // 读取错误流（FFmpeg 输出信息在 stderr）
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                //System.out.println(line);

                if (line.contains("Input")) {
                    String file = line.split(",")[1];
                    //System.out.println(file);
                    audio.setFormat(file);
                }
                if (line.contains("Duration:")) {
                    String[] duration = line.split(",")[0].split(":");
                    int hours = Integer.parseInt(duration[1].trim()); // 注意这里的索引是1，因为第一部分是"Duration"
                    int minutes = Integer.parseInt(duration[2]);
                    float seconds = Float.parseFloat(duration[3].split("\\.")[0]); // 取小数点前的秒数
                    //System.out.println(hours * 3600 + minutes * 60 + seconds);
                    double time = hours * 3600 + minutes * 60 + seconds;
                    audio.setDuration(time);

                }
                if (line.contains("Error")) {
                    throw new RuntimeException(line);
                }
            }

            process.waitFor();
            return audio;
        } catch (Exception e) {
            throw new RuntimeException("FFmpeg 执行失败，退出码: " + e.getMessage());

        }
    }

    public String convertToWAV16KInMemory(String inputFilePath) throws IOException, InterruptedException {
        int lastSeparatorIndex = inputFilePath.lastIndexOf("\\");
        String directoryPath = inputFilePath.substring(0, lastSeparatorIndex);
        File tempFile = new File(inputFilePath.trim());
        String fileName = tempFile.getName();

        String ffmpegPath = directoryPath + "\\" + fileName.substring(0, fileName.lastIndexOf('.')) + ".wav"; // 文件路径
        String command = "ffmpeg -i " + inputFilePath + " -ar 16000 -ac 1 -y " + ffmpegPath; //command
        try {
            Process process = Runtime.getRuntime().exec(command);

            // 读取命令执行的输出
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
            // 等待命令执行完成
            int exitVal = process.waitFor();
            if (exitVal == 0) {
                System.out.println("Success!");
            } else {
                // 读取错误输出流，以便了解出错原因
                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                String errorLine;
                while ((errorLine = errorReader.readLine()) != null) {
                    System.err.println(errorLine);
                }
                System.out.println("Error!");
            }
            return ffmpegPath;
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("FFmpeg 调用异常", e);
        }
    }


    public String QWen3Asr(String model, String localFilePath, String content,String apiKey) throws Exception {

        if (apiKey == null || apiKey.isEmpty()){
            apiKey = System.getenv("DASHSCOPE_API_KEY");
        }
        MultiModalConversation conv = new MultiModalConversation();
        MultiModalMessage userMessage = MultiModalMessage.builder()
                .role(Role.USER.getValue())
                .content(Arrays.asList(
                        Collections.singletonMap("audio", localFilePath)))
                .build();

        MultiModalMessage sysMessage = MultiModalMessage.builder().role(Role.SYSTEM.getValue())
                // 此处用于配置定制化识别的Context
                .content(Arrays.asList(Collections.singletonMap("text", content)))
                .build();

        Map<String, Object> asrOptions = new HashMap<>();
        asrOptions.put("enable_lid", true);
        asrOptions.put("enable_itn", false);
        // asrOptions.put("language", "zh"); // 可选，若已知音频的语种，可通过该参数指定待识别语种，以提升识别准确率
        MultiModalConversationParam param = MultiModalConversationParam.builder()
                // 若没有配置环境变量，请用百炼API Key将下行替换为：.apiKey("sk-xxx")
                .apiKey(apiKey)
                .model(model)
                .message(userMessage)
                .message(sysMessage)
                .parameter("asr_options", asrOptions)
                .build();
        MultiModalConversationResult result = conv.call(param);
        System.out.println(JsonUtils.toJson(result));
        return JsonUtils.toJson(result);

    }

    private static ObjectMapper objectMapper = new ObjectMapper();

    public String parseTranscription(String jsonStr) throws Exception {
        JsonNode root = objectMapper.readTree(jsonStr);
        JsonNode data = root.path("output").path("choices");
        StringBuilder stringBuilder = new StringBuilder();
        for (JsonNode choice : data) {
            JsonNode message = choice.path("message");
            JsonNode content = message.path("content");
            for (JsonNode part : content) {
                stringBuilder.append(part.path("text").asText());
            }

        }


        return stringBuilder.toString();

    }

    public void deleteFile(String filePath) {
        File file = new File(filePath);
        if (file.exists()) {
            if (file.delete()) {
                //System.out.println("已完成分片1: " + file.getAbsolutePath());
            } else {
                //System.out.println("无法删除文件: " + file.getAbsolutePath());
            }
        } else {
            //System.out.println("文件不存在: " + file.getAbsolutePath());
        }
    }

    public String currentFileName() {

        return System.getProperty("user.dir");
    }

}
