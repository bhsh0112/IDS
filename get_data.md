要捕获您电脑上的网络流量并将其转换为UNSW-NB15数据集的格式，您可以按照以下步骤操作：

---

### 一、使用 Wireshark 捕获网络流量

1. **安装 Wireshark**：
   - 访问 [Wireshark 官网](https://www.wireshark.org/) 下载并安装适合您操作系统的版本。

2. **启动 Wireshark 并选择网络接口**：
   - 打开 Wireshark，您将看到可用的网络接口列表。
   - 选择您希望监控的网络接口（如以太网或无线网卡）。

3. **开始捕获数据包**：
   - 双击所选网络接口，Wireshark 将开始实时捕获通过该接口的所有数据包。

4. **使用过滤器筛选数据包（可选）**：
   - 您可以使用过滤器来聚焦特定类型的流量，例如：
     - `ip.addr == 192.168.1.1`：捕获与特定 IP 地址相关的流量。
     - `tcp.port == 80`：捕获通过特定端口的流量。
   - 这有助于减少捕获的数据量，聚焦于您感兴趣的流量。

5. **保存捕获的数据**：
   - 捕获完成后，点击“文件” > “保存”，将数据保存为 `.pcap` 格式，以便后续处理。

---

### 二、将捕获的数据转换为 UNSW-NB15 格式

UNSW-NB15 数据集包含 49 个特征，涵盖了从网络流量中提取的各种信息。要将您的 `.pcap` 文件转换为类似的格式，您可以按照以下步骤操作：

1. **使用特征提取工具**：
   - 工具如 [CICFlowMeter](https://www.unb.ca/cic/research/applications.html) 可以从 `.pcap` 文件中提取流量特征，并生成 CSV 文件。
   - 这些特征包括流持续时间、包数量、字节数、平均包大小等，与 UNSW-NB15 数据集中的特征类似。

2. **对特征进行预处理**：
   - 根据您的需求，可能需要对提取的特征进行标准化、归一化或编码处理。
   - 例如，将分类特征进行 One-Hot 编码，数值特征进行标准化处理，以适应机器学习模型的输入要求。

3. **添加标签信息**：
   - UNSW-NB15 数据集中的每条记录都有一个标签，指示该流量是否为攻击以及攻击的类型。
   - 您需要根据捕获流量的实际情况，为每条记录添加相应的标签。
   - 如果您只是捕获正常流量，可以将所有记录标记为“正常”；如果包含已知的攻击流量，需要准确标注攻击类型。

4. **保存为 CSV 格式**：
   - 将处理后的数据保存为 CSV 文件，结构应与 UNSW-NB15 数据集相似，以便于后续的分析和模型训练。

---

### 三、注意事项

- **数据合法性与隐私**：
  - 确保您有权捕获和使用所涉及的网络流量数据，遵守相关法律法规和隐私政策。

- **数据质量与多样性**：
  - 为了训练出更具泛化能力的模型，建议捕获多样化的流量数据，包括不同时间、不同应用、不同网络环境下的流量。

- **攻击流量的获取**：
  - 如果您希望包含攻击流量，可以使用安全的测试环境（如虚拟机）模拟已知的攻击行为，并捕获相应的流量。

---

通过上述步骤，您可以捕获自己电脑上的网络流量，并将其转换为类似 UNSW-NB15 数据集的格式，用于测试和研究网络入侵检测系统。如果您需要进一步的帮助，例如特征提取脚本或数据预处理代码，请随时告知。 