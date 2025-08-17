# skin_classification

列名解析：
├── md5hash: 图像的MD5哈希值，作为唯一标识符
├── fitzpatrick_scale: Fitzpatrick皮肤类型量表 (1-6)
│   ├── 1-2: 浅色皮肤 (易晒伤，难晒黑)
│   ├── 3-4: 中等肤色 (偶尔晒伤，逐渐晒黑)
│   └── 5-6: 深色皮肤 (很少晒伤，容易晒黑)
├── fitzpatrick_centaur: 另一种Fitzpatrick分类方法(可能更精确)
├── label: 数字编码的疾病类别 (0-113，对应114种疾病)
├── nine_partition_label: 具体疾病名称/描述
├── three_partition_label: 三大类疾病分类 ⭐ (关键！)
├── qc: 质量控制标志
└── url: 原始图像来源URL
