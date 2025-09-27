# Alpaca Chinese Dataset 中文指令微调数据集【持续更新】

### 更新记录

- [2024/09/27] 对所有的翻译sample进行了质量评测，95%翻译准确，5%翻译不准确，给每一条sample附带了翻译的评分情况。使用时可以看情况过滤。

以下是翻译质量分数的实例：
```json
{
        "en_instruction": "List 3 historical events related to the following country",
        "en_input": "Canada",
        "en_output": "The three historical events related to Canada are the Dominion of Canada's establishment in 1867, the adoption of the Canadian Flag in 1965, and the signing of the Canada-U.S. Free Trade Agreement in 1988.",
        "zh_instruction": "列出与以下国家相关的3个历史事件",
        "zh_input": "加拿大",
        "zh_output": "与加拿大有关的三个历史事件是1867年加拿大建国，1965年采用加拿大国旗以及加拿大 - 美国签署。1988年的自由贸易协定。",
        "metadata": {
            "translated": true,
            "score": 2
        }
    },
    {
        "en_instruction": "Compare and contrast the effects of individual and societal performance",
        "en_input": "",
        "en_output": "Individual performance is related to the actions and decisions made by individuals while societal performance focuses on the actions and decisions of an entire group or organization. Individual performance is more focused on the needs and goals of the individual while societal performance is based on collective goals. Individual performance can be more easily measured and quantified while measuring the impact of societal performance can be more difficult as it usually involves evaluating the individual components making up the larger picture.",
        "zh_instruction": "比较和对比个人和社会绩效的影响",
        "zh_input": "",
        "zh_output": "个人绩效与个人做出的行动和决定有关，而社会绩效则侧重于整个群体或组织的行动和决策。个人绩效更侧重于个人的需求和目标，而社会绩效则基于集体目标。个人绩效可以更容易地衡量和量化，而衡量社会绩效的影响可能更加困难，因为它通常涉及评估构成大局的单个组成部分。",
        "metadata": {
            "translated": true,
            "score": 5
        }
    }
```

- [2024/09/20] GPT 4o translation validations
- [2024/07/29] GPT 4o (Personal account) testing for new translations
- [2024/04/26] GPT 4o 个人账户设置好了，算了一下成本还可以接受。准备进行数据更新和生成。
- [2024/04/26] 增加GPT Provider，为了将后续更多的优质英文数据源翻译成中文数据
- [2024/04/22] 增加Google翻译的Provider


### 如何贡献修改

欢迎大家一起参与维护该数据集，如果你希望修改数据集的内容，请人工修改./data目录下的json文件，不要直接修改最外层的alpaca-chinese-52k.json
./data下的json修改后，请运行：
```commandline
python main.py
```


会自动将你的修改更新到alpaca-chinese-52k.json中


### 数据集说明
本数据集包括中文和英文的混合数据集，方便双语微调，以及后续做持续的数据修正。

原始的Alpaca英文数据集也存在不少的问题，个别的数学类的sample是错的，有少部分output字段需要修正，一些<noinput>的标签没有对齐等。本数据集会对原始的数据集进行修改和修正。再此基础上，翻译出对应的中文版本，中文版基本是原始sample的翻译，但是对于一些比如押韵类、时态类的一些instruction，直接翻译导致韵脚丢失，时态不一致等。需要对应的进行人工改写。主要分为以下几个方面：

- 1. 修改原始英文数据集的一些问题

- 2. 翻译为中文数据集

- 3. 调整直译导致的一些sample

- 4. code等一些特殊的输出不进行翻译

- 5. 对齐一些特殊的标签<noinput> 或者拒绝生成等输出



### 格式说明
数据集为json dump的json文件，一个sample包括6个字段，en_instruction, en_input, en_output是英文对应的三个字段，instruction, input, output是中文对应的三个字段。
```json
{
    "en_instruction": "Conver the temperature from Fahrenheit to Celsius.",
    "instruction": "将温度从华氏度调到摄氏度。",
    "en_input": "94F",
    "input": "94F",
    "en_output": "34.4C",
    "output": "34.4摄氏度"
}
```


### 文件说明


| Dataset   |      file      |  notes |
|:----------:|:-------------:|:------:|
| alpaca-chinese |  alpaca-chinese-52k.json | 包含了52k英文和中文的数据全集 |
| alpaca-chinese |  ./data/alpaca_chinese_part*.json | 分拆数据文件 |



### 一些修改说明

**Case1** 成语：有一些sample，直译后需要进行二次改写，例如成语类的
```json
{
    "en_instruction": "What is the meaning of the following idiom?",
    "instruction": "以下成语是什么意思？",
    "en_input": "It’s raining cats and dogs",
    "input": "倾盆大雨",
    "en_output": "The idiom \"it’s raining cats and dogs\" means that it is raining heavily.",
    "output": "成语“倾盆大雨”的意思是雨下得像盆里得水往下倒，形容雨大势急。"
}
```

**Case2** 数学题，一些英文的sample是错误的，我们需要先修改后，再翻译。

原始如下，解是一个无理数，我们直接把题目改一下：
```json
{
    "en_instruction": "Find the solution to the quadratic equation.",
    "instruction": "求二次方程的解。",
    "en_input": "3x² + 7x - 4 = 0",
    "input": "3x² + 7x - 4 = 0",
    "en_output": "The solution of the quadratic equation is x = -4 and x = 1/3.",
    "output": "二次方程的解是 x = -4 和 x = 1/3。"
}
```
修改后如下：
```json
{
    "en_instruction": "Find the solution to the quadratic equation.",
    "instruction": "求二次方程的解。",
    "en_input": "2x² - x - 10 = 0",
    "input": "2x² - x - 10 = 0",
    "en_output": "The solution of the quadratic equation is x = -2 and x = 5/2.",
    "output": "二次方程的解是 x = -2 和 x = 5/2。"
}
```

**Case3** 字数限定
原始
```json
{
    "en_instruction": "Compose a five word sentence describing your day.",
    "instruction": "写一个五个字的句子来描述你的一天。",
    "en_input": "",
    "input": "",
    "en_output": "Today was an amazing day.",
    "output": "今天是令人惊喜的一天"
}
```

修改后：
```json
{
    "en_instruction": "Compose a five word sentence describing your day.",
    "instruction": "写一个五个字的句子来描述你的一天。",
    "en_input": "",
    "input": "",
    "en_output": "Today was an amazing day.",
    "output": "今天真美好"
}
```

**Case4** 拆字
原始：
```json
{
    "en_instruction": "Find a five-letter word that begins with 'd' and ends with 's'.",
    "instruction": "找到一个以“d”开头并以“s”结尾的五个字母的单词。",
    "en_input": "",
    "input": "",
    "en_output": "davis.",
    "output": "戴维斯。"
}
```
修改后：
```json
{
    "en_instruction": "Find a five-letter word that begins with 'd' and ends with 's'.",
    "instruction": "一个木字一个羽毛的羽是什么字。",
    "en_input": "",
    "input": "",
    "en_output": "davis.",
    "output": "栩栩如生的栩(xǔ)"
}
```

**Case5** 本地化
```json
{
    "en_instruction": "Name one social networking website",
    "instruction": "说出一个社交网站",
    "en_input": "",
    "input": "",
    "en_output": "Facebook.",
    "output": "微博"
}
```



