---
title: "المقال 00: Helloworld CUDA في 2026"
date: 2026-09-29T18:41:58+03:00
weight: 1
draft: true
toc: false
images:
series: ["cuda"]
tags:
  - cuda
  - cpp
  - linux
---

في هذه السلسلة، سنقدم مقدمة مبسطة ومناسبة للمبتدئين إلى عالم برمجة وحدات المعالجة الرسومية (GPU) باستخدام **CUDA**.

يُعد CUDA منصة NVIDIA لتشغيل الأكواد العامة (general-purpose) على وحدة المعالجة الرسومية. تكتب دالة بلغة C++، وتعلّمها بأنها **نواة (Kernel)**، ثم تقوم الـ GPU بتشغيل آلاف النسخ منها بالتوازي. معظم الشروحات التي ستجدها على الإنترنت كُتبت لإصدارات CUDA 10 أو 11، وقد تغيّر شيءٌ ما منذ ذلك الحين. فقد أسقطت CUDA 13 دعمًا لبعض بطاقات الرسوميات القديمة، وأصبحت أنظمة Ubuntu توزّع حزمة الأدوات ضمن مستودعاتها الخاصة، وأصبح بإمكان CMake اكتشاف بطاقة الرسوميات لديك تلقائيًا. لذلك في هذا المقال الأول، سنكتب أمثلة hello world الكلاسيكية، لكن بطريقة 2026.

## ماذا سنتعلم؟

في هذا المقال سنتعرف على:

* كيفية تثبيت حزمة أدوات CUDA على Ubuntu 26.04.
* كيفية كتابة أول نواة (Kernel) لدينا وتشغيلها.
* كيفية بناء المشروع باستخدام `nvcc` و CMake.
* كيفية تجاوز تعارض في الملفات الرأسية (Headers) بين CUDA 13.1 والنسخ الحديثة من glibc.

## محتويات المقال

1. تثبيت CUDA.
2. التحقق من نجاح التثبيت.
3. كتابة نواة Hello World.
4. البناء والتشغيل.
5. تمارين عملية.

---

## 1. تثبيت CUDA

بدءًا من Ubuntu 26.04، أصبحت حزمة أدوات CUDA متاحة مباشرةً من مستودع `multiverse`، مما يعني أننا لم نعد بحاجة إلى إضافة مستودع NVIDIA الخاص (apt).

```bash
sudo apt update
sudo apt install cuda-toolkit
```

انتبه جيدًا لاسم الحزمة. هناك أيضًا حزمة قديمة تُسمى `nvidia-cuda-toolkit`، لكنها ما زالت على CUDA 12.4. نحن نريد `cuda-toolkit`، لأنها تمنحنا CUDA 13.1.

| الحزمة               | الإصدار | الوصف                                 |
| -------------------- | ------- | --------------------------------------- |
| cuda-toolkit         | 13.1    | الحزمة التي نريدها                      |
| nvidia-cuda-toolkit  | 12.4    | تعبئة قديمة من Debian `يمكنك تجاوزها`    |

تُثبَّت حزمة الأدوات تحت المسار `/usr/local/cuda`، وهو غير موجود في متغير `PATH` لديك بشكل افتراضي. أضِفه إلى ملف `~/.bashrc` (أو `~/.zshrc`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

إذا كنت تستخدم توزيعة أخرى، فاتبع دليل التثبيت الرسمي [1].

---

## 2. التحقق من نجاح التثبيت

نحتاج إلى شيئين ليعمل كل شيء: السائق (Driver) الذي يتحدث مع بطاقة الرسوميات، والمُترجم `nvcc` الذي يبني كودنا.

```bash
nvidia-smi
nvcc --version
```

مثال على المخرجات (Ubuntu 26.04، RTX 4090):

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_Dec_16_07:23:41_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.115
Build cuda_13.1.r13.1/compiler.37061995_0
```

شيء آخر يجب التحقق منه هو *قوة الحساب (Compute Capability)* لبطاقة الرسوميات، وهو رقم الإصدار الذي تمنحه NVIDIA لبنية (Architecture) البطاقة:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

```text
name, compute_cap
NVIDIA GeForce RTX 4090, 8.9
```

أسقطت CUDA 13 دعم بطاقات Maxwell و Pascal و Volta، لذا فإنها تتطلب قوة حساب لا تقل عن 7.5 (أي بنية Turing، بطاقات سلسلة RTX 20). إذا كان رقم بطاقاتك أقل من ذلك، فستحتاج إلى البقاء على CUDA 12 [2].

---

## 3. كتابة نواة Hello World

```bash
mkdir hello-cuda
cd hello-cuda
code main.cu
```

تستخدم ملفات مصدر CUDA الامتداد `.cu`. يمكنك كتابة الكود التالي في `main.cu`

{{< code file="cuda/post00/main.cu" >}}

لنقرأ الكود جزءًا جزءًا.

### النواة (The Kernel)

{{< code file="cuda/post00/main.cu" region="kernel" link="false" >}}

الكلمة المفتاحية `__global__` تحوّل دالة عادية إلى نواة (Kernel). تُستدعى هذه الدالة من المعالج الرئيسي CPU (يُطلق عليه *المضيف - Host*) لكنها تعمل على بطاقة الرسوميات GPU (يُطلق عليه *الجهاز - Device*). ويجب أن تكون النواة من النوع `void` (لا تعيد أي قيمة).

كل نسخة من النواة تعمل ك**خيط (Thread)**، وتُجمَّع الخيوط في **كتل (Blocks)**. داخل النواة، يخبرنا المتغير `blockIdx.x` بأي كتلة نحن الآن، ويخبرنا `threadIdx.x` بأي خيط نحن داخل تلك الكتلة. هذه المتغيرات المدمجة هي الطريقة التي يحدد بها كل خيط أي جزء من العمل يخصه.

### إطلاق النواة (Launching the kernel)

{{< code file="cuda/post00/main.cu" region="launch" link="false" >}}

الصيغة `<<<2, 4>>>` هي امتداد لـ CUDA على لغة C++. تعني هذه الصيغة *أطلق كتلتين (2 Blocks) كل واحدة تحتوي على 4 خيوط (Threads)*، وبالتالي ستعمل النواة 8 مرات.

إطلاق النواة (Launch) عملية **غير متزامنة (Asynchronous)**. يقوم المعالج الرئيسي CPU بوضع العمل في قائمة الانتظار ويواصل التنفيذ فورًا دون انتظار انتهاء بطاقة الرسوميات GPU من العمل. ولهذا السبب نحتاج إلى استدعاءَين بعد الإطلاق:

* `cudaGetLastError` يلتقط أي مشاكل في عملية الإطلاق نفسها، مثل طلب عدد أكبر من الخيوط المسموح به.
* `cudaDeviceSynchronize` ينتظر حتى تنتهي الـ GPU من عملها، ويعيد أي خطأ حدث أثناء تشغيل النواة. كما يُفرغ (Flush) مخرجات `printf` القادمة من الجهاز. دون هذا الاستدعاء، قد تخرج الدالة `main` قبل أن تُطبع أي رسالة على الشاشة.

### فحص الأخطاء (Checking errors)

{{< code file="cuda/post00/main.cu" region="check" link="false" >}}

تعيد تقريبًا جميع دوال وقت التشغيل (Runtime) في CUDA نوعًا يسمى `cudaError_t`، و CUDA لا ترمي استثناءات (Exceptions). إذا تجاهلت قيمة الإرجاع، فقد يفشل الأمر دون أن نلاحظ، ويستمر برنامجنا في العمل على بيانات خاطئة. أما ماكر `CUDA_CHECK` فيطبع اسم الملف، رقم السطر، ورسالة واضحة، ثم يخرج من `main`. ستجد هذا الماكر، أو ما يشبهه جدًا، في كل أكواد CUDA تقريبًا.

### استعلام الجهاز (Querying the device)

{{< code file="cuda/post00/main.cu" region="device" link="false" >}}

هذه الخطوة ليست ضرورية تمامًا لبرنامج hello world، لكنها تؤكد لنا أي بطاقة رسوميات نعمل عليها فعلًا. وهي مفيدة خاصة إذا كان الجهاز يحتوي على أكثر من بطاقة.

---

## 4. البناء والتشغيل

{{< tabs >}}
{{< tab "nvcc" >}}
```bash
nvcc -U_GNU_SOURCE main.cu -o hello-cuda
./hello-cuda
```
{{< /tab >}}
{{< tab "CMake" >}}
أنشئ ملف `CMakeLists.txt` بجانب `main.cu`

{{< code file="cuda/post00/CMakeLists.txt" >}}

إعداد `CMAKE_CUDA_ARCHITECTURES` إلى قيمة `native` (متاح من CMake 3.24 فما فوق) يبني البرنامج للبطاقة الرسومية المثبتة على جهازك. قبل ذلك، كان عليك أن تبحث عن رقم قوة الحساب (Compute Capability) وتكتبه يدويًا.

```bash
CUDAFLAGS=-U_GNU_SOURCE cmake -S . -B build
cmake --build build
./build/hello-cuda
```
{{< /tab >}}
{{< /tabs >}}

المخرجات:

```text
Running on NVIDIA GeForce RTX 4090 (compute capability 8.9)
Hello from block 1, thread 0
Hello from block 1, thread 1
Hello from block 1, thread 2
Hello from block 1, thread 3
Hello from block 0, thread 0
Hello from block 0, thread 1
Hello from block 0, thread 2
Hello from block 0, thread 3
```

لاحظ أن الكتلة 1 طبعت قبل الكتلة 0. الـ GPU لا تعد بأي ترتيب بين الكتل، والكود الخاص بك يجب ألا يعتمد على أي ترتيب محدد. شغّل البرنامج عدة مرات وستجد أن الترتيب قد يتغير.

### ما هو `-U_GNU_SOURCE`؟

إذا قمت بالبناء بدون هذا العلم (Flag) على Ubuntu 26.04، ستحصل على هذا الخطأ:

```text
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(206): error: exception specification is incompatible with that of previous function "rsqrt" (declared at line 629 of /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/math_functions.h)
```

دالة `rsqrt` (الجذر التربيعي العكسي) أُضيفت إلى اللغة C في المعيار C23، لذا أصبحت النسخ الحديثة من glibc تعلن عنها داخل `math.h`. بينما لدى CUDA دالتها الخاصة `rsqrt` منذ سنوات، وأعاد إصدار CUDA 13.1 تعريفها بشكل مختلف قليلًا. عند وصول التعريفَين إلى نفس الملف، يرفض المُجمِّع (Compiler) مواصلة العمل.

لا تعرض glibc امتدادات C23 مثل هذا إلا عندما تكون الخاصية `_GNU_SOURCE` معرّفة، و `g++` يفعّلها افتراضيًا. إلغاء تعريفها باستخدام `-U_GNU_SOURCE` يخفي تعريف glibc ويترك نسخة CUDA قائمة. تعامل مع هذا الحل على أنه **حل مؤقت**: عندما تصل حزمة الأدوات الجديدة وتُصلح هذا التعارض في الملفات الرأسية، يمكنك التخلّص من هذا العلم.

---

## 5. تمارين عملية

غيّر عملية الإطلاق إلى `hello<<<1, 2048>>>()` وشغّل البرنامج مرة أخرى. ماذا يحدث؟ وأي الفحصَين سيكشف هذا الخطأ؟

تلميح: لا يمكن للكتلة (Block) أن تحتوي على أكثر من 1024 خيطًا (Thread).

**الخلاصة:** كتابة برنامج hello world في CUDA في 2026 لا تزال تعتمد على دالة `__global__` وإطلاق `<<<blocks, threads>>>`، لكن الأدوات المحيطة بها تغيّرت: حزمة الأدوات أصبحت تأتي من مستودعات Ubuntu، و CMake يختار بنية GPU تلقائيًا، و CUDA 13 تتطلب بطاقة Turing أو أحدث. في المقال القادم، سنتوقف عن طباعة الرسائل ونبدأ بالحساب الفعلي، عبر نقل البيانات بين ذاكرة المعالج الرئيسي وذاكرة بطاقة الرسوميات.

[1]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ "CUDA Installation Guide for Linux"
[2]: https://developer.nvidia.com/cuda-gpus "CUDA GPU Compute Capability"