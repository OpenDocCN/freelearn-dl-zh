# 5

# 数据版本控制

**数据版本控制**指的是在整个模型开发生命周期中，包括预训练、微调、评估和部署过程中，对数据集的不同迭代进行系统跟踪和管理。它涉及为数据集或其子集分配唯一的标识符，记录随时间的变化，并通过确保任何特定模型版本都可以回溯到确切的数据版本来确保可重复性。

在本章中，您将学习如何为 LLM 开发实现有效的数据版本控制策略。例如，当我们想要将 10,000 篇新的肿瘤学研究论文添加到数据集中时，系统会自动创建一个新的数据集版本。如果模型性能随后下降，数据集可以立即回滚到之前经过验证的数据集版本，确保可重复性和维护研究过程的完整性。

这种设计模式将数据集管理从混乱的、手动的过程转变为 LLM 模型开发中的结构化、可追踪的工作流程。

在本章中，我们将涵盖以下主题：

+   理解数据版本控制的需求

+   大型语言数据集的数据版本控制策略

+   数据版本控制工具

+   将数据版本控制集成到训练工作流程中

+   文本语料库的版本控制

+   管理数据集变体和实验

+   数据版本控制的最佳实践

# 理解数据版本控制的需求

由于语言数据集的规模和复杂性巨大，数据版本控制对于 LLM 项目尤为重要。作为一名 LLM 工程师，您需要跟踪数据集中的变化，以确保模型的可重复性，并保持数据修改的清晰历史记录。

让我们从使用 Python 实现一个基本的数据版本控制系统开始：

```py
from datetime import datetime
import hashlib
import json
class DatasetVersion:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    //creation timestamp for each version
        self.version_hash = self._generate_hash()
    def _generate_hash(self):
        data_str = json.dumps(self.data, sort_keys=True).encode()
        return hashlib.sha256(data_str).hexdigest()
```

`DatasetVersion` 类的这一部分初始化了为您的 LLM 数据集进行版本控制的基本结构。它为每个数据版本生成一个唯一的哈希值，并标记版本的时间戳。`_generate_hash` 方法基于数据的排序 JSON 表示创建一个确定性的哈希值，确保相同的数据总是产生相同的哈希值。

现在，让我们为数据集版本添加 `save` 和 `load` 方法：

```py
class DatasetVersion:
    # ... (previous methods)
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                'data': self.data,
                'metadata': self.metadata,
                'timestamp': self.timestamp,
                'version_hash': self.version_hash
            }, f, indent=2)
    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        instance = cls(data['data'], data['metadata'])
        instance.timestamp = data['timestamp']
        instance.version_hash = data['version_hash']
        return instance
```

`save` 方法将数据集版本序列化为 JSON 文件，包括所有相关信息。`load` 方法是一个类方法，它从保存的文件中重建一个 `DatasetVersion` 实例。这使得您可以轻松地存储和检索数据集的不同版本。

在讨论了数据版本控制的需求之后，现在让我们概述管理大型语言数据集版本控制的关键策略，以支持可追溯性、可重复性和高效的存储。

# 大型语言数据集的数据版本控制策略

在处理大型语言数据集的迭代更新时，由于可以最小化存储成本，本节重点介绍了**基于增量的系统**，因为它在处理数据版本管理时具有潜在优势。基于增量的版本管理只存储数据集版本之间的差异，而不是复制整个文件，这使得它在涉及频繁但微小的更改的场景中特别有效。然而，当数据集结构发生重大重新格式化或涉及二进制文件时，其有效性会降低。模式更改、列重排或文件拆分可能会破坏增量机制，通常需要完整数据集的重写。同样，由于二进制文件的结构不透明和压缩，即使是微小的编辑也会导致全局变化，限制了基于增量存储的优势。这种方法在此处讨论，因为它在典型的 LLM 工作流程中具有相关性，其中数据逐渐演变，但仍然主要基于文本和结构化。

这里有一个如何实现基于增量版本化系统的示例：

```py
import difflib
class DeltaDatasetVersion(DatasetVersion):
    def __init__(
        self, data, base_version=None, metadata=None
    ):
        super().__init__(data, metadata)
        self.base_version = base_version
        self.delta = self._compute_delta() if base_version else None
    def _compute_delta(self):
        base_data = json.dumps(
            self.base_version.data, sort_keys=True).splitlines()
        current_data = json.dumps(
            self.data, sort_keys=True).splitlines()
        diff = list(
            difflib.unified_diff(
                base_data, current_data, lineterm='')
            )
        return '\n'.join(diff)
```

`DeltaDatasetVersion`类的这部分扩展了我们的先前`DatasetVersion`类，以实现基于增量的版本管理。`_compute_delta`方法使用 Python 的`difflib`计算当前版本与基础版本之间的差异。这种方法可以通过仅存储更改来显著减少大型数据集的存储需求。

现在，让我们添加保存和加载这些基于增量的版本的方法：

```py
class DeltaDatasetVersion(DatasetVersion):
    # ... (previous methods)
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'timestamp': self.timestamp,
                'version_hash': self.version_hash,
                'base_version_hash': (
                    self.base_version.version_hash
                    if self.base_version else None
                ),
                'delta': self.delta
            }, f, indent=2)
    @classmethod
    def load(cls, filename, base_version):
        with open(filename, 'r') as f:
            data = json.load(f)
        # Apply delta to base version
        base_data = json.dumps(
            base_version.data, sort_keys=True
        ).splitlines()
        patched_data = difflib.restore(
            base_data, data['delta'].splitlines(), 1
        )
        current_data = json.loads('\n'.join(patched_data))
        instance = cls(current_data, base_version, data['metadata'])
        instance.timestamp = data['timestamp']
        instance.version_hash = data['version_hash']
        instance.delta = data['delta']
        return instance
```

`save` 方法现在只存储增量数据和元数据，显著减少了大型数据集的文件大小。`load` 方法通过将增量应用于基础版本来重建完整数据集。这种方法允许高效地存储和检索大型语言数据集的多个版本。

# 数据版本管理工具

虽然自定义解决方案可能有效，但还有专门为机器学习项目中的数据版本管理设计的工具。其中一个这样的工具是**数据版本控制**（**DVC**），它与 Git 集成，并为管理大型数据集和机器学习工件提供了强大的功能，并且被广泛使用。DVC 是一个开源工具，它扩展了 Git 以管理大型数据集和机器学习工件，通过在外部存储中存储数据，同时在 Git 仓库中跟踪元数据。它使可重复的管道、高效的数据共享和实验跟踪成为可能，使其成为管理 LLM 数据集和训练工作流程的热门选择。

由于 LLM 模型规模庞大，DVC 的版本化方法必须仔细平衡全面的跟踪与计算效率，需要智能的校验和元数据计算策略，以最小化延迟和处理开销，防止版本管理成为模型开发工作流程的瓶颈。

这里有一个如何在您的 LLM 项目中使用 DVC 的示例：

```py
import subprocess
def initialize_dvc():
    subprocess.run(["dvc", "init"])
    print("DVC initialized in the current directory.")
def add_dataset_to_dvc(dataset_path):
    subprocess.run(["dvc", "add", dataset_path])
    print(f"Dataset {dataset_path} added to DVC.")
def commit_dataset_version(message):
    subprocess.run(["git", "add", ".dvc"])
    subprocess.run(["git", "commit", "-m", message])
    print(f"Dataset version committed with message: {message}")
```

此脚本部分演示了如何初始化 DVC，将数据集添加到 DVC 跟踪中，并提交数据集的新版本。DVC 与 Git 一起工作，允许你以与版本化代码类似的方式版本化你的数据。

与 Git 类似，DVC 使用 `init`、`add`、`commit` 和 `push` 命令。以下列表简要描述了每个命令：

+   `dvc init`：通过在项目中创建 `.dvc` 目录并设置必要的元数据跟踪基础设施来初始化新的 DVC 项目。这类似于 `git init`，但专门用于数据版本控制，为跟踪大型数据集和模型文件准备你的项目。

+   `dvc add`：将大型数据文件添加到 DVC 跟踪中，创建一个轻量级的 `.dvc` 元数据文件，其中包含文件的哈希值。此命令将实际数据移动到单独的存储位置，同时在你的 Git 仓库中保持引用，允许你在不膨胀 Git 仓库的情况下对大型文件进行版本控制。

+   `dvc commit`：创建当前跟踪数据文件的快照，类似于 Git 提交，但专门用于数据文件。此命令帮助你标记数据历史的显著点，并创建一个清晰记录，说明数据集何时以及如何更改。

+   `dvc push`：将你的跟踪数据文件上传到远程存储位置（如云存储、网络驱动器或本地外部存储）。此命令确保你的数据版本安全备份，并且可以被其他团队成员或不同开发环境检索。

现在，让我们添加一个函数将数据集推送到远程存储：

```py
def push_dataset_to_remote():
    subprocess.run(["dvc", "push"])
    subprocess.run(["git", "push"])
    print("Dataset pushed to remote storage.")
# Usage example
if __name__ == "__main__":
    initialize_dvc()
    add_dataset_to_dvc("path/to/your/large_language_dataset.txt")
    commit_dataset_version("Add initial version of language dataset")
    push_dataset_to_remote()
```

`push_dataset_to_remote` 函数将 DVC 跟踪的数据和 Git 仓库推送到各自的远程存储位置。这允许你将大型数据集与代码仓库分开存储，同时保持对两者的版本控制。

接下来，我们将专注于在训练工作流程中集成数据版本控制。

# 在训练工作流程中集成数据版本控制

要使数据版本控制成为你 LLM 训练工作流程的组成部分，你需要将版本检查和记录集成到你的训练脚本中。以下是一个示例，说明你可能如何做到这一点：

```py
import json
from dataclasses import dataclass
from typing import Dict, Any
@dataclass
class DatasetInfo:
    version_hash: str
    metadata: Dict[str, Any]
def load_dataset_info(filename: str) -> DatasetInfo:
    with open(filename, 'r') as f:
        data = json.load(f)
    return DatasetInfo(data['version_hash'], data['metadata'])
def train_llm(model, dataset, dataset_info: DatasetInfo):
    # Log dataset version information
    print(
        f"Training model with dataset version: "
        f"{dataset_info.version_hash}"
    )
    print(f"Dataset metadata: {dataset_info.metadata}")
    # Actual training code would go here
    # ...
    # Save model with dataset version information
    model.save(f"model_trained_on_{dataset_info.version_hash[:8]}.pt")
```

此代码片段展示了如何将数据集版本信息集成到你的 LLM 训练工作流程中。`DatasetInfo` 类封装了基本版本信息，而 `load_dataset_info` 函数从 JSON 文件中检索这些信息。`train_llm` 函数演示了如何在训练期间记录数据集版本和元数据，确保每个训练模型都与特定版本的数据相关联。

在训练脚本中，你可能这样使用它：

```py
# Usage in training script
dataset_info = load_dataset_info("dataset_info.json")
dataset = load_dataset()  # Your dataset loading function
model = initialize_model()  # Your model initialization function
train_llm(model, dataset, dataset_info)
```

通过将数据集版本信息集成到你的训练过程中，你增强了可重复性，并使跟踪每个训练模型使用的数据版本变得更容易。

# 文本语料库的版本控制

当处理用于 LLM 训练的文本语料库时，你经常需要处理大量文档。以下是一个使用文件哈希和元数据跟踪的组合方法来对文本语料库进行版本控制的方法：

```py
import os
import hashlib
from typing import Dict, List
def hash_file(filepath: str) -> str:
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()
def generate_corpus_manifest(corpus_dir: str) -> Dict[str, str]:
    manifest = {}
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            filepath = os.path.join(root, file)
            manifest[os.path.relpath(filepath, corpus_dir)] = \
                hash_file(filepath)
    return manifest
```

这部分代码定义了函数来对单个文件进行哈希处理并生成语料库目录中所有文件的清单。清单是一个将相对文件路径映射到其对应哈希值的字典，提供了整个语料库的快照。清单文件很重要，因为它作为整个数据集的紧凑、可重复的指纹，使得快速完整性检查、促进版本跟踪，并允许研究人员在不同的环境或时间点验证其语料库的确切状态，而无需存储或传输整个大型数据集。

现在，让我们添加一个函数来比较两个清单并识别变化：

```py
def compare_manifests(
    old_manifest: Dict[str, str], new_manifest: Dict[str, str]
) -> Dict[str, List[str]]:
    changes = {
        "added": [],
        "removed": [],
        "modified": []
    }
    for file, hash in new_manifest.items():
        if file not in old_manifest:
            changes["added"].append(file)
        elif old_manifest[file] != hash:
            changes["modified"].append(file)
    for file in old_manifest:
        if file not in new_manifest:
            changes["removed"].append(file)
    return changes
# Usage example
old_manifest = generate_corpus_manifest("path/to/old_corpus")
new_manifest = generate_corpus_manifest("path/to/new_corpus")
changes = compare_manifests(old_manifest, new_manifest)
print("Corpus changes:")
for change_type, files in changes.items():
    print(f"{change_type.capitalize()}:")
    for file in files:
        print(f"  - {file}")
```

`compare_manifests` 函数识别语料库两个版本之间添加的、删除的和修改的文件。这种方法允许你有效地跟踪文本语料库中的变化，即使处理大量文件也是如此。

# 管理数据集变体和实验

在 LLM 开发中，你经常需要管理数据集的多个变体以进行不同的实验。以下是一个简单的系统来管理数据集变体：

```py
from typing import Dict, Any
import json
import os
class DatasetVariantManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.variants: Dict[str, Dict[str, Any]] = {}
        self._load_variants()
    def _load_variants(self):
        if os.path.exists(
            os.path.join(self.base_path, "variants.json")
        ):
            with open(
                os.path.join(self.base_path, "variants.json"), 'r'
            ) as f:
                self.variants = json.load(f)
    def save_variants(self):
        with open(
            os.path.join(self.base_path, "variants.json"), 'w'
        ) as f:
            json.dump(self.variants, f, indent=2)
```

`DatasetVariantManager` 类的这一部分设置了管理数据集变体的基本结构。它使用基础路径初始化管理器，并在可用的情况下从 JSON 文件中加载现有变体。

现在，让我们添加创建和检索变体的方法：

```py
class DatasetVariantManager:
    # ... (previous methods)
    def create_variant(
        self, name: str, base_variant: str, changes: Dict[str, Any]
    ):
        if name in self.variants:
            raise ValueError(f"Variant {name} already exists")
        self.variants[name] = {
            "base": base_variant,
            "changes": changes
        }
        self.save_variants()
    def get_variant(self, name: str) -> Dict[str, Any]:
        if name not in self.variants:
            raise ValueError(f"Variant {name} does not exist")
        variant = self.variants[name]
        base_data = self.get_variant(variant["base"]) 
            if variant["base"] else {}
        return {base_data, variant["changes"]}
# Usage example
manager = DatasetVariantManager("path/to/dataset/variants")
manager.create_variant(
    "base", None, {"size": 1000000, "language": "en"})
manager.create_variant("large", "base", {"size": 5000000})
manager.create_variant(
    "multilingual", "large", {"language": ["en", "es", "fr"]})
print(manager.get_variant("multilingual"))
```

`create_variant` 方法允许你根据现有数据集创建新的数据集变体，只需指定更改即可。`get_variant` 方法检索一个变体，递归地应用其基础变体的所有更改。这个系统允许你有效地管理和跟踪数据集的不同配置，以进行各种实验。

在 LLM 开发中管理数据集变体时，建议使用清晰和一致的名字约定，以确保可追溯性、可重复性和清晰性。以下是一个建议的名字约定，它平衡了可读性和可扩展性，用于管理数据集变体：

`<``base>_<modifier1>_<modifier2>_..._<description>`

此格式使用 **基础名称** 来指示根数据集，后跟 **修饰符** 和可选的描述来指定区分变体的变化或属性。修饰符简洁且按层次顺序排列，以反映转换过程。

让我们仔细看看关键组件：

+   `base` 或描述性名称（例如，`clean` 或 `raw`）。

+   **修饰符**：应用于基础数据的顺序变化或转换。每个修饰符反映数据集的一个方面，例如大小、语言或应用的前处理。

+   **描述**：一个可选部分，提供关于更改的额外上下文或详细信息，通常用于实验。

# 数据版本化的最佳实践

多年来，我收集了以下最佳实践：

+   对于大规模项目，使用专用的数据版本化工具，如 DVC。

+   在您的模型元数据中包含数据集版本信息。

+   对于大型数据集，使用基于 delta 的版本化以节省存储空间。

+   定期备份您的已版本化数据集。

+   为数据集版本和变体使用一致的命名约定。

+   将数据版本检查集成到您的 `dvc status` 中，以验证是否发生了意外的修改，自动将数据集校验和与批准版本进行比较，并在检测到任何数据差异时阻止模型训练。关键步骤包括创建一个预训练验证阶段，比较当前数据集版本与预期参考版本，在检测到未经验证的数据修改时自动触发警报或停止管道，并在机器学习开发过程中维护数据集变化的全面审计记录。

# 摘要

在本章中，我们探讨了 LLM 开发中数据版本化的各个方面。我们实现了基本的版本化系统和针对大型数据集的基于 delta 的版本化。我们检查了 DVC 等工具以满足更高级的版本化需求。我们还探讨了将数据版本化集成到 LLM 训练工作流程中、管理文本语料库版本以及处理实验数据集变体的方法。

数据版本化是 LLM 开发中的关键实践，确保可重复性、促进协作并使模型治理更加稳健。通过实施这些技术和最佳实践，您可以显著提高 LLM 项目的可管理性和可靠性。

在即将到来的章节中，我们将探讨针对 LLMs 特别定制的数据集标注和标记技术。特别是，我们将介绍高效标注策略、质量控制措施以及将标注过程扩展以满足大型语言数据集需求的方法。
