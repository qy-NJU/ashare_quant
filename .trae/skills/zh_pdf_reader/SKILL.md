---
name: "zh_pdf_reader"
description: "提取中文PDF文本并进行精简总结。当用户提供PDF文件路径并要求读取、解析或总结其中文内容时，必须立即调用此技能。"
---

# 中文 PDF 阅读与总结助手 (zh_pdf_reader)

此技能专门用于协助用户读取中文 PDF 文件并进行内容精简总结。请严格按照以下步骤操作：

## 1. 确认与准备
- **确认路径**：确保用户提供了需要读取的 PDF 文件的绝对路径。如果只提供了相对路径或文件名，请结合当前工作目录（CWD）补全。
- **环境检查**：检查当前 Python 环境是否安装了 PDF 解析库（如 `PyMuPDF` / `fitz` 或 `pdfplumber`）。如果不确定，可使用 `RunCommand` 工具尝试导入或安装：`pip install pymupdf`。

## 2. 提取文本内容
由于无法直接使用常规的文本读取工具读取 PDF 文件，请编写一个简短的 Python 脚本来提取文本，并将其输出到临时文件。
- 编写一个 Python 脚本来提取文本（例如命名为 `extract_pdf.py`）。
- 使用 `RunCommand` 工具执行该脚本。为了避免标准输出过长被截断，务必将提取的文本写入到一个临时文本文件（如 `temp_pdf_content.txt`）中。

*Python 脚本参考：*
```python
import fitz  # PyMuPDF
import sys

def extract_pdf(file_path, output_path):
    try:
        doc = fitz.open(file_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for page in doc:
                f.write(page.get_text())
        print(f"Successfully extracted text to {output_path}")
    except Exception as e:
        print(f"Error extracting PDF: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_pdf.py <input.pdf> <output.txt>")
        sys.exit(1)
    extract_pdf(sys.argv[1], sys.argv[2])
```

## 3. 读取与总结
- 使用 `Read` 工具读取生成的临时文本文件（`temp_pdf_content.txt`）。如果文件很大，可结合 `limit` 和 `offset` 参数分段读取。
- **仔细阅读**提取出的中文文本，准确理解上下文。
- **输出规范**：
  - 使用中文回复。
  - 使用清晰的结构（如无序列表、加粗标题）输出精简的总结。
  - 总结应包含：文档的主题/目的、核心观点、关键数据或核心结论。
  - 如果用户有针对该 PDF 的特定问题，请结合提取的内容进行精准回答。
- **清理**：任务完成后，使用 `DeleteFile` 工具或 `RunCommand` 清理生成的临时文本文件和 Python 脚本。
