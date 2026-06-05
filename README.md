# PDF Watermark Cleaner WebUI Realtime

变更：

- Docker 外部端口示例改为 `8104`
- 新增 `/preview` 实时处理预览接口
- WebUI 右侧显示真实处理后的页面预览，不再只是区域示意图
- 新增 PDF 文本对象水印删除：
  - 精确匹配 `exact`
  - 包含匹配 `contains`
  - 正则匹配 `regex`
  - 可限制页码
  - 可限制坐标区域

## Docker 部署

```bash
docker build -t pdf-watermark-cleaner-webui .

docker run -d \
  --name pdf-watermark-cleaner-webui \
  --restart unless-stopped \
  -p 8104:8000 \
  -v pdf-watermark-jobs:/data/jobs \
  pdf-watermark-cleaner-webui
```

打开：

```text
http://服务器IP:8104/
```

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 文本水印规则示例

```json
"text_rules": [
  {"text": "天津考生", "mode": "contains", "ignore_case": false, "pages": [], "rect": null},
  {"text": "仅供学习交流使用", "mode": "contains", "ignore_case": false, "pages": [], "rect": null},
  {"text": "在.*领取更多资料", "mode": "regex", "ignore_case": false, "pages": [], "rect": null}
]
```

## API

处理并下载：

```bash
curl -L -o cleaned.pdf \
  -F "file=@input.pdf" \
  -F "options=$(cat options.json)" \
  http://127.0.0.1:8104/process
```

实时预览第 1 页：

```bash
curl -L -o preview.png \
  -F "file=@input.pdf" \
  -F "page=1" \
  -F "options=$(cat options.json)" \
  http://127.0.0.1:8104/preview
```

说明：输出 PDF 仍是图片型 PDF。如果输入 PDF 原本有文本层，处理后不会保留可搜索文本层。
