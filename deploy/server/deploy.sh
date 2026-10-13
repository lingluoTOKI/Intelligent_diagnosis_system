#!/usr/bin/env bash
# 服务器一键部署脚本（Ubuntu 22.04）
# 在解压后的部署包目录内执行：  bash deploy.sh
# 可用环境变量：APP_DIR（默认 /opt/eye-diagnosis）、PORT（默认 8000）、SKIP_SYSTEMD=1 只装不注册服务
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/eye-diagnosis}"
PORT="${PORT:-8000}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========== 眼科分类服务部署 =========="
echo "源目录: ${SOURCE_DIR}"
echo "目标目录: ${APP_DIR}"
echo "端口: ${PORT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "未找到 python3，请先安装：sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi
echo "Python: $(python3 --version)"

# 国内服务器访问官方源通常超时，默认走清华镜像，可用 PIP_INDEX 覆盖
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# 仅有 venv 模块还不够，ensurepip 缺失会导致创建失败（Ubuntu 需装 python3.x-venv）
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  PY_VER="$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  echo "缺少 ensurepip，安装 python${PY_VER}-venv ..."
  sudo apt update && sudo apt install -y "python${PY_VER}-venv"
fi

sudo mkdir -p "${APP_DIR}"
sudo cp -r "${SOURCE_DIR}/." "${APP_DIR}/"
sudo chown -R "$(id -un)":"$(id -gn)" "${APP_DIR}"

cd "${APP_DIR}"
# 之前创建失败的残缺 venv 会让重建失败，先清理
if [ -d "${APP_DIR}/venv" ]; then
  rm -rf "${APP_DIR}/venv"
fi
python3 -m venv venv
./venv/bin/pip install --upgrade pip -i "${PIP_INDEX}"
if ! ./venv/bin/pip install -r requirements-server.txt -i "${PIP_INDEX}"; then
  echo "镜像源安装失败，回退官方源重试"
  ./venv/bin/pip install -r requirements-server.txt
fi

mkdir -p "${APP_DIR}/runtime"

if [ "${SKIP_SYSTEMD:-0}" = "1" ]; then
  echo "已跳过 systemd 注册，手动启动命令："
  echo "  cd ${APP_DIR} && ./venv/bin/uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT}"
  exit 0
fi

SERVICE_FILE="/etc/systemd/system/eye-diagnosis.service"
sudo sed -e "s|__APP_DIR__|${APP_DIR}|g" \
         -e "s|__USER__|$(id -un)|g" \
         -e "s|__PORT__|${PORT}|g" \
         "${APP_DIR}/eye-diagnosis.service.example" | sudo tee "${SERVICE_FILE}" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable eye-diagnosis
sudo systemctl restart eye-diagnosis
sleep 3
sudo systemctl status eye-diagnosis --no-pager || true

echo "========== 部署完成 =========="
echo "本机自检: curl http://127.0.0.1:${PORT}/api/health"
echo "外网访问前需在阿里云控制台安全组放行 TCP ${PORT} 端口"
