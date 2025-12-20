#!/bin/bash

# NeuTTS Air 后台运行脚本
# 用法: ./run.sh [start|stop|restart|status|logs]

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.neutts.pid"
LOG_FILE="$SCRIPT_DIR/neutts.log"
PYTHON_SCRIPT="$SCRIPT_DIR/server.py"
VENV_PATH="$SCRIPT_DIR/.venv"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 激活虚拟环境（如果存在）
activate_venv() {
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
        echo -e "${GREEN}已激活虚拟环境: $VENV_PATH${NC}"
    fi
}

# 获取进程ID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

# 检查进程是否运行
is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# 启动服务
start() {
    if is_running; then
        echo -e "${YELLOW}NeuTTS Air 已经在运行中 (PID: $(get_pid))${NC}"
        return 1
    fi

    echo -e "${GREEN}正在启动 NeuTTS Air...${NC}"

    # 激活虚拟环境
    activate_venv

    # 切换到脚本目录
    cd "$SCRIPT_DIR"

    # 后台启动服务，输出重定向到日志文件
    nohup python "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1 &

    # 保存PID
    echo $! > "$PID_FILE"

    # 等待一下检查是否启动成功
    sleep 2

    if is_running; then
        echo -e "${GREEN}NeuTTS Air 启动成功!${NC}"
        echo -e "  PID: $(get_pid)"
        echo -e "  日志: $LOG_FILE"
        echo -e "  访问: http://localhost:9001"
    else
        echo -e "${RED}NeuTTS Air 启动失败，请查看日志: $LOG_FILE${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

# 停止服务
stop() {
    if ! is_running; then
        echo -e "${YELLOW}NeuTTS Air 未在运行${NC}"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid=$(get_pid)
    echo -e "${YELLOW}正在停止 NeuTTS Air (PID: $pid)...${NC}"

    # 发送 SIGTERM 信号
    kill "$pid" 2>/dev/null

    # 等待进程结束
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    # 如果还在运行，强制终止
    if is_running; then
        echo -e "${YELLOW}进程未响应，强制终止...${NC}"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi

    rm -f "$PID_FILE"
    echo -e "${GREEN}NeuTTS Air 已停止${NC}"
}

# 重启服务
restart() {
    echo -e "${YELLOW}正在重启 NeuTTS Air...${NC}"
    stop
    sleep 1
    start
}

# 查看状态
status() {
    if is_running; then
        local pid=$(get_pid)
        echo -e "${GREEN}NeuTTS Air 正在运行${NC}"
        echo -e "  PID: $pid"
        echo -e "  日志: $LOG_FILE"

        # 显示进程信息
        if command -v ps &> /dev/null; then
            echo -e "\n进程信息:"
            ps -p "$pid" -o pid,ppid,%cpu,%mem,etime,command 2>/dev/null
        fi
    else
        echo -e "${RED}NeuTTS Air 未在运行${NC}"
        return 1
    fi
}

# 查看日志
logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${GREEN}显示最近 50 行日志 (Ctrl+C 退出实时查看):${NC}"
        echo "----------------------------------------"
        tail -n 50 -f "$LOG_FILE"
    else
        echo -e "${YELLOW}日志文件不存在: $LOG_FILE${NC}"
    fi
}

# 显示帮助
usage() {
    echo "NeuTTS Air 后台运行脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start    启动服务"
    echo "  stop     停止服务"
    echo "  restart  重启服务"
    echo "  status   查看运行状态"
    echo "  logs     查看日志 (实时)"
    echo ""
    echo "示例:"
    echo "  $0 start     # 启动服务"
    echo "  $0 logs      # 查看日志"
}

# 主入口
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        usage
        exit 1
        ;;
esac
