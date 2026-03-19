import logging
import sys


class LoguruStyleFormatter(logging.Formatter):
    """
    修复后的 Loguru 风格 Formatter
    """

    # ANSI 颜色码
    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record):
        # 1. 预处理时间戳 (添加毫秒)
        log_time = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        record.full_time = f"{log_time}.{int(record.msecs):03d}"

        # 2. 获取当前级别的颜色
        level_color = self.LEVEL_COLORS.get(record.levelno, self.GREY)

        # 3. 预处理级别名称，使其对齐并带颜色
        # 模仿 Loguru 的 | INFO     | 效果
        record.colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"

        # 4. 预处理源码位置颜色
        record.colored_name = f"{self.CYAN}{record.name}{self.RESET}"
        record.colored_func = f"{self.CYAN}{record.funcName}{self.RESET}"
        record.colored_line = f"{self.CYAN}{record.lineno}{self.RESET}"

        # 5. 消息颜色
        record.colored_msg = f"{level_color}{record.getMessage()}{self.RESET}"

        # 最终组合格式
        # 注意：这里直接手动拼装字符串，不再通过 logging.Formatter 的二次 format
        format_str = (
            f"{self.GREEN}{record.full_time}{self.RESET} | "  # pyright: ignore
            f"{record.colored_level} | "  # pyright: ignore
            f"{record.colored_name}:{record.colored_func}:{record.colored_line} - "  # pyright: ignore
            f"{record.colored_msg}"  # pyright: ignore
        )

        # 处理异常堆栈信息 (如果有)
        if record.exc_info:
            format_str += "\n" + self.formatException(record.exc_info)

        return format_str


def setup_logger(name: str, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(LoguruStyleFormatter())
    logger.addHandler(stdout_handler)

    return logger
