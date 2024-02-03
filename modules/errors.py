<<<<<<< HEAD
import logging
import warnings
from rich.console import Console
from rich.theme import Theme
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from installer import log as installer_log, setup_logging


setup_logging()
log = installer_log
console = Console(log_time=True, log_time_format='%H:%M:%S-%f', theme=Theme({
    "traceback.border": "black",
    "traceback.border.syntax_error": "black",
    "inspect.value.border": "black",
}))

pretty_install(console=console)
traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False)
already_displayed = {}


def install(suppress=[]): # noqa: B006
    warnings.filterwarnings("ignore", category=UserWarning)
    pretty_install(console=console)
    traceback_install(console=console, extra_lines=1, width=console.width, word_wrap=False, indent_guides=False, suppress=suppress)
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s')
    # for handler in logging.getLogger().handlers:
    #    handler.setLevel(logging.INFO)
=======
import sys
import textwrap
import traceback
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


exception_records = []


def format_traceback(tb):
    return [[f"{x.filename}, line {x.lineno}, {x.name}", x.line] for x in traceback.extract_tb(tb)]


def format_exception(e, tb):
    return {"exception": str(e), "traceback": format_traceback(tb)}


def get_exceptions():
    try:
        return list(reversed(exception_records))
    except Exception as e:
        return str(e)


def record_exception():
    _, e, tb = sys.exc_info()
    if e is None:
        return

    if exception_records and exception_records[-1] == e:
        return

    exception_records.append(format_exception(e, tb))

    if len(exception_records) > 5:
        exception_records.pop(0)


def report(message: str, *, exc_info: bool = False) -> None:
    """
    Print an error message to stderr, with optional traceback.
    """

    record_exception()

    for line in message.splitlines():
        print("***", line, file=sys.stderr)
    if exc_info:
        print(textwrap.indent(traceback.format_exc(), "    "), file=sys.stderr)
        print("---", file=sys.stderr)


def print_error_explanation(message):
    record_exception()

    lines = message.strip().split("\n")
    for line in lines:
        log.error(line)


<<<<<<< HEAD
def display(e: Exception, task, suppress=[]): # noqa: B006
    log.error(f"{task or 'error'}: {type(e).__name__}")
    console.print_exception(show_locals=False, max_frames=10, extra_lines=1, suppress=suppress, theme="ansi_dark", word_wrap=False, width=min([console.width, 200]))
=======
def display(e: Exception, task, *, full_traceback=False):
    record_exception()

    print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
    te = traceback.TracebackException.from_exception(e)
    if full_traceback:
        # include frames leading up to the try-catch block
        te.stack = traceback.StackSummary(traceback.extract_stack()[:-2] + te.stack)
    print(*te.format(), sep="", file=sys.stderr)

    message = str(e)
    if "copying a param with shape torch.Size([640, 1024]) from checkpoint, the shape in current model is torch.Size([640, 768])" in message:
        print_error_explanation("""
The most likely cause of this is you are trying to load Stable Diffusion 2.0 model without specifying its config file.
See https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20 for how to solve this.
        """)


already_displayed = {}
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def display_once(e: Exception, task):
    record_exception()

    if task in already_displayed:
        return
    display(e, task)
    already_displayed[task] = 1


def run(code, task):
    try:
        code()
    except Exception as e:
<<<<<<< HEAD
        display(e, task)


def exception(suppress=[]): # noqa: B006
    console.print_exception(show_locals=False, max_frames=10, extra_lines=2, suppress=suppress, theme="ansi_dark", word_wrap=False, width=min([console.width, 200]))


def profile(profiler, msg: str):
    profiler.disable()
    import io
    import pstats
    stream = io.StringIO() # pylint: disable=abstract-class-instantiated
    p = pstats.Stats(profiler, stream=stream)
    p.sort_stats(pstats.SortKey.CUMULATIVE)
    p.print_stats(100)
    # p.print_title()
    # p.print_call_heading(10, 'time')
    # p.print_callees(10)
    # p.print_callers(10)
    profiler = None
    lines = stream.getvalue().split('\n')
    lines = [x for x in lines if '<frozen' not in x
             and '{built-in' not in x
             and '/logging' not in x
             and 'Ordered by' not in x
             and 'List reduced' not in x
             and '_lsprof' not in x
             and '/profiler' not in x
             and 'rich' not in x
             and x.strip() != ''
            ]
    txt = '\n'.join(lines[:min(5, len(lines))])
    log.debug(f'Profile {msg}: {txt}')


def profile_torch(profiler, msg: str):
    profiler.stop()
    lines = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
    lines = lines.split('\n')
    lines = [x for x in lines if '/profiler' not in x and '---' not in x]
    txt = '\n'.join(lines)
    # print(f'Torch {msg}:', txt)
    log.debug(f'Torch profile {msg}: \n{txt}')
=======
        display(task, e)


def check_versions():
    from packaging import version
    from modules import shared

    import torch
    import gradio

    expected_torch_version = "2.0.0"
    expected_xformers_version = "0.0.20"
    expected_gradio_version = "3.41.2"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())

    if gradio.__version__ != expected_gradio_version:
        print_error_explanation(f"""
You are running gradio {gradio.__version__}.
The program is designed to work with gradio {expected_gradio_version}.
Using a different version of gradio is extremely likely to break the program.

Reasons why you have the mismatched gradio version can be:
  - you use --skip-install flag.
  - you use webui.py to start the program instead of launch.py.
  - an extension installs the incompatible gradio version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
