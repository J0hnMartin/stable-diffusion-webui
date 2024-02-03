from functools import wraps
import html
<<<<<<< HEAD
import threading
import time
import cProfile
from modules import shared, progress, errors
=======
import time

from modules import shared, progress, errors, devices, fifo_lock
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e

queue_lock = fifo_lock.FIFOLock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)
        return res
    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
<<<<<<< HEAD
    name = func.__name__
=======
    @wraps(func)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    def f(*args, **kwargs):
        # if the first argument is a string that says "task(...)", it is treated as a job id
        if args and type(args[0]) == str and args[0].startswith("task(") and args[0].endswith(")"):
            id_task = args[0]
            progress.add_task_to_queue(id_task)
        else:
            id_task = None
        with queue_lock:
<<<<<<< HEAD
=======
            shared.state.begin(job=id_task)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
            progress.start_task(id_task)
            res = [None, '', '', '']
            try:
                res = func(*args, **kwargs)
                progress.record_results(id_task, res)
<<<<<<< HEAD
            except Exception as e:
                shared.log.error(f"Exception: {e}")
                shared.log.error(f"Arguments: args={str(args)[:10240]} kwargs={str(kwargs)[:10240]}")
                errors.display(e, 'gradio call')
                res[-1] = f"<div class='error'>{html.escape(str(e))}</div>"
=======
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
            finally:
                progress.finish_task(id_task)
        return res
    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True, name=name)


<<<<<<< HEAD
def wrap_gradio_call(func, extra_outputs=None, add_stats=False, name=None):
    job_name = name if name is not None else func.__name__
=======
def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    @wraps(func)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        t = time.perf_counter()
        shared.mem_mon.reset()
        shared.state.begin(job_name)
        try:
            if shared.cmd_opts.profile:
                pr = cProfile.Profile()
                pr.enable()
            res = func(*args, **kwargs)
            if res is None:
                msg = "No result returned from function"
                shared.log.warning(msg)
                res = [None, '', '', f"<div class='error'>{html.escape(msg)}</div>"]
            else:
                res = list(res)
            if shared.cmd_opts.profile:
                errors.profile(pr, 'Wrap')
        except Exception as e:
<<<<<<< HEAD
            errors.display(e, 'gradio call')
            if extra_outputs_array is None:
                extra_outputs_array = [None, '']
            res = extra_outputs_array + [f"<div class='error'>{html.escape(type(e).__name__+': '+str(e))}</div>"]
        shared.state.end()
=======
            # When printing out our debug argument list,
            # do not print out more than a 100 KB of text
            max_debug_str_len = 131072
            message = "Error completing request"
            arg_str = f"Arguments: {args} {kwargs}"[:max_debug_str_len]
            if len(arg_str) > max_debug_str_len:
                arg_str += f" (Argument list truncated at {max_debug_str_len}/{len(arg_str)} characters)"
            errors.report(f"{message}\n{arg_str}", exc_info=True)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            error_message = f'{type(e).__name__}: {e}'
            res = extra_outputs_array + [f"<div class='error'>{html.escape(error_message)}</div>"]

        devices.torch_gc()

        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        if not add_stats:
            return tuple(res)
        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
<<<<<<< HEAD
        elapsed_text = f"{elapsed_m}m {elapsed_s:.2f}s" if elapsed_m > 0 else f"{elapsed_s:.2f}s"
        vram_html = ''
        if not shared.mem_mon.disabled:
            vram = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.read().items()}
            if vram.get('active_peak', 0) > 0:
                vram_html = " | <p class='vram'>"
                vram_html += f"GPU active {max(vram['active_peak'], vram['reserved_peak'])} MB reserved {vram['reserved']} | used {vram['used']} MB free {vram['free']} MB total {vram['total']} MB"
                vram_html += f" | retries {vram['retries']} oom {vram['oom']}" if vram.get('retries', 0) > 0 or vram.get('oom', 0) > 0 else ''
                vram_html += "</p>"
        if isinstance(res, list):
            res[-1] += f"<div class='performance'><p class='time'>Time: {elapsed_text}</p>{vram_html}</div>"
=======
        elapsed_text = f"{elapsed_s:.1f} sec."
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m} min. "+elapsed_text

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = sys_peak/max(sys_total, 1) * 100

            toltip_a = "Active: peak amount of video memory used during generation (excluding cached data)"
            toltip_r = "Reserved: total amout of video memory allocated by the Torch library "
            toltip_sys = "System: peak amout of video memory allocated by all running programs, out of total capacity"

            text_a = f"<abbr title='{toltip_a}'>A</abbr>: <span class='measurement'>{active_peak/1024:.2f} GB</span>"
            text_r = f"<abbr title='{toltip_r}'>R</abbr>: <span class='measurement'>{reserved_peak/1024:.2f} GB</span>"
            text_sys = f"<abbr title='{toltip_sys}'>Sys</abbr>: <span class='measurement'>{sys_peak/1024:.1f}/{sys_total/1024:g} GB</span> ({sys_pct:.1f}%)"

            vram_html = f"<p class='vram'>{text_a}, <wbr>{text_r}, <wbr>{text_sys}</p>"
        else:
            vram_html = ''

        # last item is always HTML
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{elapsed_text}</span></p>{vram_html}</div>"

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        return tuple(res)
    return f
