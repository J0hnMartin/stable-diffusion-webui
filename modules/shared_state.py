<<<<<<< HEAD
import os
import sys
import time
import datetime
from modules.errors import log


class State:
    skipped = False
    interrupted = False
    paused = False
    job = ""
    job_no = 0
    job_count = 0
    total_jobs = 0
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    id_live_preview = 0
    textinfo = None
    api = False
    time_start = None
    need_restart = False
    server_start = time.time()
    oom = False
    debug_output = os.environ.get('SD_STATE_DEBUG', None)

    def skip(self):
        log.debug('Requested skip')
        self.skipped = True

    def interrupt(self):
        log.debug('Requested interrupt')
        self.interrupted = True

    def pause(self):
        self.paused = not self.paused
        log.debug(f'Requested {"pause" if self.paused else "continue"}')

    def nextjob(self):
        self.do_set_current_image()
        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }
        return obj

    def begin(self, title="", api=None):
        import modules.devices
        self.total_jobs += 1
        self.current_image = None
        self.current_image_sampling_step = 0
        self.current_latent = None
        self.id_live_preview = 0
        self.interrupted = False
        self.job = title
        self.job_count = -1
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.paused = False
        self.sampling_step = 0
        self.skipped = False
        self.textinfo = None
        self.api = api if api is not None else self.api
        self.time_start = time.time()
        if self.debug_output:
            log.debug(f'State begin: {self.job}')
        modules.devices.torch_gc()

    def end(self, api=None):
        import modules.devices
        if self.time_start is None: # someone called end before being
            log.debug(f'Access state.end: {sys._getframe().f_back.f_code.co_name}') # pylint: disable=protected-access
            self.time_start = time.time()
        if self.debug_output:
            log.debug(f'State end: {self.job} time={time.time() - self.time_start:.2f}')
        self.job = ""
        self.job_count = 0
        self.job_no = 0
        self.paused = False
        self.interrupted = False
        self.skipped = False
        self.api = api if api is not None else self.api
        modules.devices.torch_gc()

    def set_current_image(self):
        from modules.shared import opts, cmd_opts
        """sets self.current_image from self.current_latent if enough sampling steps have been made after the last call to this"""
        if cmd_opts.lowvram or self.api:
            return
        if abs(self.sampling_step - self.current_image_sampling_step) >= opts.show_progress_every_n_steps and opts.live_previews_enable and opts.show_progress_every_n_steps > 0:
            self.do_set_current_image()

    def do_set_current_image(self):
        if self.current_latent is None or self.api:
            return
        from modules.shared import opts
        import modules.sd_samplers # pylint: disable=W0621
        try:
            image = modules.sd_samplers.samples_to_image_grid(self.current_latent) if opts.show_progress_grid else modules.sd_samplers.sample_to_image(self.current_latent)
            self.assign_current_image(image)
            self.current_image_sampling_step = self.sampling_step
        except Exception:
            # log.error(f'Error setting current image: step={self.sampling_step} {e}')
            pass

    def assign_current_image(self, image):
        self.current_image = image
        self.id_live_preview += 1
=======
import datetime
import logging
import threading
import time

from modules import errors, shared, devices
from typing import Optional

log = logging.getLogger(__name__)


class State:
    skipped = False
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    processing_has_refined_job_count = False
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    id_live_preview = 0
    textinfo = None
    time_start = None
    server_start = None
    _server_command_signal = threading.Event()
    _server_command: Optional[str] = None

    def __init__(self):
        self.server_start = time.time()

    @property
    def need_restart(self) -> bool:
        # Compatibility getter for need_restart.
        return self.server_command == "restart"

    @need_restart.setter
    def need_restart(self, value: bool) -> None:
        # Compatibility setter for need_restart.
        if value:
            self.server_command = "restart"

    @property
    def server_command(self):
        return self._server_command

    @server_command.setter
    def server_command(self, value: Optional[str]) -> None:
        """
        Set the server command to `value` and signal that it's been set.
        """
        self._server_command = value
        self._server_command_signal.set()

    def wait_for_server_command(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wait for server command to get set; return and clear the value and signal.
        """
        if self._server_command_signal.wait(timeout):
            self._server_command_signal.clear()
            req = self._server_command
            self._server_command = None
            return req
        return None

    def request_restart(self) -> None:
        self.interrupt()
        self.server_command = "restart"
        log.info("Received restart request")

    def skip(self):
        self.skipped = True
        log.info("Received skip request")

    def interrupt(self):
        self.interrupted = True
        log.info("Received interrupt request")

    def nextjob(self):
        if shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps == -1:
            self.do_set_current_image()

        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }

        return obj

    def begin(self, job: str = "(unknown)"):
        self.sampling_step = 0
        self.time_start = time.time()
        self.job_count = -1
        self.processing_has_refined_job_count = False
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent = None
        self.current_image = None
        self.current_image_sampling_step = 0
        self.id_live_preview = 0
        self.skipped = False
        self.interrupted = False
        self.textinfo = None
        self.job = job
        devices.torch_gc()
        log.info("Starting job %s", job)

    def end(self):
        duration = time.time() - self.time_start
        log.info("Ending job %s (%.2f seconds)", self.job, duration)
        self.job = ""
        self.job_count = 0

        devices.torch_gc()

    def set_current_image(self):
        """if enough sampling steps have been made after the last call to this, sets self.current_image from self.current_latent, and modifies self.id_live_preview accordingly"""
        if not shared.parallel_processing_allowed:
            return

        if self.sampling_step - self.current_image_sampling_step >= shared.opts.show_progress_every_n_steps and shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps != -1:
            self.do_set_current_image()

    def do_set_current_image(self):
        if self.current_latent is None:
            return

        import modules.sd_samplers

        try:
            if shared.opts.show_progress_grid:
                self.assign_current_image(modules.sd_samplers.samples_to_image_grid(self.current_latent))
            else:
                self.assign_current_image(modules.sd_samplers.sample_to_image(self.current_latent))

            self.current_image_sampling_step = self.sampling_step

        except Exception:
            # when switching models during genration, VAE would be on CPU, so creating an image will fail.
            # we silently ignore this error
            errors.record_exception()

    def assign_current_image(self, image):
        self.current_image = image
        self.id_live_preview += 1
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
