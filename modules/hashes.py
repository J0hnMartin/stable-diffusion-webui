import copy
import hashlib
import os.path
<<<<<<< HEAD
from rich import progress, errors
=======

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
from modules import shared
import modules.cache

<<<<<<< HEAD
cache_filename = os.path.join(data_path, "cache.json")
cache_data = None
progress_ok = True

def dump_cache():
    shared.writefile(cache_data, cache_filename)


def cache(subsection):
    global cache_data # pylint: disable=global-statement
    if cache_data is None:
        cache_data = {} if not os.path.isfile(cache_filename) else shared.readfile(cache_filename, lock=True)
    s = cache_data.get(subsection, {})
    cache_data[subsection] = s
    return s
=======
dump_cache = modules.cache.dump_cache
cache = modules.cache.cache
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e


def calculate_sha256(filename, quiet=False):
    global progress_ok # pylint: disable=global-statement
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    if not quiet:
        if progress_ok:
            try:
                with progress.open(filename, 'rb', description=f'[cyan]Calculating hash: [yellow]{filename}', auto_refresh=True, console=shared.console) as f:
                    for chunk in iter(lambda: f.read(blksize), b""):
                        hash_sha256.update(chunk)
            except errors.LiveError:
                shared.log.warning('Hash: attempting to use function in a thread')
                progress_ok = False
        if not progress_ok:
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(blksize), b""):
                    hash_sha256.update(chunk)
    else:
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
<<<<<<< HEAD
=======
    ondisk_mtime = os.path.getmtime(filename)

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    if title not in hashes:
        return None
    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)
    ondisk_mtime = os.path.getmtime(filename) if os.path.isfile(filename) else 0
    if ondisk_mtime > cached_mtime or cached_sha256 is None:
        return None
    return cached_sha256


def sha256(filename, title, use_addnet_hash=False):
<<<<<<< HEAD
    global progress_ok # pylint: disable=global-statement
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
=======
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value
    if shared.cmd_opts.no_hashing:
        return None
<<<<<<< HEAD
    if not os.path.isfile(filename):
        return None
    orig_state = copy.deepcopy(shared.state)
    shared.state.begin("hash")
    if use_addnet_hash:
        if progress_ok:
            try:
                with progress.open(filename, 'rb', description=f'[cyan]Calculating hash: [yellow]{filename}', auto_refresh=True, console=shared.console) as f:
                    sha256_value = addnet_hash_safetensors(f)
            except errors.LiveError:
                shared.log.warning('Hash: attempting to use function in a thread')
                progress_ok = False
        if not progress_ok:
            with open(filename, 'rb') as f:
                sha256_value = addnet_hash_safetensors(f)
    else:
        sha256_value = calculate_sha256(filename)
=======

    print(f"Calculating sha256 for {filename}: ", end='')
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value
    }
    shared.state.end()
    shared.state = orig_state
    dump_cache()
    return sha256_value


def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
<<<<<<< HEAD
    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")
=======

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)
<<<<<<< HEAD
    return hash_sha256.hexdigest()
=======

    return hash_sha256.hexdigest()

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
