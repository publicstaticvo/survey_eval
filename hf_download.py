import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--modelscope", action="store_true")
args = parser.parse_args()
if args.modelscope:
    from modelscope.hub.snapshot_download import snapshot_download
    snapshot_download(args.path, local_dir=args.output, ignore_patterns=["*.msgpack", "*.h5"])
else:
    from huggingface_hub import snapshot_download
    snapshot_download(args.path, local_dir=args.output, local_dir_use_symlinks=False, ignore_patterns=["*.msgpack", "*.h5"])