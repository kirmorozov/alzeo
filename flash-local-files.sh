#!/bin/bash
# flash-local-files.sh <index> <copy>...: what the Flash FP4-Local lane's engine reads has to be there
# before it runs, and its unit asks this before the engine, as its main process (the head of its
# ExecStart, after the memory guard every lane runs).
#
# The checkpoint at its pinned revision, the copies of its n-gram table the unit mounts, and, when
# the unit serves with the dflash2 drafter, the drafter's snapshot at its pinned revision, its
# config and its one weights file, non-empty, and the capture plugin's files, every one the repo's
# folder has, in the copy the unit mounts read-only at /opt/df2-plugin. A start
# without them made runc create what was missing, owned by root: the checkpoint's folders inside the
# Hugging Face cache, which install.sh then read as a lane begun, and a folder in place of a missing
# copy (found in review, proved on the reference box, 2026-10-20). The checkpoint is its index and
# every file the index names: a fetch cut short leaves the small index before the large files, and
# the engine then died at every start, which systemd started again every 15 s (found in review,
# 2026-10-20). A copy has its original's size in the cache: an empty one, or one cut short, is no
# copy, and a folder left in a copy's place is none either. The files the dflash2 unit asks for have
# ordinary names, and are whole when they are there and non-empty: a fetch cut short links none of
# them, and a folder left in one's place is no file. A cache on a disk mounted late at boot
# gets 90 s to come, counted in tries rather than read off the clock, which can jump at boot.
# Without them this says what is missing and what brings it back, the disk first (a Load with it
# away fetches the checkpoint onto the system's disk), on a line the cockpit reads ("preflight:
# <unit> lacks its files"), and exits 78, which the unit's RestartPreventExitStatus holds for its
# main process: from an ExecStartPre= the same 78 was started again every 15 s (found in review,
# 2026-10-20).
#
# A script of its own, as flash-ats-check.sh is the flash lane's: engine-inputs.py leaves it out of
# what the engine reads (NOT_INPUTS), so a new wording of it never asks a serving lane to restart.
# FLASH_LOCAL_FILES_TRIES sets the tries of 3 s (30 by default, the tests' to shorten).
set -u
tries="${FLASH_LOCAL_FILES_TRIES:-30}"

# The first of the files given that is missing or not whole, said on stdout; nothing when all are.
first_missing() {
  local f blobs="" orig
  for f in "$@"; do
    if [ ! -f "$f" ] || [ ! -s "$f" ]; then
      echo "$f"; return
    fi
    case "$f" in
      */snapshots/*/model.safetensors.index.json)
        # the copies are named by their sha256, as the cache's blobs they copy
        blobs="${f%/snapshots/*}/blobs"
        python3 -I - "$f" <<'EOF'
import json, os, sys
index = sys.argv[1]
try:
    shards = sorted(set(json.load(open(index))["weight_map"].values()))
except (OSError, ValueError, KeyError, AttributeError, TypeError):
    print(index)
    sys.exit()
for shard in shards:
    path = os.path.join(os.path.dirname(index), shard)
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        print(path)
        break
EOF
        ;;
      *)
        # The names of 64 hex are the copies, the checkpoint's files by their sha256: only they are
        # held to their original's size in the checkpoint's blobs. The files the dflash2 unit asks
        # for, the drafter's snapshot and the plugin's, have ordinary names and are whole by the
        # non-empty check above: compared against the checkpoint's blobs, the drafter's config.json
        # would read missing there, which it is
        if [[ "${f##*/}" =~ ^[0-9a-f]{64}$ ]]; then
          orig="$blobs/${f##*/}"
          if [ -n "$blobs" ] && { [ ! -f "$orig" ] || [ "$(stat -c %s "$f")" != "$(stat -L -c %s "$orig")" ]; }; then
            echo "$f"; return
          fi
        fi
        ;;
    esac
  done
}

missing=""
for ((t = 0; t <= tries; t++)); do
  missing="$(first_missing "$@" | head -1)"
  [ -z "$missing" ] && exit 0
  [ "$t" -lt "$tries" ] && sleep 3
done
# The what the line names, of what the lane reads: today's pair, and the drafter's snapshot and the
# capture plugin too when this run was asked for them, by the paths the unit gives them: the mtp
# unit asks for neither, and its line stays today's, byte for byte (the drafter's snapshot is in
# the same cache the checkpoint is in, the plugin in the config directory the unit mounts it from)
what="its checkpoint, or a copy of its n-gram table"
target="flash-local"
for f in "$@"; do
  case "$f" in
    *DFlash2*|*/flash-local-dflash2/*)
      # the drafter's snapshot by its arch, whatever repo owns it, and the plugin's copy by its
      # folder: the two files only the dflash2 unit is ever asked for
      what="its checkpoint, a copy of its n-gram table, its DFlash2 drafter's snapshot, or its capture plugin"
      target="flash-local-dflash2" ;;
  esac
done
# The target named is the one this start was asked for: the drafter's files are missing when the
# dflash2 target started without them, and a line that then said ./switch-model.sh flash-local
# sent the operator to the MTP target, which starts fine and buries what was actually missing
# (found on the reference box, 2026-11-07)
echo "preflight: qwen38-flash-local.service lacks its files: $missing is missing or not whole ($what), or the" \
     "disk that holds the Hugging Face cache is not mounted. With that disk mounted, Load the lane from" \
     "the cockpit, or run ./switch-model.sh $target" >&2
exit 78
