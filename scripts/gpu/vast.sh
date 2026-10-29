#!/usr/bin/env bash
# Run GPU training jobs on a rented vast.ai instance. Run this script on your own machine.
#
# Needs: the vastai CLI (`uv tool install vastai`) with an API key (`vastai set api-key <key>`),
# jq, the frozen checkpoints (`uv run python -m fbplan.frozen`), and VAST_LABEL: the label that
# marks your instances. FBPLAN_CKPT_DIR, FBPLAN_BASELINE_DIR and FBPLAN_APPROACH_B_DIR move the
# local folders, as in the Python code.
#
#   scripts/gpu/vast.sh offers                         suitable offers, cheapest first
#   scripts/gpu/vast.sh create <offer-id>              rent one with label $VAST_LABEL and your SSH key
#   scripts/gpu/vast.sh list                           your instances (label $VAST_LABEL)
#   scripts/gpu/vast.sh push <instance-id>             code at HEAD + checkpoints, then remote setup
#   scripts/gpu/vast.sh train <instance-id> <env> [seed ...]   retrain pi_h in tmux (seeds 0-4)
#   scripts/gpu/vast.sh run <instance-id> <session> <command ...>   any command in tmux, in fbplan/
#   scripts/gpu/vast.sh status <instance-id>           GPU use and the last log line of each run
#   scripts/gpu/vast.sh pull <instance-id>             runs/ and the trained models
#   scripts/gpu/vast.sh ssh <instance-id> [command]    shell or one command on the instance
#   scripts/gpu/vast.sh destroy <instance-id> [-y]     delete the instance and its disk (stops billing)
#
# An API key can give access to instances of other people. Every command that takes an instance
# id refuses an instance whose label is not VAST_LABEL. The SSH key is attached to the new
# instance only, not to the account.
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
REMOTE_DIR=fbplan
CACHE=$HOME/.cache/fbplan
CKPT_DIR=${FBPLAN_CKPT_DIR:-$CACHE/checkpoints}
BASELINE_DIR=${FBPLAN_BASELINE_DIR:-$CACHE/baselines}
APPROACH_B_DIR=${FBPLAN_APPROACH_B_DIR:-$CACHE/approach_b}
SSH_PUBKEY=${SSH_PUBKEY:-$HOME/.ssh/id_rsa.pub}
# jaxlib 0.10.2 ships CUDA 12.9 libraries; an older driver may fail to run them. Hosts in CN
# are excluded: setup downloads from PyPI, GitHub and the OGBench server, which can be slow there.
GPU_QUERY=${GPU_QUERY:-'gpu_name in ["RTX 4090", "RTX 3090", "RTX A6000", "L40S"] num_gpus=1 reliability>0.98 cuda_vers>=12.8 disk_space>=40 inet_down>=200 direct_port_count>=1 geolocation notin [CN]'}
# The CUDA image sets NVIDIA_DRIVER_CAPABILITIES; JAX brings its own CUDA libraries via pip.
IMAGE=${IMAGE:-nvidia/cuda:12.8.1-base-ubuntu22.04}
# Instances are short-lived: keep their host keys out of the main known_hosts file.
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/known_hosts.vast")

die() {
  echo "error: $*" >&2
  exit 1
}

# label: VAST_LABEL, which every command that creates or touches an instance needs.
label() {
  [[ -n ${VAST_LABEL:-} ]] || die "set VAST_LABEL to the label of your instances"
  echo "$VAST_LABEL"
}

# require_ours <instance-id>: stop unless the instance exists and carries label VAST_LABEL.
require_ours() {
  local label expected
  expected=$(label)
  [[ $1 =~ ^[0-9]+$ ]] || die "instance id must be a number, got '$1'"
  label=$(vastai show instance "$1" --raw | jq -r '.label // ""') ||
    die "cannot read instance $1"
  [[ $label == "$expected" ]] || die "instance $1 has label '$label', not '$expected'"
}

# remote <instance-id> [command ...]: run a command on the instance over SSH.
# Goes through vast.ai's SSH proxy (ssh_host:ssh_port), not the direct port that `vastai ssh-url`
# gives: on the direct port, uploads of 20 MB or more broke, on the proxy 60 MB passed both ways.
remote() {
  local info
  info=$(vastai show instance "$1" --raw)
  shift
  ssh "${SSH_OPTS[@]}" -p "$(jq -r .ssh_port <<<"$info")" "root@$(jq -r .ssh_host <<<"$info")" "$@"
}

wait_for_ssh() {
  for _ in $(seq 60); do
    remote "$1" true 2>/dev/null && return 0
    sleep 10
  done
  die "instance $1 does not accept SSH after 10 min; check '$0 list'"
}

cmd_offers() {
  vastai search offers "$GPU_QUERY" -o dph | head -n 20
}

cmd_create() {
  local out id
  [[ -f $SSH_PUBKEY ]] || die "no public key at $SSH_PUBKEY"
  out=$(vastai create instance "$1" --image "$IMAGE" --disk 40 --ssh --direct \
    --label "$(label)" --raw)
  id=$(jq -r '.new_contract // empty' <<<"$out")
  [[ -n $id ]] || die "create failed: $out"
  echo "created instance $id (label $(label)); it bills until: $0 destroy $id"
  vastai attach ssh "$id" "$(cat "$SSH_PUBKEY")" ||
    die "could not attach $SSH_PUBKEY to instance $id; retry the attach or destroy it"
  echo "next: $0 push $id"
}

cmd_list() {
  vastai show instances --label "$(label)"
}

# The instance gets exactly the committed tree (and the submodule commit that HEAD records),
# so every run maps to a commit. runs/ and .venv on the instance survive a new push.
cmd_push() {
  local id=$1 rev ssm_rev
  [[ -z $(git -C "$REPO" status --porcelain) ]] || die "commit first: the instance gets HEAD"
  [[ -d $CKPT_DIR ]] || die "no checkpoints in $CKPT_DIR"
  rev=$(git -C "$REPO" rev-parse HEAD)
  ssm_rev=$(git -C "$REPO" rev-parse HEAD:third_party/ssm)
  wait_for_ssh "$id"
  remote "$id" "mkdir -p $REMOTE_DIR .cache/fbplan && find $REMOTE_DIR -mindepth 1 -maxdepth 1" \
    "! -name runs ! -name .venv -exec rm -rf {} +"
  git -C "$REPO" archive HEAD | remote "$id" "tar -x -C $REMOTE_DIR"
  git -C "$REPO/third_party/ssm" archive --prefix=third_party/ssm/ "$ssm_rev" |
    remote "$id" "tar -x -C $REMOTE_DIR"
  echo "$rev" | remote "$id" "cat > $REMOTE_DIR/REVISION"
  tar -c -C "$CKPT_DIR" . | remote "$id" "mkdir -p .cache/fbplan/checkpoints && tar -x -C .cache/fbplan/checkpoints"
  remote "$id" "bash $REMOTE_DIR/scripts/gpu/remote_setup.sh"
}

cmd_train() {
  local id=$1 env=$2
  shift 2
  cmd_run "$id" "train-$env" bash scripts/gpu/train_baselines.sh "$env" "$@"
}

# run <instance-id> <session> <command ...>: start a command in a detached tmux session.
# The command is escaped twice: once as tmux's command string, once for the remote shell, so
# an argument with spaces (a seed list) stays one argument.
cmd_run() {
  local id=$1 session=$2 command
  shift 2
  command=$(printf '%q ' "$@")
  remote "$id" "cd $REMOTE_DIR && tmux new-session -d -s $(printf %q "$session") $(printf %q "$command")"
  echo "started $session; follow with: $0 status $id"
}

cmd_status() {
  remote "$1" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv;" \
    "cd $REMOTE_DIR && for f in \$(find runs -name stdout.log | sort); do" \
    "echo \"\$f: \$(tail -n 1 \"\$f\" | cut -c1-100)\"; done; true"
}

# runs/ and the trained models; not the frozen checkpoints, which came from this machine.
cmd_pull() {
  local kind dest
  mkdir -p "$REPO/runs"
  remote "$1" "tar -c -C $REMOTE_DIR runs" | tar -x -C "$REPO"
  for kind in baselines approach_b; do
    dest=$BASELINE_DIR
    [[ $kind == approach_b ]] && dest=$APPROACH_B_DIR
    remote "$1" "test -d .cache/fbplan/$kind" 2>/dev/null || continue
    mkdir -p "$dest"
    remote "$1" "tar -c -C .cache/fbplan/$kind ." | tar -x -C "$dest"
  done
  echo "pulled into $REPO/runs, $BASELINE_DIR and $APPROACH_B_DIR"
}

cmd_ssh() {
  remote "$@"
}

# vastai asks for confirmation unless -y follows the id (needed when no terminal is attached).
cmd_destroy() {
  vastai destroy instance "$@"
}

cmd=${1:-}
case $cmd in
offers | list) "cmd_$cmd" ;;
create)
  [[ $# -eq 2 ]] || die "usage: $0 create <offer-id>"
  cmd_create "$2"
  ;;
push | train | run | status | pull | ssh | destroy)
  shift
  [[ $# -ge 1 ]] || die "usage: $0 $cmd <instance-id> ..."
  require_ours "$1"
  "cmd_$cmd" "$@"
  ;;
*)
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
  exit 1
  ;;
esac
