#!/usr/bin/env bash
# Run e-learner GPU jobs on a rented vast.ai instance. Run this script on your own machine.
#
# Needs: the vastai CLI (`uv tool install vastai`) with an API key (`vastai set api-key <key>`),
# and the frozen checkpoints in ~/.cache/fbplan/checkpoints (`uv run python -m fbplan.frozen`).
#
#   scripts/gpu/vast.sh offers                         suitable offers, cheapest first
#   scripts/gpu/vast.sh create <offer-id>              rent one with our label and SSH key
#   scripts/gpu/vast.sh list                           our instances (label $VAST_LABEL)
#   scripts/gpu/vast.sh push <instance-id>             code at HEAD + checkpoints, then remote setup
#   scripts/gpu/vast.sh train <instance-id> <env> [seed ...]   retrain pi_h in tmux (seeds 0-4)
#   scripts/gpu/vast.sh status <instance-id>           GPU use and the last log line of each run
#   scripts/gpu/vast.sh pull <instance-id>             runs/train_high and ~/.cache/fbplan/baselines
#   scripts/gpu/vast.sh ssh <instance-id> [command]    shell or one command on the instance
#   scripts/gpu/vast.sh destroy <instance-id>          delete the instance and its disk (stops billing)
#
# The API key belongs to a team account that holds other members' instances. Every command that
# takes an instance id refuses an instance whose label is not ours. The SSH key is attached to
# our instance only, never to the account.
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
REMOTE_DIR=fbplan
# Our instances carry a fixed label (VAST_LABEL); commands refuse any other label.
LABEL=${VAST_LABEL:-fbplan-gpu}
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

# require_ours <instance-id>: stop unless the instance exists and carries our label.
require_ours() {
  local label
  [[ $1 =~ ^[0-9]+$ ]] || die "instance id must be a number, got '$1'"
  label=$(vastai show instance "$1" --raw | jq -r '.label // ""') ||
    die "cannot read instance $1"
  [[ $label == "$LABEL" ]] || die "instance $1 has label '$label', not '$LABEL': not ours"
}

# remote <instance-id> [command ...]: run a command on the instance over SSH.
remote() {
  local url
  url=$(vastai ssh-url "$1") # ssh://root@HOST:PORT
  url=${url#ssh://}
  shift
  ssh "${SSH_OPTS[@]}" -p "${url##*:}" "${url%:*}" "$@"
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
    --label "$LABEL" --raw)
  id=$(jq -r '.new_contract // empty' <<<"$out")
  [[ -n $id ]] || die "create failed: $out"
  echo "created instance $id (label $LABEL); it bills until: $0 destroy $id"
  vastai attach ssh "$id" "$(cat "$SSH_PUBKEY")" ||
    die "could not attach $SSH_PUBKEY to instance $id; retry the attach or destroy it"
  echo "next: $0 push $id"
}

cmd_list() {
  vastai show instances --label "$LABEL"
}

# The instance gets exactly the committed tree (and the submodule commit that HEAD records),
# so every run maps to a commit. runs/ and .venv on the instance survive a new push.
cmd_push() {
  local id=$1 rev ssm_rev
  [[ -z $(git -C "$REPO" status --porcelain) ]] || die "commit first: the instance gets HEAD"
  [[ -d $HOME/.cache/fbplan/checkpoints ]] || die "no checkpoints in ~/.cache/fbplan/checkpoints"
  rev=$(git -C "$REPO" rev-parse HEAD)
  ssm_rev=$(git -C "$REPO" rev-parse HEAD:third_party/ssm)
  wait_for_ssh "$id"
  remote "$id" "mkdir -p $REMOTE_DIR .cache/fbplan && find $REMOTE_DIR -mindepth 1 -maxdepth 1" \
    "! -name runs ! -name .venv -exec rm -rf {} +"
  git -C "$REPO" archive HEAD | remote "$id" "tar -x -C $REMOTE_DIR"
  git -C "$REPO/third_party/ssm" archive --prefix=third_party/ssm/ "$ssm_rev" |
    remote "$id" "tar -x -C $REMOTE_DIR"
  echo "$rev" | remote "$id" "cat > $REMOTE_DIR/REVISION"
  tar -c -C "$HOME/.cache/fbplan" checkpoints | remote "$id" "tar -x -C .cache/fbplan"
  remote "$id" "bash $REMOTE_DIR/scripts/gpu/remote_setup.sh"
}

cmd_train() {
  local id=$1 env=$2
  shift 2
  local seeds=${*:-0 1 2 3 4}
  remote "$id" "cd $REMOTE_DIR && tmux new-session -d -s 'train-$env'" \
    "'bash scripts/gpu/train_baselines.sh $env $seeds'"
  echo "started; follow with: $0 status $id"
}

cmd_status() {
  remote "$1" "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv;" \
    "cd $REMOTE_DIR && for f in runs/train_high/*/seed*/stdout.log; do" \
    "[ -e \"\$f\" ] && echo \"\$f: \$(tail -n 1 \"\$f\")\"; done; true"
}

cmd_pull() {
  mkdir -p "$REPO/runs" "$HOME/.cache/fbplan"
  remote "$1" "tar -c -C $REMOTE_DIR runs/train_high" | tar -x -C "$REPO"
  remote "$1" "tar -c -C .cache/fbplan baselines" | tar -x -C "$HOME/.cache/fbplan"
  echo "pulled into $REPO/runs/train_high and ~/.cache/fbplan/baselines"
}

cmd_ssh() {
  remote "$@"
}

# vastai asks for confirmation before it destroys anything.
cmd_destroy() {
  vastai destroy instance "$1"
}

cmd=${1:-}
case $cmd in
offers | list) "cmd_$cmd" ;;
create)
  [[ $# -eq 2 ]] || die "usage: $0 create <offer-id>"
  cmd_create "$2"
  ;;
push | train | status | pull | ssh | destroy)
  shift
  [[ $# -ge 1 ]] || die "usage: $0 $cmd <instance-id> ..."
  require_ours "$1"
  "cmd_$cmd" "$@"
  ;;
*)
  sed -n '2,21p' "$0"
  exit 1
  ;;
esac
