#!/usr/bin/env bash
# Realtime interactive LLM interception for AgentEvolve (STEP 0).
#
# Usage:
#   ./docker/observability/proxy.sh up        # start interceptor
#   ./docker/observability/proxy.sh env       # print the vars to export
#   ./docker/observability/proxy.sh run -- <cmd>   # run <cmd> through the proxy
#   ./docker/observability/proxy.sh tail      # follow captured calls
#   ./docker/observability/proxy.sh down
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CA="$HERE/certs/mitmproxy-ca-cert.pem"
PROXY="http://127.0.0.1:8082"
UI="http://127.0.0.1:8083"

case "${1:-}" in
  up)
    docker compose -f "$HERE/compose.yml" up -d mitmproxy
    # The CA is generated on first boot; wait for it rather than racing it.
    for _ in $(seq 1 30); do [ -f "$CA" ] && break; sleep 1; done
    echo "proxy : $PROXY"
    echo "UI    : $UI   (password: agentevolve)"
    echo "CA    : $CA"
    ;;
  structured)
    # Adds the LLM-aware logging layer. Needs LITELLM_API_KEY/LITELLM_BASE_URL.
    docker compose -f "$HERE/compose.yml" --profile structured up -d
    echo "litellm: http://127.0.0.1:4000"
    ;;
  env)
    # SSL_CERT_FILE covers stdlib ssl; REQUESTS_CA_BUNDLE covers requests/httpx.
    # NO_PROXY keeps loopback services (searxng, CUGA's own HTTP) direct, so the
    # proxy only sees outbound model traffic.
    cat <<EOF
export HTTP_PROXY=$PROXY
export HTTPS_PROXY=$PROXY
export SSL_CERT_FILE=$CA
export REQUESTS_CA_BUNDLE=$CA
export NO_PROXY=localhost,127.0.0.1,::1
EOF
    ;;
  run)
    shift; [ "${1:-}" = "--" ] && shift
    [ -f "$CA" ] || { echo "CA missing; run '$0 up' first" >&2; exit 1; }
    HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY" \
    SSL_CERT_FILE="$CA" REQUESTS_CA_BUNDLE="$CA" \
    NO_PROXY="localhost,127.0.0.1,::1" \
      exec "$@"
    ;;
  tail)
    tail -f "$HERE/captures/calls.jsonl" 2>/dev/null | python3 -c '
import json,sys
for line in sys.stdin:
    try: r=json.loads(line)
    except ValueError: continue
    c=r.get("correlation",{})
    tag=" ".join(f"{k}={v}" for k,v in c.items()) or "-"
    m="MOCK["+str(r.get("mock_rule"))+"]" if r.get("mocked") else "live"
    print(f'"'"'#{r["seq"]:<4} {m:<22} {r["response"]["status"]} {r["duration_s"]}s  {tag}'"'"')
'
    ;;
  down) docker compose -f "$HERE/compose.yml" --profile structured down ;;
  *) sed -n '3,9p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
