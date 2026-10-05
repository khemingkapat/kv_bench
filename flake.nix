{
  description = "KV Cache Optimization & Benchmarking Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        slurmUser = "66070503408@cpe.kmutt.ac.th";
        slurmHost = "portal.slurm.cpe.kmutt.ac.th";

        slurm-ssh = pkgs.writeShellScriptBin "slurm-ssh" ''
          exec ssh -l ${slurmUser} ${slurmHost} "$@"
        '';

        ssh-slurm = pkgs.writeShellScriptBin "ssh-slurm" ''
          exec ssh -l ${slurmUser} ${slurmHost} "$@"
        '';

        slurm-scp = pkgs.writeShellScriptBin "slurm-scp" ''
          REMOTE="${slurmUser}@${slurmHost}"

          opts=()
          args=()

          while [ $# -gt 0 ]; do
            case "$1" in
              -P|-i|-o|-F|-l)
                opts+=("$1" "$2")
                shift 2
                ;;
              -*)
                opts+=("$1")
                shift
                ;;
              *)
                args+=("$1")
                shift
                ;;
            esac
          done

          if [ ''${#args[@]} -lt 2 ]; then
            echo "Error: Both source and destination must be specified." >&2
            echo "" >&2
            echo "Usage: slurm-scp [options] <source>... <destination>" >&2
            echo "" >&2
            echo "Examples:" >&2
            echo "  Upload:   slurm-scp main.py kv_bench/             # Uploads to remote ~/kv_bench/" >&2
            echo "  Upload:   slurm-scp file1.py file2.py kv_bench/   # Uploads multiple files" >&2
            echo "  Download: slurm-scp :kv_bench/output.txt .        # Downloads from remote ~/kv_bench/" >&2
            echo "  Download: slurm-scp :kv_bench/results/ ./results/ # Downloads directory" >&2
            exit 1
          fi

          # Check if any argument has an explicit remote prefix (:path, remote:path, slurm:path)
          has_remote=false
          for arg in "''${args[@]}"; do
            if [[ "$arg" == :* || "$arg" == remote:* || "$arg" == slurm:* || "$arg" == "$REMOTE:"* ]]; then
              has_remote=true
              break
            fi
          done

          final_args=()
          last_idx=$(( ''${#args[@]} - 1 ))

          for i in "''${!args[@]}"; do
            a="''${args[$i]}"
            if [[ "$a" == :* ]]; then
              final_args+=("$REMOTE:''${a#:}")
            elif [[ "$a" == remote:* ]]; then
              final_args+=("$REMOTE:''${a#remote:}")
            elif [[ "$a" == slurm:* ]]; then
              final_args+=("$REMOTE:''${a#slurm:}")
            elif [ "$has_remote" = false ] && [ "$i" -eq "$last_idx" ]; then
              # Default: if no remote prefix was used anywhere, treat destination as remote
              final_args+=("$REMOTE:$a")
            else
              final_args+=("$a")
            fi
          done

          exec scp -r "''${opts[@]}" "''${final_args[@]}"
        '';

        scp-slurm = pkgs.writeShellScriptBin "scp-slurm" ''
          exec slurm-scp "$@"
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          name = "kv-bench-shell";

          buildInputs = with pkgs; [
            python311
            uv
            git
            zlib
            zsh
            stdenv.cc.cc.lib
            slurm-ssh
            ssh-slurm
            slurm-scp
            scp-slurm
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
            
            # Ensure local .venv exists and activate it
            if [ ! -d ".venv" ]; then
              echo "Creating isolated uv environment in .venv..."
              uv venv .venv --python python3.11
            fi
            source .venv/bin/activate
            echo "🚀 KV Cache Benchmarking Shell Initialized (.venv active)"

            # Define shell aliases
            alias slurm-ssh='ssh -l ${slurmUser} ${slurmHost}'
            alias ssh-slurm='ssh -l ${slurmUser} ${slurmHost}'
            alias scp-slurm=slurm-scp

            echo "Available Shorthands:"
            echo "  slurm-ssh                     - SSH login to Slurm portal (ssh -l ${slurmUser} ${slurmHost})"
            echo "  slurm-scp <src>... <dest>     - SCP to/from Slurm (e.g. slurm-scp main.py kv_bench/ or slurm-scp :kv_bench/out.txt .)"
          '';
        };
      });
}
